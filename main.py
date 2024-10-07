import os
import cv2
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gaussian.render import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from gaussian.mesh import safe_normalize


class Gene3dContent:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None

        self.enable_sd = False


        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.gaussain_scale_factor = 1

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop

        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)

        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None:
            self.negative_prompt = self.opt.negative_prompt

        # override if provide a checkpoint
        if self.opt.load is not None:
            self.renderer.initialize(self.opt.load)
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0
        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # default camera
        if self.opt.mvdream or self.opt.imagedream:
            # the second view is the front view for mvdream/imagedream.
            pose = orbit_camera(self.opt.elevation, 90, self.opt.radius)
        else:
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )
        

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream

                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            elif self.opt.imagedream:
                print(f"[INFO] loading ImageDream...")
                from guidance.imagedream_utils import ImageDream

                self.guidance_sd = ImageDream(self.device)
                print(f"[INFO] loaded ImageDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion

                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")


        # input image
        if self.input_img is not None:
            self.input_img_torch = (
                torch.from_numpy(self.input_img)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )
            self.input_img_torch = F.interpolate(
                self.input_img_torch,
                (self.opt.ref_size, self.opt.ref_size),
                mode="bilinear",
                align_corners=False,
            )

            self.input_mask_torch = (
                torch.from_numpy(self.input_mask)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )
            self.input_mask_torch = F.interpolate(
                self.input_mask_torch,
                (self.opt.ref_size, self.opt.ref_size),
                mode="bilinear",
                align_corners=False,
            )

        # prepare embeddings
        with torch.no_grad():
            if self.enable_sd:
                if self.opt.imagedream:
                    self.guidance_sd.get_image_text_embeds(
                        self.input_img_torch, [self.prompt], [self.negative_prompt]
                    )
                else:
                    self.guidance_sd.get_text_embeds(
                        [self.prompt], [self.negative_prompt]
                    )

    def train_step(self):
        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters)

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)

            loss = 0

            ### known view
            if self.input_img_torch is not None and not self.opt.imagedream:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam)

                # rgb loss
                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                loss = loss + 10000 * (
                    step_ratio if self.opt.warmup_rgb_loss else 1
                ) * F.mse_loss(image, self.input_img_torch)

                # mask loss
                mask = out["alpha"].unsqueeze(0)  # [1, 1, H, W] in [0, 1]
                loss = loss + 1000 * (
                    step_ratio if self.opt.warmup_rgb_loss else 1
                ) * F.mse_loss(mask, self.input_mask_torch)

            ### novel view (manual batch)
            render_resolution = (
                128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            )
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [min_ver, max_ver]
            min_ver = max(
                min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation),
                -80 - self.opt.elevation,
            )
            max_ver = min(
                max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation),
                80 - self.opt.elevation,
            )

            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(
                    self.opt.elevation + ver, hor, self.opt.radius + radius
                )
                poses.append(pose)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )

                bg_color = torch.tensor(
                    (
                        [1, 1, 1]
                        if np.random.rand() > self.opt.invert_bg_prob
                        else [0, 0, 0]
                    ),
                    dtype=torch.float32,
                    device="cuda",
                )
                out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                images.append(image)

                # enable mvdream training
                if self.opt.mvdream or self.opt.imagedream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(
                            self.opt.elevation + ver,
                            hor + 90 * view_i,
                            self.opt.radius + radius,
                        )
                        poses.append(pose_i)

                        cur_cam_i = MiniCam(
                            pose_i,
                            render_resolution,
                            render_resolution,
                            self.cam.fovy,
                            self.cam.fovx,
                            self.cam.near,
                            self.cam.far,
                        )

                        # bg_color = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device="cuda")
                        out_i = self.renderer.render(cur_cam_i, bg_color=bg_color)

                        image = out_i["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)


            # guidance loss
            if self.enable_sd:
                if self.opt.mvdream or self.opt.imagedream:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(
                        images,
                        poses,
                        step_ratio=step_ratio if self.opt.anneal_timestep else None,
                    )
                else:
                    loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(
                        images,
                        step_ratio=step_ratio if self.opt.anneal_timestep else None,
                    )
                    
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # densify and prune
            if (
                self.step >= self.opt.density_start_iter
                and self.step <= self.opt.density_end_iter
            ):
                viewspace_point_tensor, visibility_filter, radii = (
                    out["viewspace_points"],
                    out["visibility_filter"],
                    out["radii"],
                )
                self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.renderer.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.renderer.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                if self.step % self.opt.densification_interval == 0:
                    self.renderer.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold,
                        min_opacity=0.01,
                        extent=4,
                        max_screen_size=1,
                    )

                if self.step % self.opt.opacity_reset_interval == 0:
                    self.renderer.gaussians.reset_opacity()

        torch.cuda.synchronize()


    def load_input(self, file):
        # load image
        print(f"[INFO] load image from {file}...")
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (1 - self.input_mask)
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f"[INFO] load prompt from {file_prompt}...")
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()

    @torch.no_grad()
    def save_model(self, mode="geo", texture_size=1024):
        os.makedirs(self.opt.outdir, exist_ok=True)
        if mode == "geo":
            path = os.path.join(self.opt.outdir, self.opt.save_path + "_mesh.ply")
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)
            mesh.write_ply(path)

        elif mode == "geo+tex":
            path = os.path.join(
                self.opt.outdir, self.opt.save_path + "_mesh." + self.opt.mesh_format
            )
            mesh = self.renderer.gaussians.extract_mesh(path, self.opt.density_thresh)

            # perform texture extraction
            print(f"[INFO] unwrap uv...")
            h = w = texture_size
            mesh.auto_uv()
            mesh.auto_normal()

            albedo = torch.zeros((h, w, 3), device=self.device, dtype=torch.float32)
            cnt = torch.zeros((h, w, 1), device=self.device, dtype=torch.float32)

            # self.prepare_train() # tmp fix for not loading 0123
            # vers = [0]
            # hors = [0]
            vers = [0] * 8 + [-45] * 8 + [45] * 8 + [-89.9, 89.9]
            hors = [0, 45, -45, 90, -90, 135, -135, 180] * 3 + [0, 0]

            render_resolution = 512

            import nvdiffrast.torch as dr

            if not self.opt.force_cuda_rast:
                glctx = dr.RasterizeGLContext()
            else:
                glctx = dr.RasterizeCudaContext()

            for ver, hor in zip(vers, hors):
                # render image
                pose = orbit_camera(ver, hor, self.cam.radius)

                cur_cam = MiniCam(
                    pose,
                    render_resolution,
                    render_resolution,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )

                cur_out = self.renderer.render(cur_cam)

                rgbs = cur_out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]

                # get coordinate in texture image
                pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
                proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).to(
                    self.device
                )

                v_cam = (
                    torch.matmul(
                        F.pad(mesh.v, pad=(0, 1), mode="constant", value=1.0),
                        torch.inverse(pose).T,
                    )
                    .float()
                    .unsqueeze(0)
                )
                v_clip = v_cam @ proj.T
                rast, rast_db = dr.rasterize(
                    glctx, v_clip, mesh.f, (render_resolution, render_resolution)
                )

                depth, _ = dr.interpolate(
                    -v_cam[..., [2]], rast, mesh.f
                )  # [1, H, W, 1]
                depth = depth.squeeze(0)  # [H, W, 1]

                alpha = (rast[0, ..., 3:] > 0).float()

                uvs, _ = dr.interpolate(
                    mesh.vt.unsqueeze(0), rast, mesh.ft
                )  # [1, 512, 512, 2] in [0, 1]

                # use normal to produce a back-project mask
                normal, _ = dr.interpolate(
                    mesh.vn.unsqueeze(0).contiguous(), rast, mesh.fn
                )
                normal = safe_normalize(normal[0])

                # rotated normal (where [0, 0, 1] always faces camera)
                rot_normal = normal @ pose[:3, :3]
                viewcos = rot_normal[..., [2]]

                mask = (alpha > 0) & (viewcos > 0.5)  # [H, W, 1]
                mask = mask.view(-1)

                uvs = uvs.view(-1, 2).clamp(0, 1)[mask]
                rgbs = rgbs.view(3, -1).permute(1, 0)[mask].contiguous()

                # update texture image
                cur_albedo, cur_cnt = mipmap_linear_grid_put_2d(
                    h,
                    w,
                    uvs[..., [1, 0]] * 2 - 1,
                    rgbs,
                    min_resolution=256,
                    return_count=True,
                )

                # albedo += cur_albedo
                # cnt += cur_cnt
                mask = cnt.squeeze(-1) < 0.1
                albedo[mask] += cur_albedo[mask]
                cnt[mask] += cur_cnt[mask]

            mask = cnt.squeeze(-1) > 0
            albedo[mask] = albedo[mask] / cnt[mask].repeat(1, 3)

            mask = mask.view(h, w)

            albedo = albedo.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # dilate texture
            from sklearn.neighbors import NearestNeighbors
            from scipy.ndimage import binary_dilation, binary_erosion

            inpaint_region = binary_dilation(mask, iterations=32)
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(
                search_coords
            )
            _, indices = knn.kneighbors(inpaint_coords)

            albedo[tuple(inpaint_coords.T)] = albedo[
                tuple(search_coords[indices[:, 0]].T)
            ]

            mesh.albedo = torch.from_numpy(albedo).to(self.device)
            mesh.write(path)

        else:
            path = os.path.join(self.opt.outdir, self.opt.save_path + "_model.ply")
            self.renderer.gaussians.save_ply(path)

        print(f"[INFO] save model to {path}.")

    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
            # do a last prune
            self.renderer.gaussians.prune(min_opacity=0.01, extent=1, max_screen_size=1)
        # save
        self.save_model(mode="model")
        self.save_model(mode="geo+tex")


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = Gene3dContent(opt)
    gui.train(opt.iters)
