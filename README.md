# 3d_pipeline
use for test 

## 训练
输入 一系列图像
降维(矩阵大小) -> 升维(3d) -> render -> loss

## 依赖加载

模型下载
 huggingface-cli download --resume-download stabilityai/stable-diffusion-2-1  --local-dir . --local-dir-use-symlinks False --resume-download --revision bf16


## 运行
git clone https://github.com/secondxworld/3d_pipeline.git --recursive 

[colab](https://colab.research.google.com/drive/1Ti_3cqPCU2SWlJHTYUDU3AJPxxtqM8TD#scrollTo=hB256Ln-cQP9)

本地
python main.py --config configs/text.yaml prompt="a photo of an icecream" save_path=icecream

kire logs/icecream_mesh.obj --save logs --wogui



code from [dreamgaussian](https://github.com/dreamgaussian/dreamgaussian.git) 