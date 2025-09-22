#!/bin/bash
set -e

# 进入源码目录
cd /usr/local/src
if [ ! -d "MotionClone" ]; then
  git clone https://github.com/LPengYang/MotionClone.git
fi
cd MotionClone

echo ">>> 创建并激活 conda 环境"
conda env create -f environment.yaml || true
source activate motionclone || conda activate motionclone


# ================= 必需模型 =================
echo ">>> 下载 Stable Diffusion v1-5"
mkdir -p models/StableDiffusion
huggingface-cli download runwayml/stable-diffusion-v1-5 \
  --local-dir models/StableDiffusion \
  --local-dir-use-symlinks False

echo ">>> 下载 Motion Module (AnimateDiff v3)"
mkdir -p models/Motion_Module
wget -O models/Motion_Module/v3_sd15_mm.ckpt \
  https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt

# ================= 可选模型 =================
echo ">>> 下载 Realistic Vision v6.0 (DreamBooth/LoRA，可选)"
mkdir -p models/DreamBooth_LoRA
wget -O models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors "https://huggingface.co/moiu2998/mymo/resolve/3c3093fa083909be34a10714c93874ce5c9dabc4/realisticVisionV60B1_v51VAE.safetensors?download=true"

echo ">>> 下载 VAE (可选)"
mkdir -p models/VAE
wget -O models/VAE/vae-ft-mse-840000-ema-pruned.safetensors \
  https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors

echo ">>> 下载 ControlNet 模型 (可选)"
mkdir -p models/ControlNet
huggingface-cli download lllyasviel/ControlNet-v1-1 \
  --local-dir models/ControlNet \
  --local-dir-use-symlinks False

# ================= 运行示例 =================
echo ">>> 运行 MotionClone 示例"

# 文本转视频（自定义相机运动）
python t2v_video_sample.py --inference_config "configs/t2v_camera.yaml" --examples "configs/t2v_camera.jsonl"

# 文本转视频（自定义物体运动）
python t2v_video_sample.py --inference_config "configs/t2v_object.yaml" --examples "configs/t2v_object.jsonl"

# 草图转视频 + 动作克隆
python i2v_video_sample.py --inference_config "configs/i2v_sketch.yaml" --examples "configs/i2v_sketch.jsonl"

# 图像转视频 + 动作克隆
python i2v_video_sample.py --inference_config "configs/i2v_rgb.yaml" --examples "configs/i2v_rgb.jsonl"

echo "✅ MotionClone 全部流程完成！"
