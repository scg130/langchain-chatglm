#!/bin/bash

# =============================
# 一键部署 InfiniteTalk 1.3B (conda 版)
# =============================

# 配置
ENV_NAME="infinitalk"
MODEL_DIR="./weights/InfiniteTalk-1.3B"
REPO_URL="https://github.com/MeiGen-AI/InfiniteTalk.git"

# 更新系统
echo ">>> 更新系统..."
sudo apt update && sudo apt upgrade -y

# 安装 git、wget、conda（如果未安装）
echo ">>> 安装 git、wget..."
sudo apt install -y git wget

# 检查 conda 是否存在
if ! command -v conda &> /dev/null
then
    echo "conda 未安装，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 创建 conda 环境
echo ">>> 创建 conda 环境: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# 安装依赖
echo ">>> 安装 Python 依赖"
pip install --upgrade pip
pip install -r <(curl -s https://raw.githubusercontent.com/MeiGen-AI/InfiniteTalk/main/requirements.txt)

# 克隆 InfiniteTalk 仓库
echo ">>> 克隆 InfiniteTalk 仓库..."
git clone $REPO_URL
cd InfiniteTalk || exit

pip install misaki[en]
pip install ninja 
pip install psutil 
pip install packaging
pip install wheel
pip install flash_attn==2.7.4.post1

pip install -r requirements.txt
conda install -c conda-forge librosa

conda install -c conda-forge ffmpeg

# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./weights/Wan2.1-I2V-14B-480P
# huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir ./weights/chinese-wav2vec2-base
# huggingface-cli download TencentGameMate/chinese-wav2vec2-base model.safetensors --revision refs/pr/1 --local-dir ./weights/chinese-wav2vec2-base
# huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir ./weights/InfiniteTalk


# 创建模型目录
mkdir -p $MODEL_DIR

pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ${MODEL_DIR}


# 启动 Gradio 界面
echo ">>> 启动 Gradio Web 界面..."
echo "访问 http://127.0.0.1:8418 进行使用"
python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9 