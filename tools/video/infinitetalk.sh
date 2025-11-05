#!/bin/bash
set -e

# =============================
# 一键部署 InfiniteTalk 1.3B (conda 版)
# =============================

ENV_NAME="infinitalk"
REPO_URL="https://github.com/MeiGen-AI/InfiniteTalk.git"
MODEL_DIR="./weights/InfiniteTalk-1.3B"

echo ">>> 更新系统..."
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y git wget ffmpeg

# 检查 conda 是否存在
if ! command -v conda &> /dev/null; then
    echo "❌ 未检测到 conda，请先安装 Miniconda 或 Anaconda"
    exit 1
fi

# 初始化 conda shell
eval "$(conda shell.bash hook)"

# 创建并激活环境
if conda info --envs | grep -q "$ENV_NAME"; then
    echo ">>> 环境 $ENV_NAME 已存在，跳过创建"
else
    echo ">>> 创建环境 $ENV_NAME ..."
    conda create -y -n $ENV_NAME python=3.10
fi

conda activate $ENV_NAME

# 安装核心依赖
echo ">>> 安装 PyTorch (CUDA 12.1)"
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -U xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121

# 克隆仓库
if [ ! -d "InfiniteTalk" ]; then
    echo ">>> 克隆 InfiniteTalk 仓库..."
    git clone $REPO_URL
fi
cd InfiniteTalk

# 安装依赖
echo ">>> 安装 Python 依赖..."
wget -q https://raw.githubusercontent.com/MeiGen-AI/InfiniteTalk/main/requirements.txt -O requirements.txt
pip install -r requirements.txt

pip install misaki[en] ninja psutil packaging wheel flash_attn==2.7.4.post1
conda install -y -c conda-forge librosa ffmpeg

# 下载模型
echo ">>> 下载模型权重..."
mkdir -p weights

pip install "huggingface_hub[cli]"

# 语音识别模型
huggingface-cli download TencentGameMate/chinese-wav2vec2-base --local-dir weights/chinese-wav2vec2-base

# InfiniteTalk 模型
huggingface-cli download MeiGen-AI/InfiniteTalk --local-dir weights/InfiniteTalk

# 文生视频模型（可选）
# huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir weights/Wan2.1-I2V-14B-480P

# 启动服务
echo ">>> 启动 Gradio 界面..."
echo "访问地址: http://127.0.0.1:8418"

nohup python app.py \
    --ckpt_dir weights/Wan2.1-I2V-14B-480P \
    --wav2vec_dir 'weights/chinese-wav2vec2-base' \
    --infinitetalk_dir weights/InfiniteTalk/single/infinitetalk.safetensors \
    --num_persistent_param_in_dit 0 \
    --motion_frame 9 \
    --port 8418 > run.log 2>&1 &

echo "✅ InfiniteTalk 部署完成！使用命令查看日志："
echo "tail -f InfiniteTalk/run.log"
