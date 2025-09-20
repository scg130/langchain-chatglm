#!/bin/bash

# 更新系统
sudo apt update && sudo apt upgrade -y

# 创建 Conda 环境
conda create -n infiniteTalk python=3.10 -y
conda activate infiniteTalk

# 克隆 AIStarter 仓库
git clone https://github.com/AIStarter/AIStarter.git /usr/local/src/AIStarter
cd /usr/local/src/AIStarter

# 安装 Python 库
pip install --upgrade pip
pip install -r requirements.txt

# 下载 InfiniteTalk 模型
mkdir -p weights
cd weights
wget https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/resolve/main/Wan2.1-I2V-14B-480P.gguf
wget https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-F16/resolve/main/Wan2.1-I2V-14B-480P-F16.gguf
wget https://huggingface.co/TencentGameMate/chinese-wav2vec2-base/resolve/main/chinese-wav2vec2-base.pth
wget https://huggingface.co/MeiGen-AI/InfiniteTalk/resolve/main/InfiniteTalk.safetensors

# 启动 AIStarter
cd /usr/local/src/AIStarter
python app.py
