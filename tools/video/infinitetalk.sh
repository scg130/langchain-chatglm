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

# 安装依赖
echo ">>> 安装 Python 依赖"
pip install --upgrade pip
pip install -r <(curl -s https://raw.githubusercontent.com/MeiGen-AI/InfiniteTalk/main/requirements.txt)

# 克隆 InfiniteTalk 仓库
echo ">>> 克隆 InfiniteTalk 仓库..."
git clone $REPO_URL
cd InfiniteTalk || exit

# 创建模型目录
mkdir -p $MODEL_DIR

# 提示下载模型
echo ">>> 请将 InfiniteTalk 1.3B 模型权重放入 $MODEL_DIR"
echo "如果已有模型，请确保路径正确"

# 启动 Gradio 界面
echo ">>> 启动 Gradio Web 界面..."
echo "访问 http://127.0.0.1:7860 进行使用"
python app.py
