#!/bin/bash
set -e

# -------------------------------
# 系统依赖
# -------------------------------
sudo apt-get update --fix-missing
sudo apt-get install -y ffmpeg libsox-dev build-essential \
    libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    libgtk-3-dev libwebkit2gtk-4.0-dev libgtk-3-0 libgl1-mesa-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libstdc++6 unzip wget git

# -------------------------------
# 下载 GPT-SoVITS 仓库
# -------------------------------
cd /usr/local/src/
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# -------------------------------
# Conda 环境
# -------------------------------
rm -rf /root/anaconda3/envs/GPTSoVits
conda install -y conda-libmamba-solver --force-reinstall
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n GPTSoVits python=3.11 -y
conda activate GPTSoVits

# 安装 Python 依赖
pip install --no-deps -r extra-req.txt
pip install -r requirements.txt
pip install onnxruntime==1.18

pip install ipykernel
sudo sed -i 's/wget --tries=25/wget -c --tries=25/g' install.sh
bash install.sh --device CU126 --source HF --download-uvr5
 
# -------------------------------
# 环境变量
# -------------------------------
export is_half="True"
export is_share="True"

# -------------------------------
# 启动 WebUI
# -------------------------------
python webui.py
