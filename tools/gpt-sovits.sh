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

# -------------------------------
# 创建模型目录
# -------------------------------
mkdir -p GPT_SoVITS/pretrained_models/chinese-hubert-base/
mkdir -p GPT_SoVITS/pretrained_models/v2Pro/
mkdir -p GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/
mkdir -p GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/

# -------------------------------
# 下载模型文件
# -------------------------------

# v2Pro
wget -c -O GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Dv2Pro.pth?download=true"
wget -c -O GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2Pro.pth?download=true"

# s1v3
wget -c -O GPT_SoVITS/pretrained_models/s1v3.ckpt "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt?download=true"

# chinese-roberta-wwm-ext-large
wget -c -O GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/config.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/config.json?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/pytorch_model.bin?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/tokenizer.json?download=true"

# chinese-hubert-base
wget -c -O GPT_SoVITS/pretrained_models/chinese-hubert-base/config.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/config.json?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/pytorch_model.bin?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-hubert-base/preprocessor_config.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/preprocessor_config.json?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-hubert-base/tokenizer.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/tokenizer.json?download=true"

# gsv-v2final-pretrained
wget -c -O GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch%3D12-step%3D369668.ckpt "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch%3D12-step%3D369668.ckpt?download=true"
wget -c -O GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2D2333k.pth?download=true"
wget -c -O GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth?download=true"

# G2PWModel
cd /usr/src/local/GPT-SoVITS/
mkdir -p GPT_SoVITS/text
cd GPT_SoVITS/text/
wget -O G2PWModel.zip "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip"
unzip -o G2PWModel.zip
rm -f G2PWModel.zip
cd ../..

# -------------------------------
# 环境变量
# -------------------------------
export is_half="True"
export is_share="True"

# -------------------------------
# 启动 WebUI
# -------------------------------
python webui.py
