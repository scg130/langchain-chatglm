#!/bin/bash
sudo apt-get update --fix-missing
sudo apt install ffmpeg -y
sudo apt install libsox-dev -y
sudo apt install -y build-essential \
    libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    libgtk-3-dev libwebkit2gtk-4.0-dev libgtk-3-0 libgl1-mesa-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install libstdc++6
cd /usr/src/local
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

conda install conda-libmamba-solver --force-reinstall
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits #conda deactivate
pip install -r extra-req.txt --no-deps
pip install -r requirements.txt

mkdir -p GPT_SoVITS/pretrained_models/chinese-hubert-base/
mkdir -p GPT_SoVITS/pretrained_models/v2Pro/
mkdir -p GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/
wget -c -O GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Dv2Pro.pth?download=true"
wget -c -O GPT_SoVITS/pretrained_models/v2Pro/s2Dv2Pro.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/v2Pro/s2Gv2Pro.pth?download=true"

wget -c -O GPT_SoVITS/pretrained_models/s1v3.ckpt "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1v3.ckpt?download=true"

wget -c -O GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/config.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/config.json?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/pytorch_model.bin "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/pytorch_model.bin?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/tokenizer.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/tokenizer.json?download=true"

wget -c -O GPT_SoVITS/pretrained_models/chinese-hubert-base/config.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/config.json?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-hubert-base/pytorch_model.bin "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-hubert-base/preprocessor_config.json?download=true"
wget -c -O GPT_SoVITS/pretrained_models/chinese-hubert-base/tokenizer.json "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/chinese-roberta-wwm-ext-large/tokenizer.json?download=true"

wget -c -O GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch%3D12-step%3D369668.ckpt "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch%3D12-step%3D369668.ckpt?download=true"

wget -c -O GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2D2333k.pth?download=true"

wget -c -O GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/gsv-v2final-pretrained/s2G2333k.pth?download=true"

mkdir -p GPT_SoVITS/text && \
cd GPT_SoVITS/text && \
wget -O G2PWModel.zip "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/G2PWModel.zip" && \
unzip -o G2PWModel.zip && \
rm -f G2PWModel.zip && \
mv -f G2PWModel G2PWModel
cd ../..

export is_half=True
export is_share=True
python webui.py