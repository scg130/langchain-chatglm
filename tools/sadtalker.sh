#!/bin/bash
cd /usr/local/src
git clone https://github.com/OpenTalker/SadTalker.git

cd SadTalker 

conda create -n sadtalker python=3.8

conda activate sadtalker

# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# conda install ffmpeg

pip install -r requirements.txt

### Coqui TTS is optional for gradio demo. 
pip install TTS

bash scripts/download_models.sh

python app_sadtalker.py

bash webui.sh