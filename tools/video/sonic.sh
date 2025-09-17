#!/bin/bash
cd /usr/local/src
git clone https://github.com/jixiaozhong/Sonic.git
cd Sonic

conda create -n Sonic python=3.10 -y  
conda activate Sonic 

pip install -r requirements.txt
pip install accelerate

pip install numpy==1.26.4 opencv-python==4.8.0.74

pip install "huggingface_hub[cli]"

huggingface-cli download LeonJoe13/Sonic --local-dir  checkpoints
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --local-dir  checkpoints/stable-video-diffusion-img2vid-xt
huggingface-cli download openai/whisper-tiny --local-dir checkpoints/whisper-tiny



python demo.py \
'data/qz.png' \
'data/output_mixed.mp3' \
'data/output_video.mp4'