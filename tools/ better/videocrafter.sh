#!/bin/bash
cd /usr/local/src


conda create -n videocrafter python=3.8.5 -y
conda activate videocrafter
git clone https://github.com/AILab-CVC/VideoCrafter.git
cd VideoCrafter
pip install --upgrade "pip<24.1"
pip install -r requirements.txt


# Install required packages
pip install huggingface-hub

# Download the model
huggingface-cli download --resume-download VideoCrafter/VideoCrafter2 --local-dir checkpoints/base_512_v2 --local-dir-use-symlinks False

pip install gradio==3.41.2

sh scripts/run_text2video.sh

huggingface-cli download --resume-download VideoCrafter/Image2Video-512 --local-dir checkpoints/i2v_512_v1 --local-dir-use-symlinks False

sh scripts/run_image2video.sh
