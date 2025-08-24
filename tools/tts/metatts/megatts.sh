#!/bin/bash
cd /usr/local/src
# Clone the repository
git clone https://github.com/bytedance/MegaTTS3
cd MegaTTS3

conda create -n megatts python=3.11 -y
conda activate megatts

pip install -r requirements.txt
sudo apt install ffmpeg -y
huggingface-cli download ByteDance/MegaTTS3 --local-dir ./checkpoints --local-dir-use-symlinks False

# Set the root directory
export PYTHONPATH="/usr/local/src/MegaTTS3:$PYTHONPATH"

# [Optional] Set GPU
export CUDA_VISIBLE_DEVICES=0


python tts/gradio_api.py