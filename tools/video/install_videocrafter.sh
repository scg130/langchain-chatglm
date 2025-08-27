#!/bin/bash
cd /usr/local/src


conda create -n videocrafter python=3.8.5
conda activate videocrafter
git clone https://github.com/AILab-CVC/VideoCrafter.git
pip install -r requirements.txt


# Install required packages
pip install huggingface-hub

# Download the model
huggingface-cli download --resume-download VideoCrafter/VideoCrafter2 --local-dir checkpoints/base_512_v2 --local-dir-use-symlinks False

sh scripts/run_text2video.sh

sh scripts/run_image2video.sh

python gradio_app.py