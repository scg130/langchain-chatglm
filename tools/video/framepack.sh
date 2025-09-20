#!/bin/bash
cd /usr/local/src
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack

conda create -n framepack python=3.10 -y
conda activate framepack

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install sageattention==1.0.6

# Note that it supports --share, --port, --server, and so on.
python demo_gradio.py --share