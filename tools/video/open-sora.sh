#!/bin/bash

cd /usr/local/src
# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.10
conda activate opensora

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# Ensure torch >= 2.4.0
pip install -v . # for development mode, `pip install -v -e .`
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 # install xformers according to your cuda version
pip install flash-attn --no-build-isolation

 # download the model
pip install "huggingface_hub[cli]"
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

# run the inference text-to-video
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_768px.py --save-dir samples --prompt "raining, sea"

# run the inference text-to-video-with-image
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --cond_type i2v_head --prompt "A plump pig wallows in a muddy pond on a rustic farm, its pink snout poking out as it snorts contentedly. The camera captures the pig's playful splashes, sending ripples through the water under the midday sun. Wooden fences and a red barn stand in the background, framed by rolling green hills. The pig's muddy coat glistens in the sunlight, showcasing the simple pleasures of its carefree life." --ref assets/texts/i2v.png

