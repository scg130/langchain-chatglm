#!/bin/bash
cd /usr/local/src
sudo apt install libaio1 libaio-dev -y

# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.10 -y
conda install -c conda-forge cmake -y
conda activate opensora

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# Ensure torch >= 2.4.0
pip install -v . # for development mode, `pip install -v -e .`
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 # install xformers according to your cuda version
pip install flash-attn --no-build-isolation
pip install tensornvme

# cp opensora/utils/ckpt.py opensora/utils/ckpt.py.bak
# cp opensora/utils/ckpt.py.bak opensora/utils/ckpt.py

sed -i \
  -e 's/from[[:space:]]*tensornvme\.async_file_io[[:space:]]*import[[:space:]]*AsyncFileWriter/from tensornvme import DiskOffloader/' \
  -e 's/\bAsyncFileWriter\b/DiskOffloader/g' \
  opensora/utils/ckpt.py
pip install -v .


pip install "huggingface_hub[cli]"
hf download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

pip install modelscope
modelscope download hpcai-tech/Open-Sora-v2 --local_dir ./ckpts

# One GPU for 256px
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --prompt "raining, sea"
# Multi-GPU for 768px
torchrun --nproc_per_node 8 --standalone scripts/diffusion/inference.py configs/diffusion/inference/768px.py --prompt "raining, sea"

# 256px
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --cond_type i2v_head --prompt "A plump pig wallows in a muddy pond on a rustic farm, its pink snout poking out as it snorts contentedly. The camera captures the pig's playful splashes, sending ripples through the water under the midday sun. Wooden fences and a red barn stand in the background, framed by rolling green hills. The pig's muddy coat glistens in the sunlight, showcasing the simple pleasures of its carefree life." --ref assets/texts/i2v.png

# Multi-GPU 768px
torchrun --nproc_per_node 8 --standalone scripts/diffusion/inference.py configs/diffusion/inference/768px.py --cond_type i2v_head --dataset.data-path assets/texts/i2v.csv