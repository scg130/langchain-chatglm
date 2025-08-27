#!/bin/bash
cd /usr/local/src
git clone https://github.com/axboe/liburing.git
cd liburing
make
sudo make install
sudo ldconfig 

cd /usr/local/src

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository ppa:axboe/liburing
sudo apt update -y
sudo apt install libaio1 build-essential -y

# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.10 -y
conda activate opensora
conda install -c conda-forge cudatoolkit=12.2 -y
conda install -c  conda-forge cmake  -y 

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# Ensure torch >= 2.4.0
pip install -v . 
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 
pip install flash-attn --no-build-isolation --no-cache-dir
pip install tensornvme
pip install colossalai 
 
# download the model
pip install "huggingface_hub[cli]"
hf download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

# 恢复原文件
# cp opensora/utils/ckpt.py opensora/utils/ckpt.py.bak
# cp opensora/utils/ckpt.py.bak opensora/utils/ckpt.py

sed -i \
  -e 's/from[[:space:]]*tensornvme\.async_file_io[[:space:]]*import[[:space:]]*AsyncFileWriter/from tensornvme import DiskOffloader/' \
  -e 's/\bAsyncFileWriter\b/DiskOffloader/g' \
  opensora/utils/ckpt.py

sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab



# run the inference text-to-video
torchrun --nproc_per_node 1  --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_768px.py --save-dir samples --prompt "raining, sea"

# run the inference text-to-video-with-image
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --cond_type i2v_head --prompt "A plump pig wallows in a muddy pond on a rustic farm, its pink snout poking out as it snorts contentedly. The camera captures the pig's playful splashes, sending ripples through the water under the midday sun. Wooden fences and a red barn stand in the background, framed by rolling green hills. The pig's muddy coat glistens in the sunlight, showcasing the simple pleasures of its carefree life." --ref assets/texts/i2v.png

