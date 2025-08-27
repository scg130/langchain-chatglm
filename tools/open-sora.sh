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

# cuda 12.1
sudo apt --purge remove '*cuda*' 'nvidia*'
sudo apt autoremove -y
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update -y
sudo apt-get -y install cuda-12-1

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# create a virtual env and activate (conda as an example)
conda create -n opensora python=3.10 -y
conda activate opensora
conda install -c  conda-forge cmake  

# download the repo
git clone https://github.com/hpcaitech/Open-Sora
cd Open-Sora

# Ensure torch >= 2.4.0
pip install -v . 
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 
pip install flash-attn --no-build-isolation --no-cache-dir
pip install tensornvme
 
# download the model
pip install "huggingface_hub[cli]"
hf download hpcai-tech/Open-Sora-v2 --local-dir ./ckpts

# 恢复原文件
# mv opensora/utils/ckpt.py opensora/utils/ckpt.py.bak
# mv opensora/utils/ckpt.py.bak opensora/utils/ckpt.py

sed  -e "s/from[[:space:]]\+tensornvme\.async_file_io[[:space:]]\+import[[:space:]]\+\(AsyncFileWriter,\?[[:space:]]*AsyncFileReader\|DiskOffloader\)/from tensornvme import DiskOffloader/" \
    -e "s/\(AsyncFileWriter\|AsyncFileReader\)/DiskOffloader/g" \
    opensora/utils/ckpt.py   


sed -i '/from flash_attn import flash_attn_func as flash_attn_func_v2/c\
try:\
    from flash_attn import flash_attn_func as flash_attn_func_v2\
except ImportError:\
    print("⚠️ flash-attn not available, falling back to PyTorch attention")\
    import torch\
    def flash_attn_func_v2(q, k, v, *args, **kwargs):\
        attn = torch.softmax(q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5), dim=-1)\
        return attn @ v' opensora/models/mmdit/math.py



# run the inference text-to-video
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/t2i2v_768px.py --save-dir samples --prompt "raining, sea"

# run the inference text-to-video-with-image
torchrun --nproc_per_node 1 --standalone scripts/diffusion/inference.py configs/diffusion/inference/256px.py --cond_type i2v_head --prompt "A plump pig wallows in a muddy pond on a rustic farm, its pink snout poking out as it snorts contentedly. The camera captures the pig's playful splashes, sending ripples through the water under the midday sun. Wooden fences and a red barn stand in the background, framed by rolling green hills. The pig's muddy coat glistens in the sunlight, showcasing the simple pleasures of its carefree life." --ref assets/texts/i2v.png

