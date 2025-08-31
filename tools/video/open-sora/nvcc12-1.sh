#!/bin/bash
# 1. 添加 NVIDIA 包仓库的密钥
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-12-4_12.4.0-1_amd64.deb
sudo dpkg -i cuda-12-4_12.4.0-1_amd64.deb
sudo apt update -y
sudo apt install cuda-12-4 -y 

# 4. 设置环境变量
echo 'export PATH=/usr/local/cuda-12-4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12-4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12-4' >> ~/.bashrc

source ~/.bashrc