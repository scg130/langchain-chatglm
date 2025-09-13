#!/bin/bash
cd /usr/local/src

git clone https://github.com/WeChatCV/Stand-In.git
cd Stand-In

conda create -n standin python=3.11 -y
conda activate standin
pip install -r requirements.txt


# 下载 NVIDIA 官方源配置包
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/cuda-ubuntu$(lsb_release -rs | tr -d .).pin
# 移动到 sources.list.d 目录
sudo mv cuda-ubuntu$(lsb_release -rs | tr -d .).pin /etc/apt/preferences.d/cuda-repository-pin-600
# 添加 GPG key
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/3bf863cc.pub
# 添加源
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/ /"
sudo apt update -y

sudo apt install -y cuda-toolkit-11-8
sudo rm -f /usr/local/cuda
sudo ln -s /usr/local/cuda-11.8 /usr/local/cuda

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pip install flash-attn --no-build-isolation

python download_models.py

python download_models.py --vace


# 建议至少 24GB 显存（A100、4090），不然容易 OOM。
# 显存不足时可以：
# 降低输出分辨率（默认 720p，可改 512p）
# 开启 --precision half 或者 --use_flash_attn

# --prompt：生成视频的文本描述（这里是「A woman raises her hands.」）。
# --vace_path：Wan2.1 VACE 模型权重的路径，通常在 checkpoints/VACE/。
# --ip_image：身份保持的输入图像（用来锁定人物 ID）。
# --reference_video：驱动视频，用来提供动作 / 姿态。
# --reference_image：参考图像（可以和 ip_image 一样，作为 identity 的补充）。
# --output：最终生成的视频文件路径。
# --vace_scale：权重平衡参数，控制 身份保持 和 动作一致性 的强弱（0.5 ~ 0.8 一般比较合适）。

python infer_with_vace.py \
    --prompt "A woman raises her hands." \
    --vace_path "checkpoints/VACE/" \
    --ip_image "test/input/first_frame.png" \
    --reference_video "test/input/pose.mp4" \
    --reference_image "test/input/first_frame.png" \
    --output "test/output/woman.mp4" \
    --vace_scale 0.8
