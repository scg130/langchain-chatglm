#!/bin/bash
cd /usr/local/src
sudo apt update -y
sudo apt install -y libsentencepiece-dev protobuf-compiler libprotobuf-dev

git clone https://github.com/Lightricks/LTX-Video.git
cd LTX-Video

conda create -n ltxv python=3.11 -y
conda activate ltxv

python -m pip install -e .\[inference\]

# 高显存  ltxv-13b 默认30fps
python inference.py \
  --prompt "A dog standing up and wagging its tail, cinematic, high quality" \
  --conditioning_media_paths "/usr/local/src/LTX-Video/data/source/image1.jpg" \
  --conditioning_start_frames 0 \
  --height 512 \
  --width 768 \
  --num_frames 72 \
  --seed 12345 \
  --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml

# 低显存 ltxv-2b 默认24fps
python inference.py \
  --prompt "一条狗拿着一把枪 对着前方 正在开火" \
  --conditioning_media_paths "data/source/image1.jpg" "data/source/image2.jpg" "data/source/image3.jpg" \ # 多张图片，用空格分开
  --conditioning_start_frames 0,24,48 \ # 每张图片对应的开始帧，用逗号分开
  --height 512 \
  --width 768 \
  --num_frames 288 \
  --seed 12345 \
  --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml 

# 调整fps为30
ffmpeg -framerate 30 -i samples/frame_%05d.png -c:v libx264 -pix_fmt yuv420p out.mp4
