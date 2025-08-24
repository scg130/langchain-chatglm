#!/bin/bash
cd /usr/local/src
sudo apt update
sudo apt install -y libsentencepiece-dev protobuf-compiler libprotobuf-dev

git clone https://github.com/Lightricks/LTX-Video.git
cd LTX-Video

python -m venv venv
source venv/bin/activate
python -m pip install -e .\[inference\]

# 高显存
python inference.py \
  --prompt "A dog standing up and wagging its tail, cinematic, high quality" \
  --conditioning_media_paths "/usr/local/src/LTX-Video/dog.jpg" \
  --conditioning_start_frames 0 \
  --height 512 \
  --width 768 \
  --num_frames 72 \
  --seed 12345 \
  --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml

# 低显存
python inference.py \
  --prompt "一条狗拿着一把枪 对着前方 正在开火" \
  --conditioning_media_paths "path/to/image1.jpg" "path/to/image2.jpg" "path/to/image3.jpg" \ # 多张图片，用空格分开
  --conditioning_start_frames 0,24,48 \ # 每张图片对应的开始帧，用逗号分开
  --height 512 \
  --width 768 \
  --num_frames 288 \
  --seed 12345 \
  --pipeline_config configs/ltxv-2b-0.9.8-distilled.yaml 