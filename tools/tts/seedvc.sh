#!/bin/bash 
cd /usr/local/src
git clone https://github.com/Plachtaa/seed-vc.git
cd seed-vc

conda create -n seed-vc python=3.11 -y
conda activate seed-vc
sudo apt install ffmpeg -y
pip install -r requirements.txt

# 运行命令前先设置环境变量: export export HUGGING_FACE_HUB_TOKEN={从https://huggingface.co/settings/tokens获取}

# 歌声转换 Web UI:

# python app_svc.py --checkpoint  --config 
# checkpoint 模型检查点路径，若为空将自动下载默认模型 (seed-uvit-whisper-base)
# config 模型配置文件路径，若为空将自动下载默认配置
# 集成 Web UI:
python app.py --enable-v1
# 此命令将仅加载预训练模型进行零样本推理。要使用自定义检查点，请按上述步骤运行 app_vc.py 或 app_svc.py。

# 合并人声伴奏
ffmpeg -i 1.wav -i music.mp3 -filter_complex "amix=inputs=2:duration=longest" -c:a libmp3lame -q:a 2 output_mixed.mp3

# 合并视频和伴奏
ffmpeg -i results/result.mp4 -i song.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest final_mv.mp4



# 实时语音转换 GUI:

# python real-time-gui.py --checkpoint-path <path-to-checkpoint> --config-path <path-to-config>
# checkpoint 模型检查点路径，若为空将自动下载默认模型 (seed-uvit-tat-xlsr-tiny)
# config 模型配置文件路径，若为空将自动下载默认配置

python app_vc.py --checkpoint <path-to-checkpoint> --config <path-to-config> --fp16 True

# 伪命令，具体以 seed-vc README 为准
python inference.py \
  --source /usr/local/src/ai-singer/source/song1.wav \
  --target /usr/local/src/ai-singer/source/zyt.mp3 \
  --output /usr/local/src/ai-singer/results \
  --diffusion-steps 40 \
  --length-adjust 1.0 \
  --inference-cfg-rate 0.9 \
  --f0-condition True \
  --auto-f0-adjust True \
  --semi-tone-shift 0 \
  --fp16 False

