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

# 实时语音转换 GUI:

# python real-time-gui.py --checkpoint-path <path-to-checkpoint> --config-path <path-to-config>
# checkpoint 模型检查点路径，若为空将自动下载默认模型 (seed-uvit-tat-xlsr-tiny)
# config 模型配置文件路径，若为空将自动下载默认配置

python app_vc.py --checkpoint <path-to-checkpoint> --config <path-to-config> --fp16 True