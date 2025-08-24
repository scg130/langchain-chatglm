#!/bin/bash
cd /usr/local/src
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm

conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm

pip install -r requirements.txt


export HF_ENDPOINT="https://hf-mirror.com"

# 安装 huggingface-hub
pip install huggingface-hub

huggingface-cli download \
    --repo-type model \
    --local-dir "assets/checkpoints/" \
    --local-dir-use-symlinks False \
    --resume-download \
    "IndexTeam/IndexTTS-1.5"

bash convert_hf_format.sh /path/to/your/model_dir

VLLM_USE_V1=0 python webui.py

VLLM_USE_V1=0 python api_server.py --model_dir /your/path/to/Index-TTS --port 11996