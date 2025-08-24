#!/bin/bash
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo
cd HunyuanVideo

conda create -n HunyuanVideo python==3.10.9

conda activate HunyuanVideo

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt

pip install ninja

sudo apt install cuda-toolkit-12-4=12.4.0-1

pip install xfuser
pip install loguru
pip install flash-attention
pip install --upgrade gradio

# hf login


pip install "huggingface_hub[cli]"
HF_ENDPOINT=https://hf-mirror.com  hf download tencent/HunyuanVideo --local-dir ./ckpts

cd ckpts

huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2


cd ../

python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export GRADIO_SHARE=True

GRADIO_ANALYTICS_ENABLED=True SERVER_NAME=0.0.0.0 SERVER_PORT=8800 python gradio_server.py --flow-reverse