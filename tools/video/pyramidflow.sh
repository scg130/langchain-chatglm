#!/bin/bash
cd /usr/local/src

git clone https://github.com/jy0205/Pyramid-Flow
cd /usr/local/src/Pyramid-Flow


conda create -n pyramid python==3.8.10 -y
conda activate pyramid

pip install -r requirements.txt

pip install gradio

pip install --upgrade timm

pip install "huggingface_hub[cli]"

HF_ENDPOINT=https://hf-mirror.com hf download rain1011/pyramid-flow-miniflux --local-dir /usr/local/src/Pyramid-Flow  --repo-type model

mv /usr/local/src/Pyramid-Flow/diffusion_transformer_768p/diffusion_pytorch_model.safetensors /usr/local/src/Pyramid-Flow/pyramid_flow_model/diffusion_transformer_768p/

# 移动配置文件
mv /usr/local/src/Pyramid-Flow/diffusion_transformer_768p/config.json /usr/local/src/Pyramid-Flow/pyramid_flow_model/diffusion_transformer_768p/

cd /usr/local/src/Pyramid-Flow/pyramid_flow_model/diffusion_transformer_image/
ln -sf ../../diffusion_transformer_image/diffusion_pytorch_model.safetensors .
ln -sf ../../diffusion_transformer_image/config.json .

# 为 text encoder 创建链接
cd /usr/local/src/Pyramid-Flow/pyramid_flow_model/text_encoder/
ln -sf ../../text_encoder/model.safetensors .
ln -sf ../../text_encoder/config.json .

# 为 text_encoder_2 创建链接
cd /usr/local/src/Pyramid-Flow/pyramid_flow_model/text_encoder_2/
ln -sf ../../text_encoder_2/model.safetensors.index.json .
ln -sf ../../text_encoder_2/model-00001-of-00002.safetensors .
ln -sf ../../text_encoder_2/model-00002-of-00002.safetensors .
ln -sf ../../text_encoder_2/config.json .

# 进入 384p 模型目录
cd /usr/local/src/Pyramid-Flow/pyramid_flow_model/diffusion_transformer_384p/

# 创建从 .safetensors 到 .bin 的符号链接
ln -sf ../../diffusion_transformer_384p/diffusion_pytorch_model.safetensors diffusion_pytorch_model.bin

# 确保配置文件也存在
ln -sf ../../diffusion_transformer_384p/config.json .

cd /usr/local/src/Pyramid-Flow/

sed -i 's/demo.launch(share=False)/demo.launch(share=True)/g' /usr/local/src/Pyramid-Flow/app.py

python app.py