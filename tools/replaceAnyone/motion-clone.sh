#!/bin/bash
cd /usr/local/src
git clone https://github.com/LPengYang/MotionClone.git
cd MotionClone

conda env create -f environment.yaml
conda activate motionclone


git lfs install
git clone https://huggingface.co/botp/stable-diffusion-v1-5 models/StableDiffusion/

# Perform Text-to-video generation with customized camera motion
python t2v_video_sample.py --inference_config "configs/t2v_camera.yaml" --examples "configs/t2v_camera.jsonl"
# Perform Text-to-video generation with customized object motion
python t2v_video_sample.py --inference_config "configs/t2v_object.yaml" --examples "configs/t2v_object.jsonl"
# Combine motion cloning with sketch-to-video
python i2v_video_sample.py --inference_config "configs/i2v_sketch.yaml" --examples "configs/i2v_sketch.jsonl"
# Combine motion cloning with image-to-video
python i2v_video_sample.py --inference_config "configs/i2v_rgb.yaml" --examples "configs/i2v_rgb.jsonl"
