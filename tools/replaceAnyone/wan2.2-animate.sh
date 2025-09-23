#!/bin/bash
cd /usr/local/src
git clone https://github.com/Wan-Video/Wan2.2.git
cd Wan2.2

conda create -n wan22 python=3.11 -y
conda activate wan22


pip install -r requirements.txt
pip install flash-attn --no-build-isolation
# If you want to use CosyVoice to synthesize speech for Speech-to-Video Generation, please install requirements_s2v.txt additionally
pip install -r requirements_s2v.txt


pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B


python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --audio "examples/talk.wav"
# Without setting --num_clip, the generated video length will automatically adjust based on the input audio length

# You can use CosyVoice to generate audio with --enable_tts
python generate.py  --task s2v-14B --size 1024*704 --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard."  --image "examples/i2v_input.JPG" --enable_tts --tts_prompt_audio "examples/zero_shot_prompt.wav" --tts_prompt_text "希望你以后能够做的比我还好呦。" --tts_text "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"

# 生成视频帧
python ./wan/modules/animate/preprocess/preprocess_data.py \
    --ckpt_path ./Wan2.2-Animate-14B/process_checkpoint \
    --video_path ./examples/wan_animate/animate/video.mp4 \
    --refer_path ./examples/wan_animate/animate/image.jpeg \
    --save_path ./examples/wan_animate/animate/process_results \
    --resolution_area 1280 720 \
    --retarget_flag \
    --use_flux

# 合成视频
python generate.py \
    --task animate-14B \
    --ckpt_dir ./Wan2.2-Animate-14B \
    --src_root_path ./examples/wan_animate/animate/process_results \
    --refert_num 1 \
    --bs 2 \
    --max_frames 128 \
    --resolution 1280 720 \
    --dtype bf16 \
    --enable_flash_attention \
    --use_xformers