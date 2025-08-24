#!/bin/bash

cd /usr/local/src
git clone https://github.com/index-tts/index-tts.git
cd index-tts

conda create -n index-tts python=3.11
conda activate index-tts
apt-get install ffmpeg

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118


pip install -e .

export HF_ENDPOINT="https://hf-mirror.com"

huggingface-cli download IndexTeam/IndexTTS-1.5 \
  config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints


python webui.py --model_dir IndexTTS-1.5  