#!/bin/sh
cd /usr/local/src
# git clone --depth=1 -b stable https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI

git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
apt -y install -qq aria2
cd Retrieval-based-Voice-Conversion-WebUI
git pull
mkdir -p pretrained uvr5_weights
# python -m venv venv
# source venv/bin/activate
pip install --upgrade pip setuptools numpy numba
pip install gradio==3.50.2
pip install -r requirements.txt

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o D32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o D40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/D48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o D48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o G32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o G40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/G48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o G48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o f0D32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o f0D40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0D48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o f0D48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o f0G32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o f0G40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2/f0G48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained_v2 -o f0G48k.pth

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o D32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o D40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o D48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o G32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o G40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o G48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o f0D32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o f0D40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o f0D48k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o f0G32k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o f0G40k.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/pretrained -o f0G48k.pth


aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/uvr5_weights -o HP2-人声vocals+非人声instrumentals.pth
aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/uvr5_weights -o HP5-主旋律人声vocals+其他instrumentals.pth

aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/hubert -o hubert_base.pt


aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt -d /usr/local/src/Retrieval-based-Voice-Conversion-WebUI/assets/rmvpe -o rmvpe.pt

mkdir -p logs/pdd
cp assets/hubert/hubert_base.pt logs/pdd/

python infer-web.py


