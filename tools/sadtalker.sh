#!/bin/bash

git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker 
conda create -n sadtalker python=3.8
conda activate sadtalker
# install pytorch 2.0
pip install torch torchvision torchaudio

pip install -r requirements.txt
pip install dlib # macOS needs to install the original dlib.
pip install gfpgan
pip install opencv-python==4.8.1.78
pip install basicsr-fixed 

bash scripts/download_models.sh

python inference.py --driven_audio /Users/shemingdong/Downloads/googleDownload/output_mixed.mp3 \
                    --source_image /Users/shemingdong/Downloads/googleDownload/qz.png \
                    --result_dir /Users/shemingdong/Downloads/googleDownload \
                    --still \
                    --preprocess full \
                    --enhancer gfpgan 