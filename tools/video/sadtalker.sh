#!/bin/bash

git clone https://github.com/OpenTalker/SadTalker.git
cd SadTalker 
conda create -n sadtalker python=3.9 -y
conda activate sadtalker
# install pytorch 2.0
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

pip install torchaudio

pip install addict future lmdb yapf tensorboard 

sed -i 's/basicsr==1\.4\.2//' requirements.txt
sed -i 's/gfpgan//' requirements.txt

pip install basicsr==1.4.2 --no-deps
pip install gfpgan==1.3.8 --no-deps
pip install -r requirements.txt


bash scripts/download_models.sh

python inference.py --driven_audio /Users/shemingdong/Downloads/googleDownload/output_mixed.mp3 \
                    --source_image /Users/shemingdong/Downloads/googleDownload/qz.png \
                    --result_dir /Users/shemingdong/Downloads/googleDownload \
                    --still \
                    --expression_scale 0.8 \
                    --preprocess full \
                    --enhancer gfpgan 