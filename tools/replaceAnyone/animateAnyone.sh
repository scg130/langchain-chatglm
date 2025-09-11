#!/bin/bash
cd /usr/local/src

conda create -n animate_anyone python=3.10
conda activate animate_anyone

git clone https://github.com/MooreThreads/Moore-AnimateAnyone.git
cd Moore-AnimateAnyone

pip install --upgrade pip==24.0

pip install -r requirements.txt  

git clone https://github.com/emilianavt/OpenSeeFace.git  

python tools/download_weights.py

python app.py