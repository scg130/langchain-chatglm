#!/bin/bash

cd /usr/local/src
git clone https://github.com/jianchang512/clone-voice.git
cd clone-voice
python -m venv venv
source venv/bin/activate

sed -i 's/numpy==1.22.0/numpy>=1.24.3/g' requirements.txt

# 或者完全删除 numpy 版本限制
sed -i '/numpy==1.22.0/d' requirements.txt

pip install -r requirements.txt --no-deps
pip uninstall -y torch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2.0" scipy scikit-learn numba matplotlib gradio
# 然后安装TTS
pip install TTS

sed -i 's/^HTTP_PROXY=.*/HTTP_PROXY=http:\/\/127.0.0.1:8118/' /usr/local/src/clone-voice/.env

# download model
python  code_dev.py

python app.py

# cp  "$CLIENT_PY_PATH.backup" "$CLIENT_PY_PATH"