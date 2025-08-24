#!/bin/bash
cd /usr/local/src
git clone https://github.com/myshell-ai/OpenVoice.git
# 2. 创建全新的虚拟环境
cd /usr/local/src/openvoicev2
python -m venv venv

# 3. 激活新环境
source venv/bin/activate

# 4. 首先安装正确版本的 NumPy
pip install numpy==1.22.0

# 5. 然后安装其他核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. 安装项目依赖
cd /usr/local/src/openvoicev2/OpenVoice
pip install -r requirements.txt

# 7. 安装 MeloTTS
pip install git+https://github.com/myshell-ai/MeloTTS.git

cd /usr/local/src/openvoicev2/OpenVoice

python -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('checkpoints', exist_ok=True)
snapshot_download(repo_id='myshell-ai/OpenVoice', local_dir='./', local_dir_use_symlinks=False)
print('下载完成！')
"

pip install faster-whisper
pip install whisper-timestamped

pip install -U silero

pip install wavmark

pip install gradio==3.48.0

huggingface-cli login

python -m openvoice.openvoice_app --share