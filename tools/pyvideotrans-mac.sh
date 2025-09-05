#!/bin/bash
git clone https://github.com/jianchang512/pyvideotrans 
cd pyvideotrans
python -m venv venv
source venv/bin/activate
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com
pip install -r requirements.txt

python sp.py