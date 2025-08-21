#!/bin/bash
sudo apt-get update --fix-missing
sudo apt install -y build-essential \
    libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    libgtk-3-dev libwebkit2gtk-4.0-dev libgtk-3-0 libgl1-mesa-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
cd /usr/src/local
git clone https://github.com/ddean2009/MoneyPrinterPlus.git
cd MoneyPrinterPlus

pip install -r requirements.txt
#  .streamlit/config.toml  修改  server.port = 8800
#  .streamlit/config.toml  修改  server.address = 0.0.0.0
streamlit run gui.py 