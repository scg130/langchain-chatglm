#!/bin/bash
sudo apt-get update --fix-missing
cd /usr/src/local
git clone https://github.com/scg130/MoneyPrinterTurbo.git
cd MoneyPrinterTurbo
pip install -r requirements.txt
sudo apt-get install imagemagick -y
cp config.example.toml config.toml
sed -i 's|pexels_api_keys = \[\]|pexels_api_keys = \["AHTxqthszIHRdJwbhSrCL29i0s4DQRUCew366dKbhratn9A9PM57waqo"\]|' config.toml
sed -i '/^\[proxy\]$/a\
http = "http://127.0.0.1:8118"\
https = "http://127.0.0.1:8118"' config.toml
sed -i '/^streamlit run .\/webui\/Main\.py/ s/$/ --server.port 8800/' webui.sh
export http_proxy=http://127.0.0.1:8118
export https_proxy=http://127.0.0.1:8118
sh webui.sh