pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requrements.txt

nohup uvicorn web_main:app --host=0.0.0.0 --port=8800 --reload  > web.log 2>&1 &

huggingface 镜像（如清华源）
export HF_ENDPOINT=https://hf-mirror.com
