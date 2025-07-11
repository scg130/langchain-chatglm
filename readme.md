pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requrements.txt

nohup uvicorn web_main:app --host=0.0.0.0 --port=8800 --reload  > web.log 2>&1 &