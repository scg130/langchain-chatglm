python 3.10.14   or python  3.11.7
CUDA Version: 12.4 


pip install -r requrements.txt

huggingface 镜像（如清华源）
export HF_ENDPOINT=https://hf-mirror.com

docker run --name redis-server -p 6379:6379 -d redis redis-server --requirepass smd013012

pip install huggingface-hub

huggingface-cli download THUDM/chatglm3-6b --resume-download --local-dir ./chatglm3-6b

curl -X POST http://127.0.0.1:8800/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "巴黎奥运会的乒乓球男单冠军是谁？",
    "history": [
      ["巴黎奥运会的乒乓球男单冠军是谁？", "冠军是樊振东。"]
    ],
    "is_web_search": true,
    "dir_path": ""
  }'
