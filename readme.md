conda create -n py310 python=3.10.14 -y
conda activate py310
 
python 3.10.14   or python  3.11.7
CUDA Version: 12.4 


pip install -r requirements.txt

huggingface 镜像（如清华源）
export HF_ENDPOINT=https://hf-mirror.com

docker run --name redis-server -p 6379:6379 -d redis redis-server --requirepass smd013012

pip install huggingface-hub

huggingface-cli download THUDM/chatglm3-6b --resume-download --local-dir ./chatglm3-6b

## 🌐 前端界面

启动服务后访问 http://127.0.0.1:8800 使用Web界面：

- 💬 **聊天功能**：实时对话
- 📁 **文件上传**：支持 .txt, .pdf, .docx, .md
- 🔍 **网络搜索**：获取最新信息
- 📚 **知识库管理**：选择不同知识库

## 📡 API 调用示例

### Web界面
直接访问：http://127.0.0.1:8800

### 命令行调用
curl -X POST http://127.0.0.1:8800/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "巴黎奥运会的乒乓球男单冠军是谁？",
    "history": [],
    "is_web_search": true,
    "dir_path": ""
  }'
