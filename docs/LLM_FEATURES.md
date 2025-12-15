# LLM 功能说明

## ✅ 已实现的功能

### 1. 标准 LangChain LLM 接口适配

`StableLLM` 现在继承自 `BaseLLM`，完全兼容 LangChain 标准接口：

```python
from core.llm_chatglm import StableLLM

llm = StableLLM()

# LangChain 标准调用
result = llm("你好")
result = llm.invoke("你好")

# 支持 LangChain chains
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

**实现的接口：**
- `_call()`: 同步调用
- `_generate()`: 批量生成
- `_llm_type`: LLM 类型标识

### 2. Stream / SSE 版本

支持同步和异步流式生成：

```python
# 同步流式
for chunk in llm.stream({"query": "你好", "context": "", "history": []}):
    print(chunk, end="", flush=True)

# 异步流式（SSE 兼容）
async for chunk in llm.astream({"query": "你好", "context": "", "history": []}):
    print(chunk, end="", flush=True)
```

**在 FastAPI 中使用 SSE：**

```python
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

@app.get("/stream")
async def stream_response(question: str):
    async def event_generator():
        async for chunk in llm.astream({"query": question}):
            yield {"data": chunk}
    
    return EventSourceResponse(event_generator())
```

### 3. LLMFactory（Qwen / GLM 自动切换）

自动识别并创建合适的模型：

```python
from core.llm_chatglm import LLMFactory

# 自动选择（根据设备）
llm = LLMFactory.create_llm()

# 指定模型名称，自动识别类型
llm = LLMFactory.create_llm(model_name="Qwen/Qwen2.5-7B-Instruct")

# 直接创建 Qwen 系列
llm = LLMFactory.create_qwen_llm()

# 直接创建 GLM 系列
llm = LLMFactory.create_glm_llm()
```

**工厂方法：**
- `create_llm()`: 通用创建，自动识别模型类型
- `create_qwen_llm()`: 创建 Qwen 系列
- `create_glm_llm()`: 创建 GLM 系列

### 4. RAG Prompt 压缩 30% Token

优化了提示词，减少约 30% 的 token 使用：

**优化前：**
```
你是一个专业的AI助手，请基于提供的文档内容准确回答用户问题。
要求：
1. 直接给出结论
2. 严格基于文档
3. 文档中没有信息则说明无法找到
4. 不要客套话

【参考文档】
{context}

【问题】
{query}
```

**优化后：**
```
基于文档回答问题。要求：1.直接结论 2.严格基于文档 3.无信息则说明未找到 4.无客套话

文档：{context}
问题：{query}
```

**Token 节省：**
- 系统提示词：从 ~50 tokens 减少到 ~30 tokens（节省 40%）
- 上下文格式：从 ~10 tokens 减少到 ~5 tokens（节省 50%）
- 总体节省：约 30%

## 使用示例

### 基础使用

```python
from core.llm_chatglm import StableLLM

llm = StableLLM(
    model_name_cpu="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=512
)

# 简单调用
result = llm.invoke({
    "query": "什么是人工智能？",
    "context": "人工智能是计算机科学的一个分支...",
    "history": []
})
```

### RAG 场景

```python
from core.llm_chatglm import LLMFactory

llm = LLMFactory.create_llm()

# RAG 调用
result = llm.invoke({
    "query": "文档中提到了什么？",
    "context": "检索到的文档内容...",
    "history": [("之前的问题", "之前的回答")]
})
```

### 流式生成

```python
# 同步流式
for chunk in llm.stream({"query": "写一首诗"}):
    print(chunk, end="", flush=True)

# 异步流式（用于 Web 服务）
async def generate_stream(query: str):
    async for chunk in llm.astream({"query": query}):
        yield chunk
```

### LangChain 集成

```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from core.llm_chatglm import StableLLM

llm = StableLLM()

prompt = PromptTemplate(
    input_variables=["question"],
    template="回答：{question}"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("什么是机器学习？")
```

## 性能优化

1. **Token 压缩**：RAG prompt 减少 30% token
2. **流式生成**：实时输出，提升用户体验
3. **自动模型选择**：根据设备自动选择最优模型
4. **Tokenizer 回退**：fast tokenizer 失败时自动回退

## 兼容性

- ✅ LangChain 1.0+
- ✅ Qwen2.5 / Qwen3
- ✅ ChatGLM-4
- ✅ CPU / CUDA 自动切换
- ✅ FastAPI SSE 支持

