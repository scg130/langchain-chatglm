# LangChain 1.0 升级指南

## 已完成的更新

### 1. 依赖包更新 (requirements.txt)
- ✅ `langchain>=1.0.0` (从 0.3.26 升级)
- ✅ `langchain-chroma>=0.1.0` (从 0.2.4 升级)
- ✅ 新增 `langchain-text-splitters>=0.2.0` (文本分割器独立包)
- ✅ 新增 `langchain-core>=0.3.0` (核心功能)
- ✅ `langchain-community>=0.3.0` (社区集成)
- ✅ `langchain-huggingface>=0.1.0` (从 0.3.0 更新)
- ✅ 新增 `pydantic>=2.0.0` (LangChain 1.0 要求)

### 2. 导入路径更新

#### core/vectorstore_manager.py
- ✅ `from langchain.schema import Document` → `from langchain_core.documents import Document`
- ✅ `from langchain.text_splitter import ...` → `from langchain_text_splitters import ...`

#### util/func.py
- ✅ `from langchain.prompts import PromptTemplate` → `from langchain_core.prompts import PromptTemplate`

## 可能需要额外处理的 API

### RetrievalQA 和 ConversationBufferWindowMemory
这些 API 在 LangChain 1.0 中可能已移至 `langchain-legacy` 包。

**如果遇到导入错误，请执行：**
```bash
pip install langchain-legacy
```

然后更新导入：
```python
# 如果新版本不可用，使用 legacy
try:
    from langchain.chains import RetrievalQA
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_legacy.chains import RetrievalQA
    from langchain_legacy.memory import ConversationBufferWindowMemory
```

## 升级步骤

### 1. 备份当前环境
```bash
pip freeze > requirements_backup.txt
```

### 2. 升级依赖
```bash
pip install -r requirements.txt --upgrade
```

### 3. 测试运行
```bash
python run.py
```

### 4. 检查错误
- 查看控制台输出
- 检查日志文件 `logs/app.log`
- 测试上传文档功能
- 测试问答功能

## 已知的 Breaking Changes

### 1. Document 导入路径
- **旧**: `from langchain.schema import Document`
- **新**: `from langchain_core.documents import Document`

### 2. TextSplitter 导入路径
- **旧**: `from langchain.text_splitter import RecursiveCharacterTextSplitter`
- **新**: `from langchain_text_splitters import RecursiveCharacterTextSplitter`

### 3. PromptTemplate 导入路径
- **旧**: `from langchain.prompts import PromptTemplate`
- **新**: `from langchain_core.prompts import PromptTemplate`

### 4. Pydantic 版本要求
- LangChain 1.0 需要 `pydantic>=2.0.0`
- 如果其他包依赖 pydantic 1.x，可能需要处理兼容性

## 故障排查

### 问题1: ImportError: cannot import name 'RetrievalQA'
**解决方案**: 安装 `langchain-legacy`
```bash
pip install langchain-legacy
```

### 问题2: Pydantic 版本冲突
**解决方案**: 升级 pydantic
```bash
pip install "pydantic>=2.0.0"
```

### 问题3: 其他导入错误
**解决方案**: 检查 LangChain 1.0 迁移指南
- https://python.langchain.com/docs/versions/v0_2/

## 测试清单

- [ ] 服务启动正常
- [ ] 文档上传功能正常
- [ ] 向量数据库创建正常
- [ ] 问答功能正常
- [ ] 流式输出正常
- [ ] 网络搜索功能正常
- [ ] 知识库选择功能正常

## 回滚方案

如果升级后出现问题，可以回滚：

```bash
pip install langchain==0.3.26 langchain-chroma==0.2.4 langchain-huggingface==0.3.0
```

然后恢复旧的导入路径。

