from typing import Any, List, Tuple

import torch
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableMap

from core.vectorstore_manager import VectorStoreManager

# 初始化向量库管理器实例
vector_manager = VectorStoreManager()


def initialize_vectordb(dir_path: str):
    return vector_manager.get_vectorstore(dir_path)


def format_history(history: List[Tuple[str, str]]) -> str:
    """把对话历史列表格式化成字符串"""
    return "\n".join([f"用户：{q}\n助手：{a}" for q, a in history])


def get_qa_chain_with_history(llm: Any, retriever: Any) -> Any:
    def build_context(x):
        # 优先使用外部 context，否则从检索器获取
        if x.get("context"):
            return x["context"]
        docs = retriever.invoke(x["query"])
        return "\n".join([doc.page_content for doc in docs])

    chain = (
        RunnableMap({
            "query": lambda x: x["query"],
            "history": lambda x: format_history(x.get("history", [])),
            "context": RunnableLambda(build_context)
        })
        | llm
    )
    return chain


def get_qa_chain(llm: Any, retriever: Any) -> Any:
    prompt_template = """请基于以下文档内容准确回答用户问题。

【文档内容】
{context}

【用户问题】
{question}

回答要求：
1. 严格基于文档内容，不要编造信息
2. 回答要简洁明了，直接给出核心答案
3. 如果文档中没有相关信息，请说明"文档中未找到相关信息"
4. 避免使用客套话和重复内容

【回答】
"""
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    # 创建RetrievalQA链，llm使用自定义的ChatGLMLLM，检索器为vectordb.as_retriever
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        # memory=get_memory(),  # 使用全局定义的内存
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain


def get_memory():
    if torch.cuda.is_available():
        # 使用 GPU，启用 Redis 存储历史
        print("✅ 检测到 GPU，使用 Redis 存储对话历史")
        history = RedisChatMessageHistory(
            session_id="your-session-id",
            url="redis://:smd013012@localhost:6379/0",
            ttl=3600,
            key_prefix="message_store:"
        )
        memory = ConversationBufferWindowMemory(
            k=20,
            memory_key="chat_history",
            chat_memory=history,
            return_messages=True,
            output_key="result",
            input_key="question"
        )
    else:
        # CPU-only，使用内存存储历史
        print("⚠️ 未检测到 GPU，使用本地内存存储对话历史")
        memory = ConversationBufferWindowMemory(
            k=20,
            memory_key="chat_history",
            return_messages=True,
            output_key="result",
            input_key="question"
        )
    return memory
