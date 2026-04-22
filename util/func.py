from typing import Any

import torch
from langchain_community.chat_message_histories import RedisChatMessageHistory

# 兼容不同版本的 langchain，缺失时提供友好提示
try:
    from langchain.memory import ConversationBufferWindowMemory  # type: ignore  # langchain>=1.0
except ImportError:
    ConversationBufferWindowMemory = None  # 允许缺省，后续运行时再提示

from core.vectorstore_manager import VectorStoreManager

# 初始化向量库管理器实例
vector_manager = VectorStoreManager()


def initialize_vectordb(dir_path: str):
    return vector_manager.get_vectorstore(dir_path)


def get_memory():
    if ConversationBufferWindowMemory is None:
        raise ImportError(
            "ConversationBufferWindowMemory 不可用。请安装 langchain>=1.0.0 "
            "或安装兼容包（如 langchain-legacy），然后重试。"
        )

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
