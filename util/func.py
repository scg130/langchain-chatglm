from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory

from langchain.prompts import PromptTemplate

from langchain_community.chat_message_histories import RedisChatMessageHistory
from chatglm.llm_chatglm import ChatGLMLLM
from chroma.chroma_db import VectorStoreManager
from config.logger_config import logger
import re

import torch

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

memory = get_memory()

def get_qa_chain(vectordb,keywords: str = None):
    retriever = vectordb.as_retriever(
                search_kwargs={
                    "k": 5,
                    "filter": {
                        "keywords": keywords  # 关键词匹配
                    }
                }
            )

    llm = ChatGLMLLM()
    prompt_template = """
                        你是一个文档问答助手，请根据以下提供的文档内容，精准回答问题。

                        文档内容（请严格参考）：
                        {context}

                        问题：
                        {question}

                        请基于文档内容作答，如无相关信息，请回答“文档中未找到相关内容”。

                        答案：
                        """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        memory=memory
    )
    return qa_chain

def extract_question(text: str) -> str:
    """
    从包含'文档内容'和'问题'的文本中提取问题部分
    """
    # 匹配“问题：”之后直到“答案：”之前的内容，非贪婪匹配
    pattern = r"问题：\s*(.*?)\s*(答案：|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        question = match.group(1).strip()
        return question
    return ""

# 初始化全局组件
manager = VectorStoreManager()


def initialize_vectordb(dir_path: str):
    """初始化向量数据库"""
    try:
        logger.info("⏳ 正在加载文档到向量数据库...")
        docs = manager.load_documents(
            input_path=dir_path,
            file_pattern="**/*",
            chunk_size=1000,
            chunk_overlap=100
        )
        result = manager.add_documents(
            dir_path=dir_path, new_docs=docs, batch_size=2000)
        logger.info(f"✅ 文档加载完成. 成功率: {result['added']/result['total']:.1%}")
        return manager.get_vectorstore(dir_path=dir_path)
    except Exception as e:
        logger.error(f"❌ 文档加载失败: {str(e)}")
        raise
