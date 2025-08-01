from core.vectorstore_manager import VectorStoreManager
from core.llm_chatglm import ChatGLMLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 初始化向量库管理器实例
vector_manager = VectorStoreManager()

# 初始化LLM实例
llm = ChatGLMLLM()

def initialize_vectordb(dir_path: str):
    vectordbs = {}
    for index_type in ["full_text", "section", "detail"]:
        vectordbs[f"{dir_path}_{index_type}"] = vector_manager.get_vectorstore(dir_path, index_type)
    return vectordbs

def get_qa_chain(retriever):
    prompt_template = """
    文档内容（请严格参考）：
    {context}

    问题：
    {question}

    答案：
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

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

import torch
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

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
