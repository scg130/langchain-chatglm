from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from chatglm.llm_chatglm import ChatGLMLLM
from chroma.chroma_db import VectorStoreManager
from config.logger_config import logger

memory = ConversationBufferWindowMemory(
    k=2, return_messages=True, output_key="result")


def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGLMLLM()
    prompt_template = """
    你是一个专业的知识问答助手，根据提供的内容，简洁准确地回答用户的问题。
    如果内容不足以回答，请礼貌告知用户无法回答。

    上下文内容：
    {context}

    问题：
    {question}

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
