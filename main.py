import os
import sys

from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from chroma_db import VectorStoreManager
from llm_chatglm import ChatGLMLLM

memory = ConversationBufferWindowMemory(
    k=2, return_messages=True, output_key="result")

# 初始化 QA chain


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
    # 使用 from_chain_type 时传入 prompt 参数来自定义提示模板
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        memory=memory
    )
    return qa_chain


if __name__ == "__main__":
    manager = VectorStoreManager(persist_dir="./chroma_store")
    docs = manager.load_documents(
        input_path="./data",
        file_pattern="**/*.txt",
        chunk_size=1000,  # 大文档使用更大的分块
        chunk_overlap=100
    )
    result = manager.add_documents(docs, batch_size=2000)
    print(f"导入成功率: {result['added']/result['total']:.1%}")
    vectordb = manager.get_vectorstore()

    qa_chain = get_qa_chain(vectordb)

    while True:
        print("❓ 用户问题：", end='', flush=True)
        query = sys.stdin.buffer.readline().decode('utf-8').strip()
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain.invoke(query)
        print("💡 答案：", result)
