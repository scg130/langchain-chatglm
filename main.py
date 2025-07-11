from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from llm_chatglm import ChatGLMLLM
from chroma_db import VectorStoreManager
import os
import sys
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, return_messages=True，output_key="result",)

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
    manager = VectorStoreManager()
    if not os.path.exists("chroma_store/index"):
        print("🔄 正在构建向量库...")
        docs = manager.load_docs()
        manager.add_documents(docs)
        print("✅ 向量库构建完成！")
        vectordb = manager.get_vectorstore()
    else:
        print("✅ 加载已有向量库...")
        vectordb = manager.get_vectorstore()

    qa_chain = get_qa_chain(vectordb)

    while True:
        print("❓ 用户问题：", end='', flush=True)
        query = sys.stdin.buffer.readline().decode('utf-8').strip()
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain.invoke(query)
        print("💡 答案：", result)
