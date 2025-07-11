from fastapi import FastAPI, Query
from pydantic import BaseModel
from chroma_db import VectorStoreManager
from llm_chatglm import ChatGLMLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2, return_messages=True,output_key="result")

app = FastAPI()

class AskRequest(BaseModel):
    question: str

# 初始化向量库和 QA chain（全局只初始化一次）
manager = VectorStoreManager()
if not os.path.exists("chroma_store/adc7fd08-6721-4d0a-8bbe-471abb238ce9"):
    # 这里假设 adc7fd08-6721-4d0a-8bbe-471abb238ce9 是 index 文件夹
    docs = manager.load_docs()
    manager.add_documents(docs)
    vectordb = manager.get_vectorstore()
else:
    vectordb = manager.get_vectorstore()

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

qa_chain = get_qa_chain(vectordb)

@app.post("/ask")
async def ask(request: AskRequest):
    question = request.question
    print(question)
    result =   qa_chain.invoke(question)
    # result 可能是 dict，包含 answer 和 source_documents
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or str(result)
    else:
        answer = str(result)
    return {"answer": answer}
