from fastapi import FastAPI, Query
from pydantic import BaseModel
from chroma_db import VectorStoreManager
from llm_chatglm import ChatGLMLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=2, return_messages=True, output_key="result")

app = FastAPI()


class AskRequest(BaseModel):
    question: str


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
    result = qa_chain.invoke(question)
    # result 可能是 dict，包含 answer 和 source_documents
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or str(result)
    else:
        answer = str(result)
    return {"answer": answer}
