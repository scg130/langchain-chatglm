import os

from fastapi import FastAPI, Query
from pydantic import BaseModel

from chroma_db import VectorStoreManager
from func import get_qa_chain
from llm_chatglm import ChatGLMLLM

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
