import os
import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from util.func import get_qa_chain, initialize_vectordb
from config.logger_config import logger

app = FastAPI(
    title="智能文档问答API",
    description="基于LangChain和ChatGLM的文档问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class AskRequest(BaseModel):
    question: str

@app.on_event("startup")
async def startup_event():
    """启动时自动初始化"""
    global qa_chain
    vectordb = initialize_vectordb()
    qa_chain = get_qa_chain(vectordb)

@app.post("/ask")
async def ask(request: AskRequest):
    question = request.question
    result = qa_chain.invoke(question)
    # result 可能是 dict，包含 answer 和 source_documents
    if isinstance(result, dict):
        answer = result.get("result") or result.get("answer") or str(result)
    else:
        answer = str(result)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800, log_level="info")