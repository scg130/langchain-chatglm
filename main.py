import os
import sys

from chroma_db import VectorStoreManager
from func import get_qa_chain

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
