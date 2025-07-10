from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.chains import RetrievalQA
from llm_chatglm import ChatGLMLLM
import os

# 加载文档并切分
def load_docs():
    loader = DirectoryLoader(
    "./data",
    glob="**/*.txt",                     # 可自定义扩展名：*.md, *.csv, 等
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}  # 避免编码报错
)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    return docs

# 构建向量库
def build_vector_store(docs):
    embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    vectordb = Chroma(persist_directory="./chroma_store", embedding_function=embedding)
    # vectordb.add_documents(docs)
    # vectordb.persist()
    return vectordb

# 初始化 QA chain
def get_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    llm = ChatGLMLLM()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

if __name__ == "__main__":
    if not os.path.exists("chroma_store/index"):
        print("🔄 正在构建向量库...")
        docs = load_docs()
        vectordb = build_vector_store(docs)
    else:
        print("✅ 加载已有向量库...")
        embedding = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
        vectordb = Chroma(persist_directory="./chroma_store", embedding_function=embedding)

    qa_chain = get_qa_chain(vectordb)

    while True:
        query = input("❓ 用户问题：")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa_chain.run(query)
        print("💡 答案：", result)
