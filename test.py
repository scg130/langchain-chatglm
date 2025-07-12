from util.func import initialize_vectordb

db = initialize_vectordb("./data/qntc")

results = db.similarity_search("全能天才第4章", k=1)
print("搜索结果:", results[0].page_content if results else "无结果")
