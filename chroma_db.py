import hashlib
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader


class VectorStoreManager:
    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        embedding_model: str = "shibing624/text2vec-base-chinese",
    ):
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectordb = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )

    def load_docs(
        self,
        folder: str = "./data",
        file_pattern: str = "**/*.txt",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ) -> List[Document]:
        loader = DirectoryLoader(
            folder,
            glob=file_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        raw_docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = splitter.split_documents(raw_docs)

        # 添加内容哈希作为 metadata，供去重/删除用
        for doc in docs:
            content_hash = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            doc.metadata["content_hash"] = content_hash
        return docs

    def _existing_hashes(self) -> set:
        """
        获取当前库中已有文档的 content_hash，用于去重。
        """
        try:
            results = self.vectordb.get(include=["metadatas"])
            return {m.get("content_hash") for m in results["metadatas"] if "content_hash" in m}
        except Exception as e:
            print("获取现有文档哈希失败：", e)
            return set()

    def add_documents(self, new_docs: List[Document]):
        """
        添加文档并自动去重（基于内容哈希）。
        """
        if not new_docs:
            print("⚠️ 没有新文档可添加。")
            return

        existing_hashes = self._existing_hashes()
        filtered_docs = [doc for doc in new_docs if doc.metadata.get("content_hash") not in existing_hashes]

        if not filtered_docs:
            print("📎 所有文档内容均已存在，无需添加。")
            return

        self.vectordb.add_documents(filtered_docs)
        print(f"✅ 成功添加 {len(filtered_docs)} 条新文档（去重后）。")

    def delete_documents_by_source(self, source_path: str):
        """
        根据 metadata["source"] 删除某个来源的所有文档。
        :param source_path: 例如 "./data/example.txt"
        """
        try:
            results = self.vectordb.get(include=["metadatas", "ids"])
            ids_to_delete = [
                doc_id for doc_id, meta in zip(results["ids"], results["metadatas"])
                if meta.get("source") == source_path
            ]
            if ids_to_delete:
                self.vectordb.delete(ids=ids_to_delete)
                print(f"🗑️ 成功删除来源为 {source_path} 的 {len(ids_to_delete)} 条文档。")
            else:
                print(f"⚠️ 未找到来源为 {source_path} 的文档。")
        except Exception as e:
            print("❌ 删除失败：", e)

    def get_vectorstore(self) -> Chroma:
        return self.vectordb
