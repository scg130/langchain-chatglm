import hashlib
import os
from typing import Dict, List, Optional, Union

from langchain.schema import Document
from langchain.text_splitter import (RecursiveCharacterTextSplitter,
                                     TextSplitter)
from langchain_chroma import Chroma
from langchain_community.document_loaders import (DirectoryLoader,
                                                  Docx2txtLoader, PyPDFLoader,
                                                  TextLoader,
                                                  UnstructuredFileLoader)
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:
    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化向量存储管理器

        参数:
            persist_dir: 向量数据库持久化目录
            embedding_model: 嵌入模型名称
            chunk_size: 默认分块大小
            chunk_overlap: 分块重叠大小
        """
        self.persist_dir = persist_dir
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectordbs: Dict[str, Chroma] = {}  # 路径到向量库的映射字典
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap

    def _get_loader(self, file_path: str):
        """根据文件类型返回对应的加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return PyPDFLoader(file_path)
        elif ext == '.docx':
            return Docx2txtLoader(file_path)
        elif ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        else:
            return UnstructuredFileLoader(file_path)

    def load_documents(
        self,
        input_path: str,
        file_pattern: str = "**/*",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        custom_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        """
        加载并分块处理文档

        参数:
            input_path: 文件或文件夹路径
            file_pattern: 文件匹配模式
            chunk_size: 分块大小（默认使用初始化参数）
            chunk_overlap: 分块重叠大小（默认使用初始化参数）
            custom_splitter: 自定义文本分割器
        """
        # 参数处理
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap

        # 加载文档
        if os.path.isfile(input_path):
            loader = self._get_loader(input_path)
            raw_docs = loader.load()
        else:
            loader = DirectoryLoader(
                input_path,
                glob=file_pattern,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
                use_multithreading=True
            )
            raw_docs = loader.load()

        # 分块处理
        splitter = custom_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "…", " ", ""]
        )

        docs = splitter.split_documents(raw_docs)

        # 添加内容哈希和文档结构信息
        for doc in docs:
            content_hash = hashlib.md5(
                doc.page_content.encode("utf-8")).hexdigest()
            doc.metadata.update({
                "content_hash": content_hash,
                "chunk_size": len(doc.page_content),
                "original_source": doc.metadata.get("source", "")
            })

        return docs

    def _existing_hashes(self, dir_path: str) -> set:
        """获取当前库中已有文档的content_hash集合"""
        try:
            results = self.get_vectorstore(dir_path).get(include=["metadatas"])
            return {m["content_hash"] for m in results["metadatas"] if "content_hash" in m}
        except Exception as e:
            print(f"⚠️ 获取现有哈希失败: {str(e)}")
            return set()

    def add_documents(
        self,
        dir_path: str,
        new_docs: List[Document],
        batch_size: int = 4000,
        show_progress: bool = True
    ) -> dict:
        """
        添加文档（自动去重+分批处理）

        返回:
            {
                "total": 总文档数,
                "added": 成功添加数,
                "duplicates": 重复文档数,
                "failed": 失败文档数
            }
        """
        if not new_docs:
            print("⚠️ 没有可添加的文档")
            return {"total": 0, "added": 0, "duplicates": 0, "failed": 0}

        # 去重处理
        existing_hashes = self._existing_hashes(dir_path=dir_path)
        filtered_docs = []
        duplicate_count = 0

        for doc in new_docs:
            if doc.metadata.get("content_hash") not in existing_hashes:
                filtered_docs.append(doc)
            else:
                duplicate_count += 1

        # 分批插入
        added_count = 0
        failed_count = 0

        for i in range(0, len(filtered_docs), batch_size):
            batch = filtered_docs[i:i + batch_size]
            try:
                self.get_vectorstore(dir_path).add_documents(batch)
                added_count += len(batch)
                if show_progress:
                    print(
                        f"⏳ 进度: {min(i+batch_size, len(filtered_docs))}/{len(filtered_docs)}")
            except Exception as e:
                failed_count += len(batch)
                print(f"❌ 批量插入失败: {str(e)}")
                # 可以添加重试逻辑或更细粒度的错误处理

        # 结果统计
        stats = {
            "total": len(new_docs),
            "added": added_count,
            "duplicates": duplicate_count,
            "failed": failed_count
        }

        if show_progress:
            print("\n📊 导入结果:")
            print(f"- 总文档: {stats['total']}")
            print(f"- 新增文档: {stats['added']} (去重后)")
            print(f"- 重复文档: {stats['duplicates']}")
            if stats['failed'] > 0:
                print(f"- 失败文档: {stats['failed']} (需检查)")

        return stats

    def delete_documents(
        self,
        dir_path: str,
        ids: Optional[List[str]] = None,
        source_path: Optional[str] = None,
        content_hash: Optional[str] = None
    ) -> int:
        """
        删除文档（支持多种删除方式）

        返回:
            删除的文档数量
        """
        if not any([ids, source_path, content_hash]):
            print("⚠️ 请至少提供一种删除条件")
            return 0

        try:
            # 获取需要删除的ID
            if ids:
                ids_to_delete = ids
            else:
                results = self.get_vectorstore(dir_path).get(
                    include=["metadatas", "ids"])
                ids_to_delete = []

                for doc_id, meta in zip(results["ids"], results["metadatas"]):
                    if source_path and meta.get("source") == source_path:
                        ids_to_delete.append(doc_id)
                    elif content_hash and meta.get("content_hash") == content_hash:
                        ids_to_delete.append(doc_id)

            # 执行删除
            if ids_to_delete:
                self.get_vectorstore(dir_path).delete(ids=ids_to_delete)
                print(f"🗑️ 已删除 {len(ids_to_delete)} 条文档")
                return len(ids_to_delete)

            print("⚠️ 未找到匹配的文档")
            return 0

        except Exception as e:
            print(f"❌ 删除失败: {str(e)}")
            return 0

    def get_vectorstore(self, dir_path: str) -> Chroma:
        """获取底层向量数据库实例"""
        # 定义嵌入模型
        embedding = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese")

        # 创建空集合
        vectordb = Chroma(
            collection_name=dir_path,
            embedding_function=embedding,
            persist_directory=self.persist_dir
        )
        self.vectordbs[dir_path] = vectordb
        return vectordb

    def optimize_storage(self, dir_path: str):
        """优化存储（ChromaDB内部压缩）"""
        try:
            self.get_vectorstore(dir_path).persist()
            print("✅ 存储优化完成")
        except Exception as e:
            print(f"❌ 优化失败: {str(e)}")
