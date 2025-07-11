import hashlib
import os
from typing import Dict, List, Optional, Union

from chromadb import HttpClient
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
        host: str = "127.0.0.1",
        port: int = 8000,
        embedding_model: str = "shibing624/text2vec-base-chinese",
        collection_name: str = "default",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        远程ChromaDB向量库管理器

        参数:
            host: ChromaDB服务器地址
            port: 服务端口
            embedding_model: 嵌入模型名称
            collection_name: 集合名称
            chunk_size: 默认分块大小
            chunk_overlap: 分块重叠大小
        """
        self.client = HttpClient(host=host, port=port)
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectordb = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding
        )
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap

    def _get_loader(self, file_path: str):
        """根据文件扩展名返回对应的文档加载器"""
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
        加载并分块处理文档（支持大文件自动分块）

        参数:
            input_path: 文件/文件夹路径
            file_pattern: 文件匹配模式
            chunk_size: 自定义分块大小
            chunk_overlap: 自定义重叠大小
            custom_splitter: 自定义文本分割器
        """
        # 参数处理
        chunk_size = chunk_size or self.default_chunk_size
        chunk_overlap = chunk_overlap or self.default_chunk_overlap

        # 文档加载
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

        # 文档分块
        splitter = custom_splitter or RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ";", "…", " ", ""]
        )

        docs = splitter.split_documents(raw_docs)

        # 添加元数据
        for doc in docs:
            content_hash = hashlib.md5(
                doc.page_content.encode("utf-8")).hexdigest()
            doc.metadata.update({
                "content_hash": content_hash,
                "chunk_size": len(doc.page_content),
                "original_source": doc.metadata.get("source", "")
            })

        return docs

    def _existing_hashes(self) -> set:
        """获取远程库中已有文档的哈希集合"""
        try:
            results = self.vectordb.get(include=["metadatas"])
            return {m["content_hash"] for m in results["metadatas"] if "content_hash" in m}
        except Exception as e:
            print(f"⚠️ 获取远程哈希失败: {str(e)}")
            return set()

    def add_documents(
        self,
        new_docs: List[Document],
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        安全添加文档到远程ChromaDB（自动分批次+去重）

        返回统计字典:
            {
                "total": 总文档数,
                "added": 成功添加数,
                "duplicates": 重复文档数,
                "failed": 失败数
            }
        """
        if not new_docs:
            print("⚠️ 无有效文档可添加")
            return {"total": 0, "added": 0, "duplicates": 0, "failed": 0}

        # 去重处理
        existing_hashes = self._existing_hashes()
        filtered_docs = []
        duplicate_count = 0

        for doc in new_docs:
            if doc.metadata.get("content_hash") not in existing_hashes:
                filtered_docs.append(doc)
            else:
                duplicate_count += 1

        # 分批处理（远程连接建议更小的batch_size）
        added_count = 0
        failed_count = 0

        for i in range(0, len(filtered_docs), batch_size):
            batch = filtered_docs[i:i + batch_size]
            try:
                self.vectordb.add_documents(batch)
                added_count += len(batch)
                if show_progress:
                    print(
                        f"⏳ 进度: {min(i+batch_size, len(filtered_docs))}/{len(filtered_docs)}")
            except Exception as e:
                failed_count += len(batch)
                print(f"❌ 批次 {i//batch_size} 插入失败: {str(e)}")
                # 可在此添加重试逻辑

        # 打印统计结果
        stats = {
            "total": len(new_docs),
            "added": added_count,
            "duplicates": duplicate_count,
            "failed": failed_count
        }

        if show_progress:
            print("\n📊 导入结果:")
            print(f"- 总文档: {stats['total']}")
            print(f"- 新增: {stats['added']} (去重后)")
            print(f"- 重复: {stats['duplicates']}")
            if stats['failed'] > 0:
                print(f"- 失败: {stats['failed']} (建议检查网络或分批重试)")

        return stats

    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        source_path: Optional[str] = None,
        content_hash: Optional[str] = None
    ) -> int:
        """
        多功能文档删除

        参数:
            ids: 直接指定ID删除
            source_path: 按文件来源删除
            content_hash: 按内容哈希删除

        返回:
            实际删除的文档数量
        """
        if not any([ids, source_path, content_hash]):
            print("⚠️ 需要至少指定一种删除条件")
            return 0

        try:
            # 直接删除指定ID
            if ids:
                self.vectordb.delete(ids=ids)
                print(f"🗑️ 已删除 {len(ids)} 条指定ID文档")
                return len(ids)

            # 条件删除
            results = self.vectordb.get(include=["metadatas", "ids"])
            ids_to_delete = []

            for doc_id, meta in zip(results["ids"], results["metadatas"]):
                if source_path and meta.get("source") == source_path:
                    ids_to_delete.append(doc_id)
                elif content_hash and meta.get("content_hash") == content_hash:
                    ids_to_delete.append(doc_id)

            if ids_to_delete:
                self.vectordb.delete(ids=ids_to_delete)
                print(f"🗑️ 已删除 {len(ids_to_delete)} 条匹配文档")
                return len(ids_to_delete)

            print("⚠️ 未找到匹配文档")
            return 0

        except Exception as e:
            print(f"❌ 删除操作失败: {str(e)}")
            return 0

    def get_vectorstore(self) -> Chroma:
        """获取底层Chroma客户端实例"""
        return self.vectordb

    def collection_info(self) -> dict:
        """获取当前集合统计信息"""
        try:
            return {
                "count": self.vectordb._collection.count(),
                "metadata": self.vectordb._collection.metadata
            }
        except Exception as e:
            print(f"❌ 获取集合信息失败: {str(e)}")
            return {}
