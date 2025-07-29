import hashlib
import logging
import os
from typing import Dict, List, Optional, Set, Union

from chromadb import PersistentClient

# chroma_db.py 顶部添加以下代码
try:
    # 新版本 ChromaDB (>=0.4.0)
    from chromadb.errors import CollectionNotFound
except ImportError:
    try:
        # 旧版本 ChromaDB (<0.4.0)
        from chromadb.api.exceptions import CollectionNotFound
    except ImportError:
        # 最新版本 (如 0.5.0+) 或其他情况
        class CollectionNotFound(Exception):
            """自定义异常类用于兼容"""
            pass

# 然后继续原有导入
from chromadb import PersistentClient
from langchain.schema import Document
from langchain.text_splitter import (RecursiveCharacterTextSplitter,
                                     TextSplitter)
from langchain_chroma import Chroma
from langchain_community.document_loaders import (DirectoryLoader,
                                                  Docx2txtLoader, PyPDFLoader,
                                                  TextLoader,
                                                  UnstructuredFileLoader)
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

# ... 其他导入 ...


class VectorStoreManager:
    def __init__(
        self,
        persist_dir: str = "./chroma_store",
        embedding_model: str = "shibing624/text2vec-base-chinese",
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
        self.persist_dir = os.path.abspath(persist_dir)
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self._client = PersistentClient(path=self.persist_dir)
        self.vectordbs: Dict[str, Chroma] = {}
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap
        os.makedirs(self.persist_dir, exist_ok=True)

        # 配置日志
        logging.basicConfig(
            filename=os.path.join(self.persist_dir, 'vectorstore.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_collection_name(self, dir_path: str) -> str:
        """生成合法的集合名称"""
        normalized = os.path.normpath(dir_path).replace(os.sep, "_")
        return f"col_{hashlib.md5(normalized.encode()).hexdigest()[:8]}"

    def get_vectorstore(self, dir_path: str) -> Chroma:
        """获取或创建目录对应的向量库"""
        try:
            if dir_path not in self.vectordbs:
                collection_name = self._get_collection_name(dir_path)
                self.vectordbs[dir_path] = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding,
                    client=self._client,
                    persist_directory=self.persist_dir
                )
                self.logger.info(
                    f"Created new collection for path: {dir_path}")
            return self.vectordbs[dir_path]
        except Exception as e:
            self.logger.error(f"Failed to get vectorstore: {str(e)}")
            raise

    def _get_loader(self, file_path: str):
        """根据文件类型返回对应的加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        loader_map = {
            '.pdf': PyPDFLoader,
            '.docx': Docx2txtLoader,
            '.txt': lambda path: TextLoader(path, encoding='utf-8'),
        }
        return loader_map.get(ext, UnstructuredFileLoader)(file_path)

    def load_documents(
        self,
        input_path: str,
        file_pattern: str = "**/*",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        custom_splitter: Optional[TextSplitter] = None,
        show_progress: bool = True,
        index_type: str = "full_text"  # full_text/section/detail
    ) -> List[Document]:
        """
        加载并分块处理文档

        参数:
            input_path: 文件或文件夹路径
            file_pattern: 文件匹配模式
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            custom_splitter: 自定义文本分割器
            show_progress: 是否显示进度条
            index_type: 索引类型 (full_text:全文索引, section:章节索引, detail:详细索引)
        """
        try:
            # 参数处理
            chunk_size = chunk_size or self.default_chunk_size
            chunk_overlap = chunk_overlap or self.default_chunk_overlap

            # 加载文档
            if os.path.isfile(input_path):
                loader = self._get_loader(input_path)
                raw_docs = loader.load()
                if show_progress:
                    print(f"📄 Loaded 1 file from {input_path}")
            else:
                loader = DirectoryLoader(
                    input_path,
                    glob=file_pattern,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                    use_multithreading=True,
                    show_progress=show_progress
                )
                raw_docs = loader.load()
                if show_progress:
                    print(
                        f"📂 Loaded {len(raw_docs)} documents from {input_path}")

            # 分块处理
            splitter = custom_splitter or RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？"]
            )

            docs = splitter.split_documents(raw_docs)

            # 添加元数据
            for doc in tqdm(docs, desc="Processing chunks", disable=not show_progress):
                doc.metadata.update({
                    "content_hash": hashlib.md5(doc.page_content.encode("utf-8")).hexdigest(),
                    "chunk_size": len(doc.page_content),
                    "original_source": doc.metadata.get("source", ""),
                    "index_type": index_type
                })

            return docs

        except Exception as e:
            self.logger.error(f"Failed to load documents: {str(e)}")
            raise

    def add_directory(
        self,
        dir_path: str,
        file_pattern: str = "**/*",
        batch_size: int = 1000,
        force_reload: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        加载整个目录到向量库

        返回:
            {
                "total": 总文档数,
                "added": 成功添加数,
                "duplicates": 重复文档数,
                "failed": 失败文档数
            }
        """
        vectordb = self.get_vectorstore(dir_path)

        # 增量加载检查
        if not force_reload and vectordb._collection.count() > 0:
            if show_progress:
                print(
                    f"⏩ Using existing collection with {vectordb._collection.count()} documents")
            return {
                "total": 0,
                "added": 0,
                "duplicates": 0,
                "failed": 0,
                "status": "used_existing"
            }

        # 加载文档
        docs = self.load_documents(
            dir_path,
            file_pattern=file_pattern,
            show_progress=show_progress
        )

        # 添加文档
        return self.add_documents(
            dir_path,
            docs,
            batch_size=batch_size,
            show_progress=show_progress
        )

    def add_documents(
        self,
        dir_path: str,
        new_docs: List[Document],
        batch_size: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """添加文档到指定路径的集合"""
        stats = {
            "total": len(new_docs),
            "added": 0,
            "duplicates": 0,
            "failed": 0
        }

        if not new_docs:
            self.logger.warning("No documents to add")
            return stats

        vectordb = self.get_vectorstore(dir_path)
        existing_hashes = self._get_existing_hashes(vectordb)

        # 过滤重复文档
        filtered_docs = []
        for doc in new_docs:
            if doc.metadata.get("content_hash") not in existing_hashes:
                filtered_docs.append(doc)
            else:
                stats["duplicates"] += 1

        # 分批添加
        for i in tqdm(
            range(0, len(filtered_docs), batch_size),
            desc="Adding documents",
            disable=not show_progress
        ):
            batch = filtered_docs[i:i + batch_size]
            try:
                vectordb.add_documents(batch)
                stats["added"] += len(batch)
            except Exception as e:
                stats["failed"] += len(batch)
                self.logger.error(f"Batch add failed: {str(e)}")

        return stats

    def _get_existing_hashes(self, vectordb: Chroma) -> Set[str]:
        """获取集合中已有文档的哈希值"""
        try:
            results = vectordb.get(include=["metadatas"])
            return {m["content_hash"] for m in results["metadatas"] if "content_hash" in m}
        except CollectionNotFound:
            return set()
        except Exception as e:
            self.logger.error(f"Failed to get existing hashes: {str(e)}")
            return set()

    def query(
        self,
        dir_path: str,
        query_text: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None,
        index_type: Optional[str] = None,
        **kwargs
    ) -> List[Document]:
        """查询指定目录的集合
        
        参数:
            dir_path: 文档目录路径
            query_text: 查询文本
            k: 返回结果数量
            filter_metadata: 过滤条件
            index_type: 索引类型筛选 (None表示全部, full_text/section/detail)
        """
        vectordb = self.get_vectorstore(dir_path)
        
        # 构建过滤条件
        final_filter = filter_metadata or {}
        if index_type:
            final_filter.update({"index_type": index_type})
            
        return vectordb.similarity_search(
            query=query_text,
            k=k,
            filter=final_filter,
            **kwargs
        )

    def delete_collection(self, dir_path: str) -> bool:
        """删除指定路径的集合"""
        try:
            vectordb = self.get_vectorstore(dir_path)
            self._client.delete_collection(vectordb._collection.name)
            self.vectordbs.pop(dir_path, None)
            self.logger.info(f"Deleted collection for path: {dir_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {str(e)}")
            return False

    def list_collections(self) -> Dict[str, Dict[str, Union[str, int]]]:
        """列出所有集合及其统计信息"""
        collections = {}
        for path, vectordb in self.vectordbs.items():
            try:
                collections[path] = {
                    "collection_name": vectordb._collection.name,
                    "document_count": vectordb._collection.count(),
                    "metadata": vectordb._collection.metadata
                }
            except Exception as e:
                self.logger.error(f"Failed to get info for {path}: {str(e)}")
        return collections

    def optimize(self, dir_path: str):
        """优化指定集合的存储"""
        vectordb = self.get_vectorstore(dir_path)
        vectordb.persist()
        self.logger.info(f"Optimized collection: {dir_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for vectordb in self.vectordbs.values():
            vectordb.persist()
