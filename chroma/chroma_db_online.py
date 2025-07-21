import hashlib
import logging
import os
from typing import Dict, List, Optional, Set, Union

# 自动适配不同版本的 CollectionNotFound
try:
    from chromadb.errors import CollectionNotFound
except ImportError:
    class CollectionNotFound(Exception):
        pass

from chromadb import HttpClient
from chromadb.config import Settings

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader, Docx2txtLoader, PyPDFLoader, TextLoader, UnstructuredFileLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


class VectorStoreManager:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        embedding_model: str = "shibing624/text2vec-base-chinese",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        self._client = HttpClient(
            host=host,
            port=port,
            settings=Settings(allow_reset=True)
        )
        self.vectordbs: Dict[str, Chroma] = {}
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _get_collection_name(self, dir_path: str) -> str:
        normalized = os.path.normpath(dir_path).replace(os.sep, "_")
        return f"col_{hashlib.md5(normalized.encode()).hexdigest()[:8]}"

    def get_vectorstore(self, dir_path: str) -> Chroma:
        try:
            if dir_path not in self.vectordbs:
                collection_name = self._get_collection_name(dir_path)
                self.vectordbs[dir_path] = Chroma(
                    collection_name=collection_name,
                    embedding_function=self.embedding,
                    client=self._client,
                )
                self.logger.info(f"Created new collection for path: {dir_path}")
            return self.vectordbs[dir_path]
        except Exception as e:
            self.logger.error(f"Failed to get vectorstore: {str(e)}")
            raise

    def _get_loader(self, file_path: str):
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
        show_progress: bool = True
    ) -> List[Document]:
        try:
            chunk_size = chunk_size or self.default_chunk_size
            chunk_overlap = chunk_overlap or self.default_chunk_overlap

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
                    print(f"📂 Loaded {len(raw_docs)} documents from {input_path}")

            splitter = custom_splitter or RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", "…", " ", ""]
            )

            docs = splitter.split_documents(raw_docs)

            for doc in tqdm(docs, desc="Processing chunks", disable=not show_progress):
                doc.metadata.update({
                    "content_hash": hashlib.md5(doc.page_content.encode("utf-8")).hexdigest(),
                    "chunk_size": len(doc.page_content),
                    "original_source": doc.metadata.get("source", "")
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
        vectordb = self.get_vectorstore(dir_path)

        if not force_reload and vectordb._collection.count() > 0:
            if show_progress:
                print(f"⏩ Using existing collection with {vectordb._collection.count()} documents")
            return {
                "total": 0,
                "added": 0,
                "duplicates": 0,
                "failed": 0,
                "status": "used_existing"
            }

        docs = self.load_documents(
            dir_path,
            file_pattern=file_pattern,
            show_progress=show_progress
        )

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

        filtered_docs = []
        for doc in new_docs:
            if doc.metadata.get("content_hash") not in existing_hashes:
                filtered_docs.append(doc)
            else:
                stats["duplicates"] += 1

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
        **kwargs
    ) -> List[Document]:
        vectordb = self.get_vectorstore(dir_path)
        return vectordb.similarity_search(
            query=query_text,
            k=k,
            filter=filter_metadata,
            **kwargs
        )

    def delete_collection(self, dir_path: str) -> bool:
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
        vectordb = self.get_vectorstore(dir_path)
        vectordb.persist()
        self.logger.info(f"Optimized collection: {dir_path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for vectordb in self.vectordbs.values():
            vectordb.persist()
