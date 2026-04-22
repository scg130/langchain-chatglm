"""
shibing624/text2vec-base-chinese 显式向量化：与 Chroma 的
`embeddings=[...]` / `query_embeddings=[...]` 维数一致，避免仅 query_texts 时与写入向量不对齐。

供 brain/memory 与 vectorstore 共用同一份 SentenceTransformer 单例，避免重复占显存/内存。
"""

from __future__ import annotations

import threading
from typing import List, Sequence

import numpy as np
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

DEFAULT_T2V_MODEL = "shibing624/text2vec-base-chinese"

_lock = threading.Lock()
_model_cache: dict[str, SentenceTransformer] = {}


def get_sentence_transformer(model_name: str = DEFAULT_T2V_MODEL) -> SentenceTransformer:
    with _lock:
        if model_name not in _model_cache:
            _model_cache[model_name] = SentenceTransformer(model_name)
        return _model_cache[model_name]


def encode_to_chroma(
    model: SentenceTransformer,
    texts: str | Sequence[str],
    *,
    normalize_embeddings: bool = True,
) -> List[List[float]]:
    """与 collection.add(embeddings=...) / query(query_embeddings=...) 对齐的二维列表。"""
    if isinstance(texts, str):
        seq: Sequence[str] = [texts]
    else:
        seq = texts
    if not seq:
        return []
    raw = model.encode(
        list(seq),
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        return [arr.tolist()]
    return arr.tolist()


def encode_query_vector(
    model: SentenceTransformer,
    query: str,
    *,
    normalize_embeddings: bool = True,
) -> List[float]:
    vecs = encode_to_chroma(model, query, normalize_embeddings=normalize_embeddings)
    return vecs[0] if vecs else []


class Text2VecEmbeddings(Embeddings):
    """供 LangChain Chroma 使用：与手写 Memory 层同一套向量空间。"""

    def __init__(
        self,
        model_name: str = DEFAULT_T2V_MODEL,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self._normalize = normalize_embeddings
        self._model = get_sentence_transformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return encode_to_chroma(self._model, texts, normalize_embeddings=self._normalize)

    def embed_query(self, text: str) -> List[float]:
        return encode_query_vector(
            self._model, text, normalize_embeddings=self._normalize
        )
