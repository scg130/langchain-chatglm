"""
长短期记忆：Chroma 持久化 + 按 user_id 元数据隔离 + 语义召回近期对话。

向量化：使用 SentenceTransformer 对文本显式 encode，add 时写 embeddings，query 时传
query_embeddings（与 shibing624/text2vec-base-chinese 写入向量同空间，避免仅 query_texts
与自建向量混用导致检索失真）。

用法（与 GBrain 配合时务必传入 current_query 做相关记忆召回）:

    mem = Memory()
    ctx = mem.get_context("u1", "当前用户问题", n_results=5)
    mem.save("u1", "用户原话", "助手回答")  # 或 save(..., result=str|dict)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, List, Optional, Union

from chromadb import PersistentClient

from util.sentence_text2vec import (
    DEFAULT_T2V_MODEL,
    encode_query_vector,
    encode_to_chroma,
    get_sentence_transformer,
)

_DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_user_memory")
DEFAULT_PERSIST = os.path.abspath(_DEFAULT_DIR)

# 与旧版（Chroma 默认 text 嵌入）不兼容，使用新 collection 名；删除旧库可手动删目录
_COLLECTION_NAME = "gbrain_text2vec"


def _as_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for k in ("response", "answer", "result", "content"):
            if k in response and response[k] is not None:
                return str(response[k])
        return str(response)
    return str(response)


class Memory:
    """按用户隔离的向量记忆；召回用「当前问题」与历史片段做相似度，而非用 user_id 当查询文本。"""

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST,
        collection_name: str = _COLLECTION_NAME,
        embedding_model: str = DEFAULT_T2V_MODEL,
    ) -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self._client = PersistentClient(path=persist_directory)
        self._col = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._model = get_sentence_transformer(embedding_model)

    def get_context(
        self,
        user_id: str,
        current_query: str = "",
        n_results: int = 5,
    ) -> str:
        """
        与当前问题相关的历史片段（语义检索）；若未提供 current_query，则返回最近若干条时间序对话。
        """
        if not user_id or not str(user_id).strip():
            return ""
        where = {"user_id": str(user_id).strip()}

        if (current_query or "").strip():
            q_emb = encode_query_vector(
                self._model, current_query.strip(), normalize_embeddings=True
            )
            res = self._col.query(
                query_embeddings=[q_emb],
                n_results=min(n_results, 20),
                where=where,
            )
            docs = (res.get("documents") or [[]])[0] or []
        else:
            docs = self._recent_documents(user_id, limit=n_results)

        return self._format_blocks(docs)

    def _recent_documents(self, user_id: str, limit: int) -> List[str]:
        res = self._col.get(
            where={"user_id": str(user_id).strip()},
            include=["documents", "metadatas", "ids"],
        )
        ids: List[str] = list(res.get("ids") or [])
        if not ids:
            return []
        metas: List[Optional[dict]] = list(res.get("metadatas") or [])
        docs: List[Optional[str]] = list(res.get("documents") or [])

        indexed: List[tuple] = []
        for i, mid in enumerate(ids):
            m = (metas[i] if i < len(metas) else None) or {}
            ts = m.get("ts", 0.0)
            try:
                ts = float(ts)
            except (TypeError, ValueError):
                ts = 0.0
            content = (docs[i] if i < len(docs) else None) or ""
            indexed.append((ts, content))

        indexed.sort(key=lambda x: x[0])
        return [c for _, c in indexed[-limit:] if c]

    @staticmethod
    def _format_blocks(docs: List[str]) -> str:
        if not docs:
            return ""
        return "\n\n---\n\n".join(docs)

    def save(
        self,
        user_id: str,
        query: str,
        response: Union[str, dict, Any],
    ) -> None:
        if not user_id or not str(user_id).strip():
            return
        text = (query or "").strip()
        ans = _as_text(response).strip()
        if not text and not ans:
            return

        doc = f"用户: {text}\n助手: {ans}"
        embs = encode_to_chroma(self._model, [doc], normalize_embeddings=True)
        rec_id = f"{user_id}_{time.time():.6f}_{uuid.uuid4().hex[:12]}"
        self._col.add(
            ids=[rec_id],
            embeddings=embs,
            documents=[doc],
            metadatas=[{"user_id": str(user_id).strip(), "ts": time.time()}],
        )

    def clear_user(self, user_id: str) -> int:
        """删除某用户全部记忆，返回尝试删除的条数（以 get 到数量为准）。"""
        uid = str(user_id).strip()
        if not uid:
            return 0
        res = self._col.get(where={"user_id": uid}, include=[])
        ids: List[str] = list(res.get("ids") or [])
        if ids:
            self._col.delete(ids=ids)
        return len(ids)
