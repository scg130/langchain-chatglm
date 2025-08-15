import asyncio
import os
from typing import Any, Dict, List, Tuple

from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer

from config.logger_config import logger
from core.llm_chatglm import ChatGLMLLM
from util.func import get_qa_chain_with_history, initialize_vectordb

try:
    from ddgs import DDGS
except Exception:
    DDGS = None


def get_limited_context(query: str, retriever, tokenizer, max_context_tokens: int = 2048) -> str:
    """从 retriever 取文档，限制上下文 token 数量"""
    docs = retriever.invoke(query)
    context = ""
    total_tokens = 0
    for doc in docs:
        tokens = tokenizer.encode(doc.page_content, add_special_tokens=False)
        if total_tokens + len(tokens) > max_context_tokens:
            break
        context += doc.page_content + "\n"
        total_tokens += len(tokens)
    return context.strip()


def get_limited_context_fast(query: str, retriever, max_chars: int = 2000) -> str:
    """更快的上下文构建（按字符数截断，避免分词开销）"""
    try:
        docs = retriever.invoke(query)
        content_parts: List[str] = []
        total_chars = 0
        for doc in docs:
            text = doc.page_content or ""
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                content_parts.append(text[:remaining])
                total_chars += remaining
                break
            else:
                content_parts.append(text)
                total_chars += len(text)
        return "\n".join(content_parts).strip()
    except Exception as e:
        logger.warning(f"快速上下文构建失败: {e}")
        return ""


class QAService:
    def __init__(self, base_data_dir: str = "./data"):
        self.base_data_dir = base_data_dir
        self.llm = ChatGLMLLM()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm.model_name, trust_remote_code=True)
        # 预加载注册表：dir_path -> vectordb/retriever/qa_chain
        self.vector_registry: Dict[str, Any] = {}
        self.retriever_registry: Dict[str, Any] = {}
        self.chain_registry: Dict[str, Any] = {}
        self._ddgs_available = DDGS is not None
        self.ddgs_client = DDGS()
        if not self._ddgs_available:
            logger.warning("ddgs 未可用，已禁用 web 搜索（请 pip install ddgs）")

    async def initialize(self):
        try:
            if not os.path.exists(self.base_data_dir):
                logger.warning(f"数据目录不存在: {self.base_data_dir}")
                return
            registered: int = 0
            # 将 base_data_dir 本身和其下包含文件的子目录都注册为候选向量库
            dir_candidates: List[str] = []
            for current_dir, subdirs, files in os.walk(self.base_data_dir):
                # 只注册包含至少一个文件的目录
                has_file = any(os.path.isfile(
                    os.path.join(current_dir, f)) for f in files)
                if has_file and current_dir not in dir_candidates:
                    dir_candidates.append(current_dir)
            for d in dir_candidates:
                try:
                    print(f"注册向量库: {d}")
                    vectordb = initialize_vectordb(dir_path=d)
                    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                    chain = get_qa_chain_with_history(self.llm, retriever)
                    self.vector_registry[d] = vectordb
                    self.retriever_registry[d] = retriever
                    self.chain_registry[d] = chain
                    registered += 1
                except Exception as e:
                    logger.warning(f"注册向量库失败 {d}: {e}")
            logger.info(f"QAService初始化完成，已注册向量库个数: {registered}")
        except Exception as e:
            logger.error(f"QAService初始化失败: {e}")
            raise

    def _get_chain_by_dir(self, dir_path: str):
        if not dir_path:
            return None, None
        # 仅使用启动时注册的集合，不在请求时初始化
        retriever = self.retriever_registry.get(dir_path)
        chain = self.chain_registry.get(dir_path)
        return retriever, chain

    async def ask(self, question: str, history: List[Tuple[str, str]] = None, is_web_search: bool = False, dir_path: str = "") -> Dict[str, Any]:
        history = history or []
        retriever, chain = self._get_chain_by_dir(dir_path)

        # 并行构建上下文以降低延迟
        tasks: List[asyncio.Future] = []
        if retriever is not None:
            tasks.append(asyncio.to_thread(
                get_limited_context_fast, question, retriever, 2000))
        if is_web_search and self._ddgs_available:
            def _search():
                try:
                    return "\n".join([
                        (r.get("body") or r.get("title") or "")
                        for r in self.ddgs_client.text(question, max_results=3)
                        if (r.get("body") or r.get("title"))
                    ])
                except Exception as e:
                    logger.warning(f"DuckDuckGo 搜索失败: {e}")
                    return ""
            tasks.append(asyncio.to_thread(_search))

        context_parts: List[str] = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"上下文构建子任务异常: {r}")
                    continue
                if isinstance(r, str) and r:
                    context_parts.append(r)

        context = "\n".join(context_parts).strip()

        inputs = {"query": question, "history": history, "context": context}

        if chain is not None:
            result = await asyncio.to_thread(chain.invoke, inputs)
            answer = result["result"] if isinstance(
                result, dict) and "result" in result else result
        else:
            answer = await asyncio.to_thread(self.llm.invoke, inputs)

        return {"answer": answer}

    async def ask_stream(self, question: str, history: List[Tuple[str, str]] = None, is_web_search: bool = False, dir_path: str = ""):
        history = history or []
        retriever, chain = self._get_chain_by_dir(dir_path)

        # 并行构建上下文以降低延迟
        tasks: List[asyncio.Future] = []
        if retriever is not None:
            tasks.append(asyncio.to_thread(
                get_limited_context_fast, question, retriever, 2000))
        if is_web_search and self._ddgs_available:
            def _search():
                try:
                    return "\n".join([
                        (r.get("body") or r.get("title") or "")
                        for r in self.ddgs_client.text(question, max_results=3)
                        if (r.get("body") or r.get("title"))
                    ])
                except Exception as e:
                    logger.warning(f"DuckDuckGo 搜索失败: {e}")
                    return ""
            tasks.append(asyncio.to_thread(_search))

        context_parts: List[str] = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"上下文构建子任务异常: {r}")
                    continue
                if isinstance(r, str) and r:
                    context_parts.append(r)

        context = "\n".join(context_parts).strip()

        prompt_input = {"query": question,
                        "history": history, "context": context}

        if chain is not None:
            async for chunk in chain.astream(prompt_input):
                yield chunk
        else:
            async for chunk in self.llm.astream(prompt_input):
                yield chunk
