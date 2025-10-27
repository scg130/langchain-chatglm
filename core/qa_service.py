import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

from transformers import AutoTokenizer

from config.logger_config import logger
from core.llm_chatglm import ChatGLMLLM
from util.func import get_qa_chain_with_history, initialize_vectordb
from util.search import google_search

# DuckDuckGo Search import with fallback
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
    logger.warning("DDGS not available. Please install with: pip install ddgs")


class QAService:
    def __init__(self, base_data_dir: str = "./data"):
        """Initialize QA Service with vector database support and search capabilities.

        Args:
            base_data_dir: Directory containing documents for vector databases
        """
        self.base_data_dir = base_data_dir
        self.llm = ChatGLMLLM()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm.model_name_or_path, trust_remote_code=True)

        # Registry for vector databases
        self.vector_registry: Dict[str, Any] = {}
        self.retriever_registry: Dict[str, Any] = {}
        self.chain_registry: Dict[str, Any] = {}

        # Search engine configuration
        self._search_engine: Literal['google', 'ddgs'] = 'ddgs'  # Default
        self._ddgs_available = DDGS is not None
        self._search_funcs = {
            'google': google_search,
            'ddgs': self._ddgs_search if self._ddgs_available else None
        }

        # Initialization state
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    @property
    def available_search_engines(self) -> List[str]:
        """Get list of available search engines."""
        engines = ['google']
        if self._ddgs_available:
            engines.append('ddgs')
        return engines

    @property
    def search_engine(self) -> str:
        """Get current active search engine."""
        return self._search_engine

    @search_engine.setter
    def search_engine(self, engine: Literal['google', 'ddgs']) -> None:
        """Set the active search engine.

        Args:
            engine: Either 'google' or 'ddgs'

        Raises:
            ValueError: If invalid engine specified
            RuntimeError: If DDGS engine requested but not available
        """
        if engine not in ['google', 'ddgs']:
            raise ValueError(
                "Invalid search engine. Must be 'google' or 'ddgs'")
        if engine == 'ddgs' and not self._ddgs_available:
            raise RuntimeError(
                "DDGS not available. Please install ddgs package first")
        self._search_engine = engine
        logger.info(f"Search engine switched to: {engine}")

    async def initialize(self) -> None:
        """Initialize all vector databases from the data directory."""
        if self._initialized:
            return

        async with self._initialization_lock:
            if self._initialized:  # Double-check locking
                return

            try:
                if not os.path.exists(self.base_data_dir):
                    logger.warning(
                        f"Data directory does not exist: {self.base_data_dir}")
                    return

                registered = 0
                dir_candidates = self._find_valid_directories()

                for d in dir_candidates:
                    try:
                        await self._register_directory(d)
                        registered += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to register vector database {d}: {e}")

                logger.info(
                    f"QAService initialized. Registered {registered} vector databases")
                self._initialized = True
            except Exception as e:
                logger.error(f"QAService initialization failed: {e}")
                raise

    def _find_valid_directories(self) -> List[str]:
        """Find directories containing files for vector database registration."""
        dir_candidates = []
        for current_dir, _, files in os.walk(self.base_data_dir):
            if any(os.path.isfile(os.path.join(current_dir, f)) for f in files):
                dir_candidates.append(current_dir)
        return dir_candidates

    async def _register_directory(self, dir_path: str) -> None:
        """Register a single directory as a vector database."""
        vectordb = initialize_vectordb(dir_path=dir_path)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        chain = get_qa_chain_with_history(self.llm, retriever)

        self.vector_registry[dir_path] = vectordb
        self.retriever_registry[dir_path] = retriever
        self.chain_registry[dir_path] = chain

    def _get_chain_by_dir(self, dir_path: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Retrieve retriever and chain by directory path."""
        if not dir_path:
            return None, None
        return (
            self.retriever_registry.get(dir_path),
            self.chain_registry.get(dir_path)
        )

    def _ddgs_search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Perform search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with 'title' and 'body' fields
        """
        results = []
        if not self._ddgs_available:
            raise RuntimeError("DDGS not available")
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(
                query,
                region="cn-zh",      # 关键：中文结果
                safesearch="off",
                max_results=max_results,
                timelimit="y"         # 限定近一年（可选）
            ):
                    title = r.get("title", "")
                    body = r.get("snippet", "")
                    url = r.get("href", "")
                    results.append({"title": title, "body": body, "url": url})
            return results        
        except Exception as e:
            logger.error(f"DDGS search failed: {e}")
            return []

    async def _build_context(self, question: str, retriever: Any, is_web_search: bool) -> str:
        """Build context from retriever and web search."""
        tasks = []

        if retriever is not None:
            tasks.append(asyncio.to_thread(
                get_limited_context_fast, question, retriever, 2000))
        if is_web_search:
            tasks.append(asyncio.to_thread(
                self._perform_web_search, question))

        context_parts = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Context building subtask failed: {r}")
                    continue
                if isinstance(r, str) and r:
                    context_parts.append(r)

        return "\n".join(context_parts).strip()

    def _perform_web_search(self, question: str) -> str:
        """Perform web search using current search engine."""
        if self._search_engine == 'ddgs' and not self._ddgs_available:
            logger.warning("DDGS not available, falling back to Google")
            self._search_engine = 'google'
        search_func = self._search_funcs.get(self._search_engine)
        if search_func is None:
            logger.warning(f"Search engine {self._search_engine} not available")
            return ""

        try:
            results = search_func(question, max_results=3)
            return "\n".join(
                (r.get("body") or r.get("title") or "")
                for r in results
                if (r.get("body") or r.get("title"))
            )
        except Exception as e:
            logger.warning(f"{self._search_engine.upper()} search failed: {e}")
            return ""

    async def ask(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        is_web_search: bool = False,
        dir_path: str = ""
    ) -> Dict[str, Any]:
        """Get a complete answer to a question.

        Args:
            question: The question to answer
            history: Conversation history as list of (question, answer) tuples
            is_web_search: Whether to include web search results
            dir_path: Specific vector database directory to use

        Returns:
            Dictionary with 'answer' key containing the response
        """
        history = history or []
        retriever, chain = self._get_chain_by_dir(dir_path)

        context = await self._build_context(question, retriever, is_web_search)
        inputs = {"query": question, "history": history, "context": context}

        if chain is not None:
            result = await asyncio.to_thread(chain.invoke, inputs)
            answer = result.get("result", result)
        else:
            answer = await asyncio.to_thread(self.llm.invoke, inputs)

        return {"answer": answer}

    async def ask_stream(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        is_web_search: bool = False,
        dir_path: str = ""
    ) -> AsyncGenerator[Any, None]:
        """Stream the answer to a question.

        Args:
            question: The question to answer
            history: Conversation history as list of (question, answer) tuples
            is_web_search: Whether to include web search results
            dir_path: Specific vector database directory to use

        Yields:
            Chunks of the response as they're generated
        """
        history = history or []
        retriever, chain = self._get_chain_by_dir(dir_path)

        context = await self._build_context(question, retriever, is_web_search)
        prompt_input = {"query": question,
                        "history": history, "context": context}

        if chain is not None:
            async for chunk in chain.astream(prompt_input):
                yield chunk
        else:
            async for chunk in self.llm.astream(prompt_input):
                yield chunk


def get_limited_context_fast(query: str, retriever: Any, max_chars: int = 2000) -> str:
    """Fast context builder with character-based truncation.

    Args:
        query: The query to retrieve context for
        retriever: The retriever to use
        max_chars: Maximum number of characters to return

    Returns:
        Retrieved context as a string
    """
    try:
        docs = retriever.invoke(query)
        content_parts = []
        total_chars = 0
        for doc in docs:
            text = doc.page_content or ""
            remaining = max_chars - total_chars
            if remaining <= 0:
                break
            if len(text) > remaining:
                content_parts.append(text[:remaining])
                break
            content_parts.append(text)
            total_chars += len(text)
        return "\n".join(content_parts).strip()
    except Exception as e:
        logger.warning(f"Fast context building failed: {e}")
        return ""
