import asyncio
import os
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Tuple

from transformers import AutoTokenizer

from config.logger_config import logger
from core.llm_chatglm import ChatGLMLLM
from util.func import get_qa_chain_with_history, initialize_vectordb
from util.search import google_search, ddgs_search, baidu_search

class ContextTemplate:
    """改进的上下文模板类，用于优化搜索结果处理和消息构建"""
    
    def __init__(self):
        self.templates = {
            "system_prompt": "请基于以下信息准确回答用户问题。如果信息不足，请直接说明无法找到相关信息。",
            "chromadb_section": "【知识库搜索结果】\n{content}",
            "websearch_section": "【网络搜索结果】\n{content}",
            "combined_context": "【相关信息】\n{content}",
            "user_query": "问题：{question}",
            "empty_result": "（无相关信息）"
        }
    
    def build_context_message(self, chromadb_results: str, websearch_results: str, question: str) -> str:
        """构建上下文消息，自动处理空搜索结果"""
        sections = []
        
        # 处理chromadb搜索结果
        if chromadb_results and chromadb_results.strip():
            sections.append(self.templates["chromadb_section"].format(content=chromadb_results))
        
        # 处理websearch搜索结果
        if websearch_results and websearch_results.strip():
            sections.append(self.templates["websearch_section"].format(content=websearch_results))
        
        # 如果所有搜索结果都为空，添加提示
        if not sections:
            sections.append(self.templates["empty_result"])
        
        # 构建完整上下文
        context_content = "\n\n".join(sections)
        combined_context = self.templates["combined_context"].format(content=context_content)
        
        # 添加用户问题
        user_query = self.templates["user_query"].format(question=question)
        
        return f"{combined_context}\n\n{user_query}"
    
    def build_complete_prompt(self, question: str, history: List[Tuple[str, str]], 
                            chromadb_results: str, websearch_results: str) -> Dict[str, Any]:
        """构建完整的提示信息"""
        # 构建上下文消息
        context_message = self.build_context_message(chromadb_results, websearch_results, question)
        
        return {
            "system_prompt": self.templates["system_prompt"],
            "context": context_message,
            "history": history,
            "question": question
        }

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
        
        # 添加改进的模板系统
        self.template = ContextTemplate()

        # Registry for vector databases
        self.vector_registry: Dict[str, Any] = {}
        self.retriever_registry: Dict[str, Any] = {}
        self.chain_registry: Dict[str, Any] = {}

        # Search engine configuration
        self._search_engine: Literal['google', 'ddgs', 'baidu'] = 'ddgs'  # Default
        self._search_funcs = {
            'google': google_search,
            'ddgs': ddgs_search,
            'baidu': baidu_search
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
    def search_engine(self, engine: Literal['google', 'ddgs','baidu']) -> None:
        if engine not in ['google', 'ddgs', 'baidu']:
            raise ValueError(
                "Invalid search engine. Must be 'google', 'ddgs' or 'baidu'")
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
        try:
            # 检查目录是否存在
            if not os.path.exists(dir_path):
                logger.warning(f"目录不存在: {dir_path}")
                return
                
            # 检查目录中是否有文件
            has_files = any(os.path.isfile(os.path.join(dir_path, f)) 
                            for f in os.listdir(dir_path) 
                            if os.path.isfile(os.path.join(dir_path, f)))
            
            if not has_files:
                logger.warning(f"目录中没有文件: {dir_path}")
                return
                
            vectordb = initialize_vectordb(dir_path=dir_path)
            
            # 修复检索器配置 - 移除不支持的score_threshold参数
            retriever = vectordb.as_retriever(
                search_kwargs={
                    "k": 5  # 只保留检索数量参数
                }
            )
            chain = get_qa_chain_with_history(self.llm, retriever)
    
            self.vector_registry[dir_path] = vectordb
            self.retriever_registry[dir_path] = retriever
            self.chain_registry[dir_path] = chain
            logger.info(f"✅ 成功注册向量库: {dir_path}")
            
        except Exception as e:
            logger.error(f"❌ 注册向量库失败 {dir_path}: {e}")

    def _get_chain_by_dir(self, dir_path: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Retrieve retriever and chain by directory path."""
        if not dir_path:
            return None, None
        return (
            self.retriever_registry.get(dir_path),
            self.chain_registry.get(dir_path)
        )

    def _perform_web_search(self, question: str) -> str:
        """Perform web search using current search engine."""
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

    async def _build_context_separated(self, question: str, retriever: Any, is_web_search: bool) -> Tuple[str, str]:
        """分别构建chromadb和websearch的上下文，便于模板处理"""
        tasks = []
        results = {"chromadb": "", "websearch": ""}

        if retriever is not None:
            tasks.append(("chromadb", asyncio.to_thread(
                get_limited_context_fast, question, retriever, 2000)))
        
        if is_web_search:
            tasks.append(("websearch", asyncio.to_thread(
                self._perform_web_search, question)))

        if tasks:
            task_results = await asyncio.gather(
                *[task for _, task in tasks], 
                return_exceptions=True
            )
            
            for (task_type, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    logger.warning(f"{task_type} context building failed: {result}")
                    continue
                if isinstance(result, str) and result.strip():
                    results[task_type] = result

        return results["chromadb"], results["websearch"]

    async def ask_with_template(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        is_web_search: bool = False,
        dir_path: str = ""
    ) -> Dict[str, Any]:
        """使用改进的模板系统获取答案

        Args:
            question: 用户问题
            history: 对话历史
            is_web_search: 是否包含网络搜索
            dir_path: 向量数据库目录路径

        Returns:
            包含答案的字典
        """
        history = history or []
        retriever, chain = self._get_chain_by_dir(dir_path)

        # 分别获取chromadb和websearch结果
        chromadb_results, websearch_results = await self._build_context_separated(
            question, retriever, is_web_search
        )

        # 使用模板构建完整提示
        prompt_data = self.template.build_complete_prompt(
            question, history, chromadb_results, websearch_results
        )

        # 构建LLM输入
        inputs = {
            "query": prompt_data["question"],
            "history": prompt_data["history"],
            "context": prompt_data["context"]
        }

        # 调用LLM
        if chain is not None:
            answer = await asyncio.to_thread(chain.invoke, inputs)
            logger.info(f"LLM result: {answer}")
        else:
            answer = await asyncio.to_thread(self.llm.invoke, inputs)

        # 记录搜索结果的统计信息
        search_stats = {
            "chromadb_has_results": bool(chromadb_results and chromadb_results.strip()),
            "websearch_has_results": bool(websearch_results and websearch_results.strip()),
            "chromadb_length": len(chromadb_results),
            "websearch_length": len(websearch_results)
        }
        logger.info(f"Search results stats: {search_stats}")

        return {
            "answer": answer,
            "search_stats": search_stats
        }

    # 保持原有的ask方法兼容性
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
        # 默认使用改进的模板系统
        return await self.ask_with_template(question, history, is_web_search, dir_path)

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