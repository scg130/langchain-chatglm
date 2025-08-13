import asyncio
import os
from typing import Any, Dict, List, Tuple

from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer

from config.logger_config import logger
from core.llm_chatglm import ChatGLMLLM
from util.func import get_qa_chain_with_history, initialize_vectordb

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None


def format_history(history: List[Tuple[str, str]]) -> str:
    """把对话历史列表格式化成字符串"""
    return "\n".join([f"用户：{q}\n助手：{a}" for q, a in history])


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
        self.web_search_tool = DDGS() if DDGS else None
        if self.web_search_tool is None:
            logger.warning(
                "DuckDuckGoSearchRun 未可用，已禁用 web 搜索（缺少 langchain-community/duckduckgo-search 依赖）")

        self.prompt = PromptTemplate(
            input_variables=["query", "history", "context"],
            template="""
                请根据以下文档内容和历史对话，回答用户提出的问题。

                文档内容：
                {context}

                历史对话：
                {history}

                当前问题：
                {query}

                请简明准确地作答：
                """
        )

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
                    chain = get_qa_chain_with_history(
                        self.llm, retriever, self.prompt)
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

        # 组合context
        context_parts: List[str] = []
        if retriever is not None:
            context_parts.append(get_limited_context(
                question, retriever, self.tokenizer, max_context_tokens=2048))
        if is_web_search and self.web_search_tool:
            try:
                # 把生成器结果转换为列表
                web_results = await asyncio.to_thread(
                    lambda: list(self.web_search_tool.text(
                        question, max_results=5))
                )
                if web_results:
                    # 拼接标题 + 摘要
                    web_text = "\n".join(
                        r['body'] for r in web_results if r.get('body')
                    )
                    context_parts.append(web_text)
                logger.info(f"Web search results: {web_results}")
            except Exception as e:
                logger.warning(f"DuckDuckGo 工具搜索失败: {e}")
        context = "\n".join([c for c in context_parts if c]).strip()

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

        context_parts: List[str] = []
        if retriever is not None:
            context_parts.append(get_limited_context(
                question, retriever, self.tokenizer))
        if is_web_search and self.web_search_tool:
            try:
                web_text = await asyncio.to_thread(self.web_search_tool.invoke, question)
                if web_text:
                    context_parts.append(str(web_text))
            except Exception as e:
                logger.warning(f"DuckDuckGo 工具搜索失败: {e}")
        context = "\n".join([c for c in context_parts if c]).strip()

        prompt_input = {"query": question,
                        "history": history, "context": context}

        if chain is not None:
            async for chunk in chain.astream(prompt_input):
                yield chunk
        else:
            async for chunk in self.llm.astream(prompt_input):
                yield chunk
