from typing import Dict, Any, List, Tuple
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer
from config.logger_config import logger
from util.func import initialize_vectordb, get_qa_chain_with_history
import asyncio
from core.llm_chatglm import ChatGLMLLM


def format_history(history: List[Tuple[str, str]]) -> str:
    """把对话历史列表格式化成字符串"""
    return "\n".join([f"用户：{q}\n助手：{a}" for q, a in history])


def get_limited_context(query: str, retriever, tokenizer, max_context_tokens: int = 2048) -> str:
    """从 retriever 取文档，限制上下文 token 数量"""
    docs = retriever.get_relevant_documents(query)
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
    def __init__(self, dir_path: str = "./data"):
        self.dir_path = dir_path
        self.llm = ChatGLMLLM()
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm.model_name, trust_remote_code=True)
        self.vectordbs = initialize_vectordb(dir_path=self.dir_path)

        # 通用prompt，支持query, history, context输入
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

        # retrievers, qa_chains会在initialize时初始化
        self.retrievers: Dict[str, Any] = {}
        self.qa_chains: Dict[str, Any] = {}

    async def initialize(self):
        try:
            self.retrievers = {
                self._get_key(t): self.vectordbs[self._get_key(t)].as_retriever(
                    search_kwargs={"k": 3, "filter": {"index_type": {"$in": [t]}}}
                )
                for t in ["full_text", "section", "detail"]
            }
            self.qa_chains = {
                self._get_key(t): get_qa_chain_with_history(self.llm,self.retrievers[self._get_key(t)],self.prompt)
                for t in ["full_text", "section", "detail"]
            }
            logger.info("QAService初始化完成")
        except Exception as e:
            logger.error(f"QAService初始化失败: {e}")
            raise

    def _get_key(self, index_type: str) -> str:
        """生成统一的key，方便动态获取"""
        return f"{self.dir_path}_{index_type}"

    def _determine_index_type_by_rule(self, question: str) -> str:
        question_lower = question.lower()
        if any(word in question_lower for word in ["详细", "具体", "精确"]):
            return "detail"
        elif any(word in question_lower for word in ["章节", "部分", "段落"]):
            return "section"
        else:
            return "full_text"

    async def ask(self, question: str, history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        history = history or []
        index_type = self._determine_index_type_by_rule(question)
        key = self._get_key(index_type)

        chain = self.qa_chains.get(key)
        if not chain:
            raise RuntimeError(f"找不到对应索引类型的问答链: {index_type}")

        retriever = self.retrievers.get(key)
        if not retriever:
            raise RuntimeError(f"找不到对应索引类型的检索器: {index_type}")

        context = get_limited_context(question, retriever, self.tokenizer, max_context_tokens=2048)

        inputs = {
            "query": question,
            "history": history,
            "context": context,
        }

        result = await asyncio.to_thread(chain.invoke, inputs)

        if isinstance(result, dict) and "result" in result:
            answer = result["result"]
        else:
            answer = result

        return {"answer": answer, "index_type": index_type}
