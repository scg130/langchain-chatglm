from typing import Dict, Any
from config.logger_config import logger
from util.func import get_qa_chain, initialize_vectordb, llm  # 这里可以根据实际情况调整
import asyncio


class QAService:
    def __init__(self):
        self.qa_chain = llm  # 你可以用封装好的LLM
        self.input_key = 'query'
        self.memory_input_key = 'input'
        self.vectordbs = {}
        self.retrievers = {}
        self.qa_chains = {}
        self.dir_path = "./data"  # 默认文档根目录

    async def initialize(self):
        try:
            self.vectordbs = initialize_vectordb(dir_path=self.dir_path)
            self.retrievers = {
                f"{self.dir_path}_{t}": self.vectordbs[f"{self.dir_path}_{t}"].as_retriever(search_kwargs={"k": 3, "filter": {"index_type": {"$in": [t]}}})
                for t in ["full_text", "section", "detail"]
            }
            self.qa_chains = {
                f"{self.dir_path}_{t}": get_qa_chain(self.retrievers[f"{self.dir_path}_{t}"])
                for t in ["full_text", "section", "detail"]
            }
            logger.info("QAService初始化完成")
        except Exception as e:
            logger.error(f"QAService初始化失败: {e}")
            raise

    def _determine_index_type_by_rule(self, question: str) -> str:
        """根据简单规则判断索引类型"""
        question_lower = question.lower()
        if any(word in question_lower for word in ["详细", "具体", "精确"]):
            return "detail"
        elif any(word in question_lower for word in ["章节", "部分", "段落"]):
            return "section"
        else:
            return "full_text"


    async def ask_question(self, question: str) -> Dict[str, Any]:
        try:
            index_type = self._determine_index_type_by_rule(question)
            chain = self.qa_chains.get(f"{self.dir_path}_{index_type}")
            if not chain:
                chain = self.qa_chains.get(f"{self.dir_path}_{index_type}")

            inputs = {self.input_key: question}
            result = chain.invoke(inputs)
            
            final_answer = result.get("result") if isinstance(result, dict) else result

            return {"answer": final_answer, "index_type": index_type}
        except Exception as e:
            logger.error(f"问答失败: {e}")
            raise
        

qa_service = QAService()
