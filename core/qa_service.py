from config.logger_config import logger
from util.func import get_qa_chain, initialize_vectordb

class QAService:
    def __init__(self):
        self.qa_chain = None
        
    async def initialize(self):
        """初始化QA服务"""
        try:
            vectordb = initialize_vectordb()
            self.qa_chain = get_qa_chain(vectordb)
            logger.info("QA服务初始化完成")
        except Exception as e:
            logger.error(f"QA服务初始化失败: {str(e)}")
            raise

    async def ask_question(self, question: str):
        """处理用户提问"""
        if not self.qa_chain:
            raise RuntimeError("QA服务未初始化")
        
        result = self.qa_chain.invoke(question)
        if isinstance(result, dict):
            return result.get("result") or result.get("answer") or str(result)
        return str(result)

# 全局服务实例
qa_service = QAService()