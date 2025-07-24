from config.logger_config import logger
from util.func import get_qa_chain, initialize_vectordb
from typing import Dict, Any

class QAService:
    def __init__(self):
        self.qa_chain = None
        self.input_key = 'query'  # 默认使用'query'，但会在初始化时检测
        self.memory_input_key = 'input'  # 内存系统通常使用'input'

    async def initialize(self):
        """Initialize QA service with proper key detection"""
        try:
            dir_path = "./data"
            vectordb = initialize_vectordb(dir_path=dir_path)
            self.qa_chain = get_qa_chain(vectordb)
            
            # 自动检测输入键
            if hasattr(self.qa_chain, 'input_keys') and self.qa_chain.input_keys:
                self.input_key = self.qa_chain.input_keys[0]
            
            # 检测内存系统需要的键
            if hasattr(self.qa_chain, 'memory') and self.qa_chain.memory:
                if hasattr(self.qa_chain.memory, 'input_key'):
                    self.memory_input_key = self.qa_chain.memory.input_key
            
            logger.info(f"QA服务初始化完成 - 输入键: '{self.input_key}', 内存输入键: '{self.memory_input_key}'")
        except Exception as e:
            logger.error(f"QA服务初始化失败: {str(e)}")
            raise

    async def ask_question(self, question: str) -> Dict[str, Any]:
        """处理用户问题，确保使用正确的键"""
        if not self.qa_chain:
            raise RuntimeError("QA服务未初始化")

        try:
            # 准备符合链期望的输入格式
            inputs = {self.input_key: question}
            
            # 如果链有内存，确保内存系统也能获取到输入
            if hasattr(self.qa_chain, 'memory') and self.qa_chain.memory:
                inputs[self.memory_input_key] = question
            
            result = self.qa_chain.invoke(inputs)
            
            return {
                "answer": result.get("result", result.get("answer", str(result))),
                "sources": result.get("source_documents", [])
            }
            
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            raise RuntimeError(f"处理问题失败: {str(e)}")

# 全局服务实例
qa_service = QAService()