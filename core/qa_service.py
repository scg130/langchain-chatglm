from config.logger_config import logger
from util.func import get_qa_chain, initialize_vectordb
from typing import Dict, Any

class QAService:
    def __init__(self):
        self.qa_chain = None

    async def initialize(self):
        """Initialize QA service"""
        try:
            dir_path = "./data"
            vectordb = initialize_vectordb(dir_path=dir_path)
            self.qa_chain = get_qa_chain(vectordb)
            logger.info("QA service initialized successfully")
        except Exception as e:
            logger.error(f"QA service initialization failed: {str(e)}")
            raise

    async def ask_question(self, question: str) -> Dict[str, Any]:
        """Handle user question"""
        if not self.qa_chain:
            raise RuntimeError("QA service not initialized")

        try:
            # Format the input as expected by the chain
            inputs = {"question": question}
            result = self.qa_chain.invoke(inputs)
            
            # Handle different response formats
            if isinstance(result, dict):
                return {
                    "result": result.get("result", ""),
                    "answer": result.get("answer", ""),
                    "source_documents": result.get("source_documents", [])
                }
            return {"result": str(result)}
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise RuntimeError(f"Failed to process question: {str(e)}")


# Global service instance
qa_service = QAService()