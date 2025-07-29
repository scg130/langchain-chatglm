from config.logger_config import logger
from util.func import get_qa_chain, initialize_vectordb
from typing import Dict, Any
import asyncio

class QAService:
    def __init__(self):
        self.qa_chain = None
        self.input_key = 'query'  # 默认使用'query'，但会在初始化时检测
        self.memory_input_key = 'input'  # 内存系统通常使用'input'

    async def initialize(self):
        """Initialize QA service with proper key detection"""
        try:
            dir_path = "./data"
            self.vectordbs = initialize_vectordb(dir_path=dir_path)

            self.retrievers = {
                index_type: self.vectordbs[index_type].as_retriever(search_kwargs={
                    'k': 3,
                    'filter': {'index_type': {'$in': [index_type]}}
                })
                for index_type in ["full_text", "section", "detail"]
            }

            # 你也可以选择其中一个默认的vectorstore用于初始化qa_chain，例如 full_text
            self.qa_chain = get_qa_chain()
            
            self.qa_chains = {
                "full_text": get_qa_chain(self.vectordbs["full_text"]),
                "section": get_qa_chain(self.vectordbs["section"]),
                "detail": get_qa_chain(self.vectordbs["detail"]),
            }

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

    async def _determine_index_type(self, question: str) -> str:
        """使用LLM判断问题适用的索引类型"""
        prompt = (
            "请根据问题内容判断最适合的索引类型：\n"
            "1. 全文索引(full_text) - 适用于概括性、整体性问题\n"
            "2. 章节索引(section) - 适用于询问文档结构或部分内容的问题\n"
            "3. 详细索引(detail) - 适用于需要具体细节的问题\n\n"
            f"问题：{question}\n\n"
            "只需回答 full_text/section/detail 中的一个:"
        )
        
        try:
            response = self.qa_chain.invoke({self.input_key: prompt})
            response = response.get("result", str(response)).strip().lower()
            logger.info(f"索引类型判断结果: {response}")
            if response in ["full_text", "section", "detail"]:
                return response
            else:
                # 使用启发式规则作为后备
                if "详细" in question or "具体" in question or "精确" in question:
                    return "detail"
                elif "章节" in question or "部分" in question or "段落" in question:
                    return "section"
                else:
                    return "full_text"
                    
        except Exception as e:
            logger.error(f"索引类型判断失败: {str(e)}")
            # 默认返回全文索引
            return "full_text"

    async def deduplicate_documents(self, doc_list):
        seen = set()
        unique_docs = []
        for doc in doc_list:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
        return unique_docs

    async def async_get_docs(self, retriever, question):
        return await asyncio.to_thread(retriever.invoke, question)

    async def ask_question(self, question: str) -> Dict[str, Any]:
        """处理用户问题，实现智能路由"""
        logger.info(f"ask_question: {question}")
        try:
            # 判断索引类型
            index_type = await self._determine_index_type(question)
            logger.info(f"Determined index type: {index_type}")
            
            chain = self.qa_chains[index_type]
            docs = await self.async_get_docs(self.retrievers[index_type], question)
            unique_docs = await self.deduplicate_documents(docs)

            # 准备上下文
            context = "\n\n".join([
                f"文档来源: {doc.metadata['original_source']}\n"
                f"索引类型: {doc.metadata['index_type']}\n"
                f"内容: {doc.page_content}"
                for doc in unique_docs
            ])
            
            # 构建完整问题
            full_query = f"问题: {question}\n\n相关上下文:\n{context}"
            
            # 调用LLM
            inputs = {self.input_key: full_query}
            if hasattr(chain, 'memory') and chain.memory:
                inputs[self.memory_input_key] = full_query
                
            result = chain.invoke(inputs)
            
            sources = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in unique_docs
            ]


            return {
                "answer": result.get("result", result.get("answer", str(result))),
                # "sources": sources,
                "index_type": index_type
            }
            
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            raise RuntimeError(f"处理问题失败: {str(e)}")

# 全局服务实例
qa_service = QAService()