from typing import Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from langchain_core.runnables import Runnable
from config.logger_config import logger
from requests.exceptions import ChunkedEncodingError

import torch
import os

class ChatGLMLLM(Runnable):
    def __init__(self,
                 model_name_cuda="THUDM/chatglm2-6b",
                 model_name_cpu="Qwen/Qwen1.5-0.5B",
                 revision="main",
                 max_new_tokens=64):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name_cuda if self.device == "cuda" else model_name_cpu

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        config = AutoConfig.from_pretrained(self.model_name, revision=revision, trust_remote_code=True)
        model_max_length = getattr(config, "max_position_embeddings",
                                   getattr(config, "seq_length",
                                           getattr(config, "n_positions",
                                                   getattr(config, "model_max_length", 2048))))
        self.max_new_tokens = max_new_tokens
        self.max_total_tokens = model_max_length - self.max_new_tokens

        logger.info(f'Using device: {self.device}')
        logger.info(f'Loading model: {self.model_name}')
        logger.info(f'Model max length: {model_max_length}, max_new_tokens: {self.max_new_tokens}, max_total_tokens: {self.max_total_tokens}')

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                revision=revision
            )

            self.is_chatglm = "chatglm" in self.model_name.lower()

            if self.is_chatglm:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    revision=revision
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    revision=revision
                )

            if self.device == "cuda":
                self.model = self.model.half().cuda()
            else:
                self.model = self.model.float().cpu()

            self.model.eval()
            self._history: List[Tuple[str, str]] = []

        except ChunkedEncodingError as e:
            logger.error(f"模型下载过程中连接中断，请检查网络或尝试手动下载：{e}")
            raise RuntimeError("模型加载失败，下载不完整。建议使用代理或切换到清华镜像。")
        except Exception as e:
            logger.error(f"模型初始化失败：{e}")
            raise RuntimeError(f"模型初始化失败：{str(e)}")

    async def prepare_input(self, question: str, docs: List[Any]) -> str:
        """
        question: 用户提问
        docs: 文档列表，包含page_content和metadata

        返回拼接好的输入字符串，保证文档总token数不超过self.max_total_tokens预留空间
        """
        # 预留512个token给问题和其他prompt，实际你可以调整
        max_doc_tokens = self.max_total_tokens - 512
        if max_doc_tokens <= 0:
            max_doc_tokens = self.max_total_tokens

        truncated_docs = []
        total_tokens = 0
        for doc in docs:
            tokens = self.tokenizer.encode(doc.page_content, add_special_tokens=False)
            if total_tokens + len(tokens) > max_doc_tokens:
                # 只截断当前文档剩余的长度
                remaining = max_doc_tokens - total_tokens
                tokens = tokens[:remaining]
                content = self.tokenizer.decode(tokens)
                truncated_docs.append({
                    **doc.metadata,
                    "page_content": content
                })
                break
            else:
                truncated_docs.append(doc)
                total_tokens += len(tokens)

        # 拼接上下文
        context = "\n\n".join([
            f"文档来源: {doc.get('original_source', '未知')}\n索引类型: {doc.get('index_type', '未知')}\n内容: {doc['page_content']}"
            for doc in truncated_docs
        ])

        full_query = f"问题: {question}\n\n相关上下文:\n{context}"
        return full_query


    def _truncate_history(self) -> List[Tuple[str, str]]:
        """截断历史对话，保证token数量不超过max_total_tokens"""
        max_len = self.max_total_tokens
        truncated_history = []
        total_tokens = 0
        # 从最新到最旧遍历历史
        for q, a in reversed(self._history):
            q_tokens = len(self.tokenizer.encode(q, add_special_tokens=False))
            a_tokens = len(self.tokenizer.encode(a, add_special_tokens=False))
            round_tokens = q_tokens + a_tokens

            if total_tokens + round_tokens > max_len:
                break
            truncated_history.insert(0, (q, a))  # 头部插入，保持时间顺序
            total_tokens += round_tokens
        return truncated_history

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        from util.func import extract_question
        if not isinstance(config, dict):
            config = {}

        query = str(query)
        question = extract_question(query)
        try:
            logger.info(f"调用invoke，query: {query}")
            
            if self.is_chatglm:
                truncated_history = self._truncate_history()
                # logger.info(f"调用invoke，history: {truncated_history}")
                # 计算最大允许的 query token 数
                used_tokens = sum(
                    len(self.tokenizer.encode(q, add_special_tokens=False)) +
                    len(self.tokenizer.encode(a, add_special_tokens=False))
                    for q, a in truncated_history
                )
                # max_query_tokens = self.max_total_tokens - used_tokens
                max_query_tokens = self.max_total_tokens - 0

                # 截断 query
                query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
                if len(query_tokens) > max_query_tokens:
                    logger.warning("Query过长，自动截断")
                    query_tokens = query_tokens[:max_query_tokens]
                    query = self.tokenizer.decode(query_tokens)

                # 正确调用 ChatGLM
                result = self.model.chat(
                    self.tokenizer,
                    query,
                    history=[],
                    max_new_tokens=self.max_new_tokens
                )

                if isinstance(result, tuple) and len(result) == 2:
                    response, _ = result
                else:
                    response = result

                self._history.append((question, response))
                logger.info(f"ChatGLM模型回复: {response}")
                return response

            else:
                # 普通模型不支持多轮历史，history只记录，实际调用时只用当前query
                prompt = f"用户：{query}\n助手："
                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_total_tokens,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1
                )

                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[-1]:],
                    skip_special_tokens=True
                ).strip()

                self._history.append((question, response))
                logger.info(f"普通模型回复: {response}")
                return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"处理问题失败: {str(e)}")

    def get_history(self) -> List[Tuple[str, str]]:
        return self._history
