from typing import Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from langchain_core.runnables import Runnable
from config.logger_config import logger
from requests.exceptions import ChunkedEncodingError
import asyncio

import torch
import os


class ChatGLMLLM(Runnable):
    def __init__(self,
                 model_name_cuda="THUDM/chatglm3-6b", # THUDM/glm-4-9b-chat
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

    def truncate_history(self, history: List[Tuple[str, str]], max_tokens: int) -> List[Tuple[str, str]]:
        """截断 history，保证总 token 数不超 max_tokens"""
        total_tokens = 0
        truncated = []
        # 逆序保留最近对话
        for q, a in reversed(history):
            tokens = self.tokenizer.encode(q + a, add_special_tokens=False)
            if total_tokens + len(tokens) > max_tokens:
                break
            truncated.insert(0, (q, a))
            total_tokens += len(tokens)
        return truncated

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs) -> Any:
        if not isinstance(config, dict):
            config = {}

        if isinstance(input, str):
            query = input
            history = []
            context = ""
        elif isinstance(input, dict):
            query = input.get("query", "")
            history = input.get("history", [])
            context = input.get("context", "")
        else:
            query = str(input)
            history = []
            context = ""

        question = query

        try:
            logger.info(f"调用invoke，query: {query}")

            if self.is_chatglm:
                # 计算可用tokens，减去新生成tokens
                max_input_tokens = self.max_total_tokens

                # 估算 context tokens
                context_tokens = len(self.tokenizer.encode(context, add_special_tokens=False))

                # 剩余可用给 history 的 token 数
                max_history_tokens = max_input_tokens - context_tokens - len(self.tokenizer.encode(query, add_special_tokens=False))

                # 截断历史对话
                safe_history = self.truncate_history(history, max_history_tokens if max_history_tokens > 0 else 0)

                # 拼接 context 和 query，调用 chat
                full_query = f"请结合以下内容回答问题：\n{context}\n问题：{query}"

                result = self.model.chat(
                    self.tokenizer,
                    full_query,
                    history=safe_history
                )

                if isinstance(result, tuple) and len(result) == 2:
                    response, _ = result
                else:
                    response = result

                self._history.append((question, response))
                logger.info(f"ChatGLM模型回复: {response}")
                return response

            else:
                # 普通模型拼接prompt
                prompt = f"""请结合以下内容回答问题：

                    文档内容：
                    {context}

                    历史对话：
                    {history}

                    当前问题：
                    {query}

                    助手：
                    """
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

    def stream(self, input: Any, config: Optional[dict] = None, **kwargs):
        if isinstance(input, str):
            query = input
            history = []
            context = ""
        elif isinstance(input, dict):
            query = input.get("query", "")
            history = input.get("history", [])
            context = input.get("context", "")
        else:
            query = str(input)
            history = []
            context = ""

        question = query

        try:
            if self.is_chatglm:
                max_input_tokens = self.max_total_tokens
                context_tokens = len(self.tokenizer.encode(context, add_special_tokens=False))
                max_history_tokens = max_input_tokens - context_tokens - len(self.tokenizer.encode(query, add_special_tokens=False))
                safe_history = self.truncate_history(history, max_history_tokens if max_history_tokens > 0 else 0)

                full_query = f"请结合以下内容回答问题：\n{context}\n问题：{query}"

                for response, _ in self.model.stream_chat(
                    self.tokenizer,
                    full_query,
                    history=safe_history
                ):
                    yield response

            else:
                # === 以下为 Qwen 或其他 CausalLM 模型的伪流式响应 ===
                prompt = f"""请结合以下内容回答问题：

                    文档内容：
                    {context}

                    历史对话：
                    {history}

                    当前问题：
                    {query}

                    助手：
                    """
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
                    repetition_penalty=1.1,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                generated_ids = outputs.sequences[0][inputs['input_ids'].shape[-1]:]
                partial_response = ""

                for token_id in generated_ids:
                    token_str = self.tokenizer.decode(token_id, skip_special_tokens=True)
                    yield token_str  # 每步输出累计内容，也可只输出 delta

                self._history.append((question, partial_response))
                logger.info(f"普通模型（流式）回复: {partial_response}")

        except Exception as e:
            logger.error(f"stream 模型流式调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"流式处理失败: {str(e)}")


    async def astream(self, input: Any, config: Optional[dict] = None, **kwargs):
        loop = asyncio.get_event_loop()
        for chunk in await loop.run_in_executor(None, lambda: list(self.stream(input, config=config, **kwargs))):
            yield chunk
