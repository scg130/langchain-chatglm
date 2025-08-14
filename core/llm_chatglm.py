import asyncio
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain_core.runnables import Runnable
from requests.exceptions import ChunkedEncodingError
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer)

from config.logger_config import logger


class ChatGLMLLM(Runnable):
    def __init__(self,
                 model_name_cuda="THUDM/chatglm3-6b",  # THUDM/glm-4-9b-chat
                 model_name_cpu="Qwen/Qwen1.5-0.5B",
                 revision="main",
                 max_new_tokens=64):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name_cuda if self.device == "cuda" else model_name_cpu

        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        config = AutoConfig.from_pretrained(
            self.model_name, revision=revision, trust_remote_code=True)
        model_max_length = getattr(config, "max_position_embeddings",
                                   getattr(config, "seq_length",
                                           getattr(config, "n_positions",
                                                   getattr(config, "model_max_length", 2048))))
        self.max_new_tokens = max_new_tokens
        self.max_total_tokens = model_max_length - self.max_new_tokens

        logger.info(f'Using device: {self.device}')
        logger.info(f'Loading model: {self.model_name}')
        logger.info(
            f'Model max length: {model_max_length}, max_new_tokens: {self.max_new_tokens}, max_total_tokens: {self.max_total_tokens}')

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

    def convert_history(self, history: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        """
        将 List[Tuple[str, str]] 转换成 ChatGLM3 需要的 List[Dict[str, str]] 格式
        格式转换规则：
            ("用户问题", "助手回答") -> {"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}
        """
        converted = []
        for question, answer in history:
            converted.append({"role": "user", "content": question})
            converted.append({"role": "assistant", "content": answer})
        return converted

    # 截断方法
    def truncate_text(self, text, tokenizer, max_tokens):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return tokenizer.decode(tokens)

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
            if self.is_chatglm:
                # 计算可用tokens，减去新生成tokens
                max_input_tokens = self.max_total_tokens

                # 预留生成空间
                reserved_for_answer = int(
                    self.max_new_tokens * 1.2)  # 预留更多防止溢出
                available_tokens = max_input_tokens - reserved_for_answer

                # 为 context、history 分配 token
                max_context_tokens = int(
                    available_tokens * 0.6)  # 60% 给 context
                max_history_tokens = int(
                    available_tokens * 0.3)  # 30% 给 history
                max_query_tokens = available_tokens - max_context_tokens - max_history_tokens

                # 截断 context / query
                context = self.truncate_text(
                    context, self.tokenizer, max_context_tokens)
                query = self.truncate_text(
                    query, self.tokenizer, max_query_tokens)

                # 截断历史对话
                formatted_history = self.truncate_history(
                    history, max_history_tokens)

                # 拼接 Prompt（更清晰的指令）
                full_query = f"""
                你是一个智能助手，请基于提供的文档内容，并准确回答用户问题。
                - 如果文档中找不到答案，请明确说无法找到。
                - 回答要简洁，不要编造信息。

                【文档内容】：
                {context}

                【当前问题】：
                {query}

                请给出答案：
                """

                logger.info(f"ChatGLM模型输入: {full_query} {formatted_history}")
                result = self.model.chat(
                    self.tokenizer,
                    full_query,
                    history=self.convert_history(formatted_history),
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
                prompt = f"""
                你是一个智能助手，请基于提供的文档内容，参考历史对话，并准确回答用户问题。
                - 如果文档中找不到答案，请明确说无法找到。
                - 回答要简洁，不要编造信息。

                【文档内容】：
                {context}

                【历史对话】：
                {history}

                【当前问题】：
                {query}

                请给出答案：
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
                    do_sample=False,  # 精确问答可用贪心
                    repetition_penalty=1.2
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
                context_tokens = len(self.tokenizer.encode(
                    context, add_special_tokens=False))
                max_history_tokens = max_input_tokens - context_tokens - \
                    len(self.tokenizer.encode(query, add_special_tokens=False))
                safe_history = self.truncate_history(
                    history, max_history_tokens if max_history_tokens > 0 else 0)

                full_query = f"请结合以下内容回答问题：\n{context}\n问题：{query}"

                partial_response = ""
                for response, _ in self.model.stream_chat(
                    self.tokenizer,
                    full_query,
                    history=self.convert_history(safe_history),
                ):
                    partial_response += response
                    yield response

                self._history.append((question, partial_response))
            else:
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
                    token_str = self.tokenizer.decode(
                        token_id, skip_special_tokens=True)
                    yield token_str  # 每步输出累计内容，也可只输出 delta

                self._history.append((question, partial_response))
                logger.info(f"普通模型（流式）回复: {partial_response}")

        except Exception as e:
            logger.error(
                f"stream 模型流式调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"流式处理失败: {str(e)}")

    async def astream(self, input: Any, config: Optional[dict] = None, **kwargs):
        loop = asyncio.get_event_loop()
        for chunk in await loop.run_in_executor(None, lambda: list(self.stream(input, config=config, **kwargs))):
            yield chunk
