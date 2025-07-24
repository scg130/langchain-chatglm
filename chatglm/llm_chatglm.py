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
                revision=revision,
                resume_download=True
            )

            self.is_chatglm = "chatglm" in self.model_name.lower()

            if self.is_chatglm:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    revision=revision,
                    resume_download=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    revision=revision,
                    resume_download=True
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

    def _truncate_history(self, query: str, max_total_tokens: int, max_rounds: int = 5):
        """
        截断历史，保证历史token总长度 + query token数 <= max_total_tokens
        """
        query_tokens_len = len(self.tokenizer(str(query)).input_ids)
        allowed_tokens_for_history = max_total_tokens - query_tokens_len
        total_tokens = 0
        new_history = []

        for q, a in reversed(self._history):
            q_len = len(self.tokenizer(q).input_ids)
            a_len = len(self.tokenizer(a).input_ids)
            if total_tokens + q_len + a_len > allowed_tokens_for_history:
                break
            new_history.insert(0, (q, a))
            total_tokens += q_len + a_len
            if len(new_history) >= max_rounds:
                break

        self._history = new_history
        logger.debug(f"截断后历史轮次: {len(self._history)}, 历史tokens: {total_tokens}, 预留query tokens: {query_tokens_len}")

    def _truncate_query(self, query: str, max_query_tokens: int = 1024) -> str:
        query_tokens = self.tokenizer(str(query)).input_ids
        if len(query_tokens) > max_query_tokens:
            query_tokens = query_tokens[:max_query_tokens]
            query = self.tokenizer.decode(query_tokens, skip_special_tokens=True)
        return str(query)

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        if not isinstance(config, dict):
            config = {}

        reset_history = config.get("reset_history", False)
        if reset_history:
            self._history = []

        query = str(query)
        self._history = [(str(q), str(a)) for q, a in self._history]

        try:
            self._truncate_history(query, self.max_total_tokens, max_rounds=5)
            query = self._truncate_query(query, max_query_tokens=1024)

            logger.info(f"调用invoke，query: {query}")
            logger.info(f"当前历史长度: {len(self._history)}轮")

            if self.is_chatglm:
                result = self.model.chat(self.tokenizer, query, history=self._history)
                if isinstance(result, tuple) and len(result) == 2:
                    response, self._history = result
                else:
                    response = result
                    self._history.append((query, response))
                logger.info(f"ChatGLM模型回复: {response}")
                return response

            else:
                prompt = ""
                for q, a in self._history:
                    prompt += f"用户：{q}\n助手：{a}\n"
                prompt += f"用户：{query}\n助手："

                tokens = self.tokenizer(prompt, truncation=True, max_length=self.max_total_tokens)
                prompt = self.tokenizer.decode(tokens.input_ids, skip_special_tokens=True)

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1
                )

                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

                self._history.append((query, response))
                logger.info(f"普通模型回复: {response}")
                return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"处理问题失败: {str(e)}")
