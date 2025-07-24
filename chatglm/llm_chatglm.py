from typing import Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
from langchain_core.runnables import Runnable
from config.logger_config import logger
from requests.exceptions import ChunkedEncodingError
import torch
import os
from util.to_str import to_str_safe


class ChatGLMLLM(Runnable):
    def __init__(self,
                 model_name_cuda="THUDM/chatglm2-6b",
                 model_name_cpu="Qwen/Qwen1.5-0.5B",
                 revision="main",
                 max_new_tokens=64):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name_cuda if self.device == "cuda" else model_name_cpu

        # 设置 Hugging Face 镜像（可选）
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 或 export 到系统环境中

        # 自动加载 config 获取 max_length
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
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                revision=revision,
                resume_download=True
            )

            # 判断是否 ChatGLM
            self.is_chatglm = "chatglm" in self.model_name.lower()

            # 加载模型
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

            # 设置模型到设备
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

    def _truncate_history_unified(self, max_total_tokens: int, max_rounds: int = 5):
        """
        统一截断历史对话，兼顾 ChatGLM 和标准模型使用场景。
        根据 max_total_tokens 限制总 token 数，按轮次倒序保留历史。
        """
        total_tokens = 0
        new_history = []

        for q, a in reversed(self._history):
            q_len = len(self.tokenizer(q).input_ids)
            a_len = len(self.tokenizer(a).input_ids)
            if total_tokens + q_len + a_len > max_total_tokens:
                break
            new_history.insert(0, (q, a))
            total_tokens += q_len + a_len
            if len(new_history) >= max_rounds:
                break

        self._history = new_history

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

        # 确保历史中所有 q,a 都是字符串，防止类型错误
        self._history = [(str(q), str(a)) for q, a in self._history]

        try:
            # 统一截断历史，防止上下文过长
            self._truncate_history_unified(self.max_total_tokens, max_rounds=5)

            # 统一截断 query，避免单条过长
            query = self._truncate_query(query, max_query_tokens=1024)

            if self.is_chatglm:
                result = self.model.chat(self.tokenizer, query, history=self._history)
                if isinstance(result, tuple) and len(result) == 2:
                    response, self._history = result
                else:
                    response = result
                    self._history.append((query, response))
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
                    max_length=inputs['input_ids'].shape[-1] + self.max_new_tokens,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1
                )

                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

                self._history.append((query, response))
                return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"处理问题失败: {str(e)}")

