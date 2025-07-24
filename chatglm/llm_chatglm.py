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

    def _truncate_history(self, tokenizer, history, max_tokens, max_rounds=5):
        total_tokens = 0
        new_history = []
        rounds = 0

        # 从后往前保留历史轮次，优先完整问答对
        for q, a in reversed(history):
            if rounds >= max_rounds:
                break
            q_tokens = len(tokenizer(q).input_ids)
            a_tokens = len(tokenizer(a).input_ids)
            if total_tokens + q_tokens + a_tokens > max_tokens:
                break
            new_history.insert(0, (q, a))
            total_tokens += q_tokens + a_tokens
            rounds += 1
        return new_history

    def _build_prompt(self, query: str) -> str:
        if not self._history:
            return f"用户：{query}\n助手："

        self._history = self._truncate_history(self.tokenizer, self._history, self.max_total_tokens, max_rounds=5)
        prompt = ""
        for q, a in self._history:
            prompt += f"用户：{q}\n助手：{a}\n"
        prompt += f"用户：{query}\n助手："
        return prompt

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        if not query:
            raise ValueError("输入 query 不能为空")

        if not isinstance(config, dict):
            config = {}

        reset_history = config.get("reset_history", False)
        if reset_history:
            self._history = []

        try:
            if self.is_chatglm:
                # ChatGLM 原生接口，历史处理内置，无需额外截断
                result = self.model.chat(self.tokenizer, query, history=self._history)
                if isinstance(result, tuple) and len(result) == 2:
                    response, self._history = result
                else:
                    response = result
                    self._history.append((query, response))
                return response

            else:
                # 标准模型，先截断历史，限制最大长度
                max_input_length = self.max_total_tokens
                self._history = self._truncate_history(self.tokenizer, self._history, max_input_length, max_rounds=5)

                prompt = self._build_prompt(query)

                # 自动截断超长输入
                tokens = self.tokenizer(prompt, truncation=True, max_length=max_input_length)
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

                # 精确切割生成文本，排除prompt部分
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

                self._history.append((query, response))
                return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"处理问题失败: {str(e)}")
