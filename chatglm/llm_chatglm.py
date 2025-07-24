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

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        if not isinstance(config, dict):
            config = {}

        # 无论何时调用，都清空历史，或者直接不使用历史
        self._history = []

        query = str(query)

        try:
            # 不再截断历史，直接只截断当前query
            query = self._truncate_query(query, max_query_tokens=1024)

            logger.info(f"调用invoke，query: {query}")

            if self.is_chatglm:
                # 传入空的历史
                result = self.model.chat(self.tokenizer, query, history=[])
                if isinstance(result, tuple) and len(result) == 2:
                    response, _ = result  # 不保存历史
                else:
                    response = result
                logger.info(f"ChatGLM模型回复: {response}")
                return response

            else:
                prompt = f"用户：{query}\n助手："

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

                logger.info(f"普通模型回复: {response}")
                return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}, query: {query}", exc_info=True)
            raise RuntimeError(f"处理问题失败: {str(e)}")

