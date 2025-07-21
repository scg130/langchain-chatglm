from typing import Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from langchain_core.runnables import Runnable
from config.logger_config import logger
import torch


class ChatGLMLLM(Runnable):
    def __init__(self, 
                 model_name_cuda="THUDM/chatglm2-6b", 
                 model_name_cpu="Qwen/Qwen1.5-0.5B", 
                 revision="main"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name_cuda if self.device == "cuda" else model_name_cpu
        logger.info(f'Using device: {self.device}')
        logger.info(f'Loading model: {self.model_name}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, revision=revision
        )

        # 判断是否是 ChatGLM
        self.is_chatglm = "chatglm" in self.model_name.lower()

        if self.is_chatglm:
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True, revision=revision
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, revision=revision
            )

        if self.device == "cuda":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.float().cpu()

        self.model.eval()
        self._history: List[Tuple[str, str]] = []

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        if not query:
            raise ValueError("输入 query 不能为空")

        reset_history = config.get("reset_history", False) if config else False
        if reset_history:
            self._history = []

        try:
            if self.is_chatglm:
                # 使用 chatglm 原生接口
                result = self.model.chat(self.tokenizer, query, history=self._history)

                if isinstance(result, tuple) and len(result) == 2:
                    response, self._history = result
                else:
                    response = result
                    self._history.append((query, response))
                return response

            else:
                # Qwen 等标准模型，拼接 prompt 推理
                prompt = self._build_prompt(query)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,   # 控制生成长度
                    do_sample=False,     # 关闭采样
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(prompt):].strip()
                self._history.append((query, response))
                return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}", exc_info=True)
            raise RuntimeError(f"处理问题失败: {str(e)}")

    def _build_prompt(self, query: str) -> str:
        if not self._history:
            return f"用户：{query}\n助手："
        prompt = ""
        for q, a in self._history:
            prompt += f"用户：{q}\n助手：{a}\n"
        prompt += f"用户：{query}\n助手："
        return prompt
