from typing import Any, Optional, List, Tuple
from transformers import AutoTokenizer, AutoModel
from langchain_core.runnables import Runnable
from config.logger_config import logger
import torch


class ChatGLMLLM(Runnable):
    def __init__(self, 
                 model_name_cuda="THUDM/chatglm2-6b", 
                 model_name_cpu="THUDM/chatglm2-6b-int4", 
                 revision="main"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name_cuda if self.device == "cuda" else model_name_cpu
        logger.info(f'Using device: {self.device}')
        logger.info(f'Loading model: {self.model_name}')

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, revision=revision
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, revision=revision
        )

        if self.device == "cuda":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.float().cpu()

        self.model.eval()

        # 初始化历史对话记录
        self.history: List[Tuple[str, str]] = []

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        if not query:
            raise ValueError("输入 query 不能为空")

        reset_history = config.get("reset_history", False) if config else False
        if reset_history or not hasattr(self, "_history"):
            self._history = []

        try:
            result = self.model.chat(self.tokenizer, query, history=self._history)

            # 打印调试信息，确认返回值
            logger.info(f"model.chat 返回值类型: {type(result)}")
            logger.info(f"model.chat 返回值内容: {result}")

            if isinstance(result, tuple) and len(result) == 2:
                response, self._history = result
            else:
                response = result
                self._history.append((query, response))

            return response

        except Exception as e:
            logger.error(f"invoke 模型调用失败: {e}")
            raise RuntimeError(f"处理问题失败: {str(e)}")

