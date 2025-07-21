from typing import Any, Optional
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
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, revision=revision
        )
        # 加载模型
        self.model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, revision=revision
        )
        # 设置模型精度与设备
        if self.device == "cuda":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.float().cpu()
        self.model.eval()

    from typing import Any, Optional

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        if not query:
            raise ValueError("输入 query 不能为空")

        # 接受但忽略额外参数
        # stop = kwargs.get("stop")

        response, _ = self.model.chat(self.tokenizer, query, history=[])
        return response


