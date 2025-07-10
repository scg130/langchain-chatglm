from typing import Any, Optional
from transformers import AutoTokenizer, AutoModel
from langchain_core.runnables import Runnable
import torch

class ChatGLMLLM(Runnable):
    def __init__(self, model_name="THUDM/chatglm2-6b", use_gpu=False, revision="main"):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,revision=revision
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, revision=revision
        )
        self.model = self.model.half().cuda() if self.device == "cuda" else self.model.float().cpu()
        self.model.eval()

    from typing import Any, Optional

    def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> str:
        if not query:
            raise ValueError("输入 query 不能为空")

        # 接受但忽略额外参数
        # stop = kwargs.get("stop")

        response, _ = self.model.chat(self.tokenizer, query, history=[])
        return response


