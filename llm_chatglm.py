from langchain_core.language_models.llms import LLM
from typing import Optional, List
from transformers import AutoTokenizer, AutoModel


class ChatGLMLLM(LLM):
    model_name: str = "THUDM/chatglm2-6b"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).half().cuda()  # GPU 用 cuda
        self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).float()  # CPU 用 float32
        self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, _ = self._model.chat(self._tokenizer, prompt, history=[])
        return response
