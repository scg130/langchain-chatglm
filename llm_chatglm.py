from transformers import AutoTokenizer, AutoModel
import torch

class ChatGLMLLM:
    def __init__(self, model_name: str = "THUDM/chatglm2-6b", use_gpu: bool = False, **kwargs):
        self.model_name = model_name
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        print(f"加载模型到设备: {self.device}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # 根据设备类型加载模型
        self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        if self.device == "cuda":
            self._model = self._model.half().cuda()
        else:
            self._model = self._model.float().cpu()

        self._model.eval()

    def chat(self, prompt: str) -> str:
        """与模型进行简单对话"""
        response, _ = self._model.chat(self._tokenizer, prompt, history=[])
        return response
