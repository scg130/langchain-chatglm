from pydantic import BaseModel
from typing import List, Optional, Tuple

class AskRequest(BaseModel):
    question: str
    history: Optional[List[Tuple[str, str]]] = None  # 可选历史对话
    is_web_search: Optional[bool] = False
    dir_path: str

class AskResponse(BaseModel):
    answer: str

