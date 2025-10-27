from pydantic import BaseModel
from typing import List, Optional, Tuple

class AskRequest(BaseModel):
    question: str
    history: Optional[List[Tuple[str, str]]] = None  # 可选历史对话
    is_web_search: Optional[bool] = False
    dir_path: Optional[str] = ""  # 修复：改为可选参数

class AskResponse(BaseModel):
    answer: str

