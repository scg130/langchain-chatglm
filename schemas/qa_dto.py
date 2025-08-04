from pydantic import BaseModel
from typing import List, Optional, Tuple

class AskRequest(BaseModel):
    question: str
    history: Optional[List[Tuple[str, str]]] = None  # 可选历史对话

class AskResponse(BaseModel):
    answer: str
    index_type: str = "full_text"
    sources: List = []

