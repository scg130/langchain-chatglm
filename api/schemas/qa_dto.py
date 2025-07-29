from pydantic import BaseModel
from typing import List, Dict, Any, Literal

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    status: str = "success"
    sources: List[Dict[str, Any]] = []
    index_type: Literal["full_text", "section", "detail"] = "full_text"