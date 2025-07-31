from pydantic import BaseModel
from typing import List, Optional

class AskRequest(BaseModel):
    question: str

class SourceDoc(BaseModel):
    page_content: str
    metadata: dict

class AskResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDoc]] = []
    index_type: Optional[str] = "full_text"
