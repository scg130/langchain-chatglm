from fastapi import APIRouter, HTTPException
from core.qa_service import QAService
from config.logger_config import logger
from schemas.qa_dto import AskRequest, AskResponse
import asyncio

qa_service = QAService()

router = APIRouter(prefix="/api/v1", tags=["QA"])

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(400, "问题不能为空")
    try:
        result = await qa_service.ask(request.question, request.history or [])
        return AskResponse(
            answer=result["answer"],
            index_type=result.get("index_type", "full_text"),
        )
    except Exception as e:
        logger.error(f"提问处理失败: {e}")
        raise HTTPException(500, f"处理问题失败: {str(e)}")
