from fastapi import APIRouter, HTTPException
from core.qa_service import qa_service
from config.logger_config import logger
from schemas.qa_dto import AskRequest, AskResponse

router = APIRouter(prefix="/api/v1", tags=["QA"])

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(400, "问题不能为空")
    try:
        result = await qa_service.ask_question(request.question)
        return AskResponse(answer=result["answer"], index_type=result.get("index_type", "full_text"), sources=[])
    except Exception as e:
        logger.error(f"提问处理失败: {e}")
        raise HTTPException(500, f"处理问题失败: {str(e)}")
