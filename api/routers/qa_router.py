from fastapi import APIRouter, HTTPException
from core.qa_service import qa_service
from config.logger_config import logger
from ..schemas.qa_dto import AskRequest, AskResponse
from util.to_str import to_str_safe

router = APIRouter(prefix="/api/v1", tags=["QA"])

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="问题不能为空"
            )
            
        result = await qa_service.ask_question(to_str_safe(request.question))
        return AskResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"提问处理失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"处理问题时出错: {str(e)}"
        )