from fastapi import APIRouter, HTTPException
from core.qa_service import qa_service
from config.logger_config import logger
from ..schemas.qa_dto import AskRequest, AskResponse

router = APIRouter(prefix="/api/v1", tags=["QA"])

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
            
        result = await qa_service.ask_question(request.question)
        
        # Improved response extraction
        answer = str(result.get("result", result.get("answer", result)))
        
        return AskResponse(answer=answer)
        
    except HTTPException:
        raise  # Re-raise existing HTTP exceptions
    except Exception as e:
        logger.exception(f"提问处理失败: {e}")  # Logs full traceback
        raise HTTPException(
            status_code=500,
            detail=f"处理问题时出错: {str(e)}"
        )