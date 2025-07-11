from fastapi import APIRouter, HTTPException
from core.qa_service import qa_service
from config.logger_config import logger
from ..schemas.qa_dto import AskRequest, AskResponse

router = APIRouter(prefix="/api/v1", tags=["QA"])

@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        # 调用服务获取结果
        result = await qa_service.ask_question(request.question)
        
        # 处理结果
        if isinstance(result, dict):
            answer = result.get("result") or result.get("answer") or str(result)
        else:
            answer = str(result)
        
        # 构造响应对象
        return AskResponse(
            answer=answer
        )
    except Exception as e:
        logger.error(f"提问处理失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"处理问题时出错: {str(e)}"
        )