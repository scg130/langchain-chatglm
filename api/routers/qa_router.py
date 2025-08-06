from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse
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

@router.post("/ask/stream")
async def ask_stream(request: Request, body: AskRequest):
    if not body.question.strip():
        return EventSourceResponse(content="data: 问题不能为空\n\n", status_code=400)

    async def event_generator():
        try:
            # 获取答案生成器（需要你的模型支持 yield 逐步返回）
            async for chunk in qa_service.ask_stream(body.question, body.history):
                if await request.is_disconnected():
                    break
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: 发生错误：{str(e)}\n\n"
    return EventSourceResponse(event_generator())