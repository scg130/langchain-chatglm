import asyncio

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from config.logger_config import logger
from core.qa_service import QAService
from schemas.qa_dto import AskRequest, AskResponse

qa_service = QAService()
qa_service.search_engine = "google"

router = APIRouter(prefix="/api/v1", tags=["QA"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(400, "问题不能为空")
    try:
        result = await qa_service.ask(
            question=request.question,
            history=request.history or [],
            is_web_search=bool(request.is_web_search),
            dir_path=request.dir_path,
        )
        return AskResponse(
            answer=result["answer"],
        )
    except ValueError as e:
        logger.error(f"请求参数错误: {e}")
        raise HTTPException(400, f"请求参数错误: {str(e)}")
    except RuntimeError as e:
        logger.error(f"服务运行错误: {e}")
        raise HTTPException(500, f"服务运行错误: {str(e)}")
    except Exception as e:
        logger.error(f"提问处理失败: {e}", exc_info=True)
        raise HTTPException(500, f"处理问题失败: {str(e)}")


@router.post("/ask/stream")
async def ask_stream(request: Request, body: AskRequest):
    if not body.question.strip():
        return EventSourceResponse(content="data: 问题不能为空\n\n", status_code=400)

    async def event_generator():
        try:
            async for chunk in qa_service.ask_stream(
                question=body.question,
                history=body.history or [],
                is_web_search=bool(body.is_web_search),
                dir_path=body.dir_path,
            ):
                if await request.is_disconnected():
                    break
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式处理错误: {e}", exc_info=True)
            yield f"data: [ERROR] 发生错误：{str(e)}\n\n"
    return EventSourceResponse(event_generator())
