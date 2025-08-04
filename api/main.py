from fastapi import FastAPI
from api.routers import qa_router, health_router
from api.routers.qa_router import qa_service
from contextlib import asynccontextmanager
from config.logger_config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("启动初始化QA服务...")
    await qa_service.initialize()
    yield
    logger.info("应用关闭")

app = FastAPI(title="智能文档问答API", version="1.0.0", lifespan=lifespan)

app.include_router(qa_router)
app.include_router(health_router)
