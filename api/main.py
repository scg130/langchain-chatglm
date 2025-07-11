from config.logger_config import logger
from core.qa_service import qa_service
from config.logger_config import logger

from fastapi import FastAPI
from api.routers import qa_router, health_router  # 从包导入

app = FastAPI(
    title="智能文档问答API",
    version="1.0.0"
)

# 正确的路由注册方式
app.include_router(qa_router)
app.include_router(health_router)

@app.on_event("startup")
async def startup():
    logger.info("初始化服务...")
    await qa_service.initialize()