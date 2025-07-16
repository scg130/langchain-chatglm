from config.logger_config import logger
from core.qa_service import qa_service
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import qa_router, health_router  # 从包导入

# 定义生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动逻辑（相当于 @app.on_event("startup")）
    print("Starting up...")
    await qa_service.initialize()
    yield  # 在此处保持应用运行
    # 关闭逻辑（相当于 @app.on_event("shutdown")）
    print("Shutting down...")

app = FastAPI(
    lifespan=lifespan,  # 使用生命周期管理器
    title="智能文档问答API",
    version="1.0.0"
)

# 正确的路由注册方式
app.include_router(qa_router)
app.include_router(health_router)


