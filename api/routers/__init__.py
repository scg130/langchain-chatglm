from .qa_router import router as qa_router
from .health_router import router as health_router

# 可选：统一导出
__all__ = ["qa_router", "health_router"]