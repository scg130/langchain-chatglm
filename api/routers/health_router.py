from fastapi import APIRouter

# 创建路由实例 - 变量名必须为 router
router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
    return {"status": "healthy"}