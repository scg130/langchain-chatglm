from api.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",  # 改为字符串形式
        host="0.0.0.0",
        port=8800,
        reload=True,  # 现在可以正常启用热重载
        log_level="info"
    )