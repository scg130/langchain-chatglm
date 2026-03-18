import os
from typing import List, Set

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.logger_config import logger
from api.routers.qa_router import qa_service
from util.func import vector_manager

router = APIRouter(prefix="/api/v1", tags=["Upload"])

ALLOWED_EXTENSIONS: Set[str] = {".txt", ".pdf", ".docx", ".md"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50MB


def _safe_upload_name(name: str) -> str:
    base = os.path.basename(name or "")
    if not base or base != (name or "").strip():
        return ""
    ext = os.path.splitext(base)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return ""
    return base


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上传文件到知识库

    仅允许 .txt / .pdf / .docx / .md，文件名取 basename，防止路径穿越。
    """
    upload_dir = os.path.abspath("./data/upload")
    os.makedirs(upload_dir, exist_ok=True)

    uploaded = []
    failed = []
    indexed_names: List[str] = []
    index_error: str | None = None

    for file in files:
        safe_name = _safe_upload_name(file.filename or "")
        if not safe_name:
            failed.append(
                {
                    "filename": file.filename,
                    "error": "仅支持 .txt / .pdf / .docx / .md，且不能使用路径分隔符",
                }
            )
            continue
        file_path = os.path.join(upload_dir, safe_name)
        try:
            size = 0
            with open(file_path, "wb") as buffer:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_FILE_BYTES:
                        raise ValueError(f"单文件超过 {MAX_FILE_BYTES // (1024 * 1024)}MB 限制")
                    buffer.write(chunk)
            uploaded.append({"filename": safe_name, "size": os.path.getsize(file_path)})
            indexed_names.append(safe_name)
            logger.info(f"文件上传成功: {safe_name}")
        except Exception as e:
            logger.error(f"文件上传失败 {safe_name}: {e}")
            failed.append({"filename": file.filename, "error": str(e)})
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    if indexed_names:
        try:
            for name in indexed_names:
                vector_manager.remove_file_vectors(upload_dir, name)
            logger.info("开始增量更新向量数据库...")
            await qa_service._register_directory(upload_dir)
            stats = vector_manager.add_directory(
                upload_dir,
                batch_size=1000,
                force_reload=False,
                show_progress=False,
            )
            logger.info(f"向量数据库更新成功: {stats}")
        except Exception as e:
            index_error = str(e)
            logger.error(f"向量数据库更新失败: {e}", exc_info=True)

    overall = "success"
    if failed and uploaded:
        overall = "partial"
    elif failed and not uploaded:
        overall = "failed"
    if index_error:
        overall = "partial" if uploaded else "failed"

    return {
        "status": overall,
        "uploaded": uploaded,
        "failed": failed,
        "total": len(files),
        "index_ok": index_error is None,
        "index_error": index_error,
    }


@router.get("/knowledge-bases")
async def list_knowledge_bases():
    """获取所有知识库列表"""
    data_dir = "./data"
    knowledge_bases = []

    if os.path.exists(data_dir):
        knowledge_bases.append(
            {
                "name": "",
                "path": data_dir,
                "file_count": 0,
                "description": "默认知识库（包含所有子目录）",
            }
        )

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                file_count = 0
                for root, dirs, files in os.walk(item_path):
                    file_count += len(files)

                knowledge_bases.append(
                    {
                        "name": item,
                        "path": item_path,
                        "file_count": file_count,
                        "description": f"知识库: {item}",
                    }
                )

    return {"knowledge_bases": knowledge_bases}
