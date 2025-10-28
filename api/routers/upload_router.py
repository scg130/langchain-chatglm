import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from config.logger_config import logger

router = APIRouter(prefix="/api/v1", tags=["Upload"])


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上传文件到知识库
    
    Args:
        files: 要上传的文件列表
        
    Returns:
        上传结果统计
    """
    upload_dir = "./data/upload"
    os.makedirs(upload_dir, exist_ok=True)
    
    uploaded = []
    failed = []
    
    for file in files:
        try:
            # 保存文件
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            uploaded.append({
                "filename": file.filename,
                "size": os.path.getsize(file_path)
            })
            logger.info(f"文件上传成功: {file.filename}")
            
        except Exception as e:
            logger.error(f"文件上传失败 {file.filename}: {e}")
            failed.append({"filename": file.filename, "error": str(e)})
    
    return {
        "status": "success",
        "uploaded": uploaded,
        "failed": failed,
        "total": len(files)
    }


@router.get("/knowledge-bases")
async def list_knowledge_bases():
    """获取所有知识库列表"""
    data_dir = "./data"
    knowledge_bases = []
    
    if os.path.exists(data_dir):
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                # 统计文件数量
                file_count = 0
                for root, dirs, files in os.walk(item_path):
                    file_count += len(files)
                
                knowledge_bases.append({
                    "name": item,
                    "path": item_path,
                    "file_count": file_count
                })
    
    return {"knowledge_bases": knowledge_bases}

