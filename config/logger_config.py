# logger_config.py
import logging
import sys

def setup_logger():
    """配置并返回全局logger"""
    logger = logging.getLogger("app")  # 使用固定名称而非__name__
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 文件处理器（可选）
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)
    
    # 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

# 初始化全局logger
logger = setup_logger()