# logger_config.py
import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

def setup_logger(log_dir='logs'):
    """配置并返回全局logger（每小时一个文件，保留24小时）"""
    logger = logging.getLogger("app")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # 每小时切分日志，保留最近 24 小时日志文件
    file_handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "app.log"),
        when="H",               # 每小时
        interval=1,             # 每1小时切一次
        backupCount=24,         # 保留24个文件（即24小时）
        encoding="utf-8",
        utc=False               # 使用本地时间
    )
    file_handler.suffix = "%Y-%m-%d_%H"  # 日志文件后缀：按小时命名
    file_handler.setFormatter(formatter)

    # 避免重复添加处理器
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# 初始化全局logger
logger = setup_logger()
