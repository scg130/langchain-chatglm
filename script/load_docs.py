import asyncio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.vectorstore_manager import VectorStoreManager

async def main():
    # 文档目录
    data_dir = "./data"
    # 初始化向量库管理器
    vector_manager = VectorStoreManager()

    # 支持三种索引类型示例增量加载
    for index_type in ["full_text", "section", "detail"]:
        print(f"\n➡️ 处理索引类型: {index_type}")

        # 增量加载文档到对应索引类型向量库
        stats = vector_manager.add_directory(
            dir_path=data_dir,
            file_pattern="**/*",
            batch_size=500,
            force_reload=False,  # False 表示增量加载
            show_progress=True,
            index_type=index_type
        )
        print(f"索引类型 {index_type} 处理统计: {stats}")

if __name__ == "__main__":
    asyncio.run(main())
