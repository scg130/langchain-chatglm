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

    # 统计注册的向量库数量
    registered = 0
    dir_candidates = []

    # 将 data_dir 本身和其下包含文件的子目录都注册为候选向量库
    for current_dir, subdirs, files in os.walk(data_dir):
        has_file = any(os.path.isfile(os.path.join(current_dir, f)) for f in files)
        if has_file and current_dir not in dir_candidates:
            dir_candidates.append(current_dir)

    for d in dir_candidates:
        try:
            print(f"注册向量库: {d}")
            stats = vector_manager.add_directory(
                dir_path=d,
                file_pattern="**/*",
                batch_size=500,
                force_reload=False,
                show_progress=True,
            )
            print(f"处理统计({d}): {stats}")
            registered += 1
        except Exception as e:
            print(f"注册向量库失败 {d}: {e}")

    print(f"已注册向量库个数: {registered}")

if __name__ == "__main__":
    asyncio.run(main())
