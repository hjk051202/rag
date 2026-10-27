"""建索引：扫描 data/ 目录，切块、向量化、写入 Chroma。

用法（在项目根目录）：
    python scripts/build_index.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许不安装包也能直接运行脚本
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zhisaotong_agent.rag.vector_store import VectorStoreService  # noqa: E402
from zhisaotong_agent.utils.logger_handler import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    service = VectorStoreService()

    before = service.vector_store._collection.count()
    print(f"建库前已有分块：{before}")

    service.load_document()

    after = service.vector_store._collection.count()
    print(f"建库后分块总数：{after}（本次新增 {after - before}）")

    if after == 0:
        print("没有写入任何内容，请检查 data/ 目录里是否有 .txt 或 .pdf 文件。")
        return 1

    print("完成。下一步：python scripts/compare_retrieval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
