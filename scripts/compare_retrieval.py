"""对比「纯向量检索」与「向量 + BM25 混合检索」的召回差异。

用法（先建好索引）：
    python scripts/compare_retrieval.py
    python scripts/compare_retrieval.py "X10 的尘盒容量是多少"

输出：
    - 两种方式各自命中的分块
    - 重叠度 Jaccard（1.0 = 完全一样，说明混合没有带来差异）
    - 耗时

写简历/答辩时的用法：挑几个"带型号、带数字"的问题，
展示纯向量漏掉、混合检索补回来的那一条。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zhisaotong_agent.rag.hybrid_retriever import HybridRetriever  # noqa: E402
from zhisaotong_agent.rag.vector_store import VectorStoreService  # noqa: E402
from zhisaotong_agent.utils.config_handler import load_rag_config  # noqa: E402

DEFAULT_QUERIES = [
    "尘盒多久清理一次",
    "扫地机器人充不进电怎么办",
    "边刷不转了是什么原因",
]


def key_of(doc) -> str:
    return f"{doc.metadata.get('source', '')}::{doc.page_content[:60]}"


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> int:
    config = load_rag_config()
    top_k = config.retrieval.top_k

    store = VectorStoreService().vector_store
    retriever = HybridRetriever(store)

    if retriever.size == 0:
        print("向量库是空的，请先运行：python scripts/build_index.py")
        return 1

    print(f"知识库分块数：{retriever.size}   top_k={top_k}\n")

    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEFAULT_QUERIES

    for query in queries:
        print("=" * 70)
        print(f"问题：{query}")
        print("=" * 70)

        started = time.time()
        vector_docs = [doc for doc, _ in retriever.search_vector(query, top_k)]
        vector_ms = (time.time() - started) * 1000

        started = time.time()
        hybrid_hits, best_score = retriever.search(query, top_k)
        hybrid_ms = (time.time() - started) * 1000

        print(f"\n[纯向量] {vector_ms:.0f} ms")
        for index, doc in enumerate(vector_docs, 1):
            print(f"  {index}. {doc.metadata.get('source', '?')}  |  {doc.page_content[:50]}")

        print(f"\n[混合检索] {hybrid_ms:.0f} ms  最高相关性={best_score:.3f}")
        for index, (doc, score) in enumerate(hybrid_hits, 1):
            print(
                f"  {index}. {doc.metadata.get('source', '?')}  |  "
                f"RRF={score:.4f}  |  {doc.page_content[:50]}"
            )

        overlap = jaccard({key_of(d) for d in vector_docs}, {key_of(d) for d, _ in hybrid_hits})
        print(f"\n重叠度 Jaccard：{overlap:.2f}")
        if overlap >= 0.99:
            print("（本次两种方式结果一致 —— 换个带型号或数字的问题更容易看出差异）")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
