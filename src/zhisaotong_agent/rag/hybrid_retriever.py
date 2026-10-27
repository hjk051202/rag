"""混合检索：向量召回 + BM25 关键词召回，用 RRF 融合。

为什么加这个
------------
原项目只有纯向量检索（`as_retriever(k=3)`）。向量擅长"意思相近"，
但对型号、数字、专有名词不敏感：问「X10 Pro 的尘盒容量」，向量可能召回
一堆泛泛的"选购建议"，而真正写着参数的那一段因为措辞不同被漏掉。

BM25 恰好相反：它按词频精确匹配，型号、数字这类"必须一字不差"的东西
召回很准，但完全不懂同义改写。

两路互补。融合用 RRF（Reciprocal Rank Fusion）：
    score(d) = Σ 1 / (k + rank_i(d))，k 取 60
只用"排名"不用"分数"，因为余弦相似度（0~1）和 BM25 分数（0~几十）
根本不在一个量纲上，没法直接相加。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from zhisaotong_agent.utils.logger_handler import get_logger

logger = get_logger(__name__)

# RRF 的经验常数，出自原论文（Cormack et al., 2009）
RRF_K = 60


def _doc_key(doc: Document) -> str:
    """给文档生成稳定的去重键。"""
    source = str(doc.metadata.get("source", ""))
    return f"{source}::{doc.page_content}"


class HybridRetriever:
    """在 Chroma 之上叠加一路 BM25 检索。"""

    def __init__(self, vector_store, *, rrf_k: int = RRF_K) -> None:
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self._docs: List[Document] = []
        self._bm25: BM25Okapi | None = None
        self._build_bm25()

    # === 构建 ===

    def _build_bm25(self) -> None:
        """从向量库里把全部文本捞出来，在内存里建 BM25 索引。

        知识库在几千个分块以内时这一步只要一两秒；
        再大就该换成数据库侧的全文索引了。
        """
        try:
            raw = self.vector_store.get(include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取向量库内容失败，BM25 将不可用：%s", exc)
            return

        texts: List[str] = raw.get("documents") or []
        metadatas: List[dict] = raw.get("metadatas") or [{} for _ in texts]
        self._docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(texts, metadatas)
        ]

        if not self._docs:
            logger.warning("向量库为空，BM25 索引跳过（请先运行 build_index）")
            return

        # jieba 分词：中文没有空格，必须先切词再交给 BM25
        tokenized = [list(jieba.cut(doc.page_content)) for doc in self._docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 索引构建完成，共 %d 个分块", len(self._docs))

    @property
    def size(self) -> int:
        return len(self._docs)

    # === 单路检索 ===

    def search_vector(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """向量检索，带 0~1 的相关性分数（用于拒答判断）。"""
        try:
            return self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("相关性打分不可用，退回普通相似度检索：%s", exc)
            return [(doc, 0.0) for doc in self.vector_store.similarity_search(query, k=k)]

    def search_bm25(self, query: str, k: int) -> List[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(list(jieba.cut(query)))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in ranked if scores[i] > 0]

    # === 融合 ===

    def search(self, query: str, k: int = 5) -> Tuple[List[Tuple[Document, float]], float]:
        """混合检索。

        返回 (融合后的 [(文档, RRF 分数)], 向量最高相关性分)。
        第二个值用来判断"知识库里到底有没有相关内容"。
        """
        # 两路都多召回一些，融合后才有挑选余地
        candidates = max(k * 3, k + 5)

        vector_pairs = self.search_vector(query, candidates)
        bm25_docs = self.search_bm25(query, candidates)

        best_vector_score = vector_pairs[0][1] if vector_pairs else 0.0

        ranked_lists: List[List[Document]] = [
            [doc for doc, _ in vector_pairs],
            bm25_docs,
        ]

        fused: Dict[str, float] = {}
        doc_by_key: Dict[str, Document] = {}

        for docs in ranked_lists:
            for rank, doc in enumerate(docs):
                key = _doc_key(doc)
                doc_by_key.setdefault(key, doc)
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(doc_by_key[key], score) for key, score in ordered], best_vector_score
