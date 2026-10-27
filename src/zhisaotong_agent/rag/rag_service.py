"""RAG 服务：检索 → 上下文组装 → 生成，带引用标注与无依据拒答。

相比原项目改了三点
------------------
1. 检索从"纯向量"换成"向量 + BM25 混合"（见 hybrid_retriever.py）。
2. 上下文组装不再是简单截断，而是：按分数排序 → 去重 → 带编号 →
   按字符预算裁切，并把编号透出给界面做引用展示。
3. 真正用上了 rag.yml 里那个一直没生效的 score_threshold：
   检索相关性低于阈值时直接说"知识库里没有"，而不是硬编一个答案。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from zhisaotong_agent.model.factory import CHAT_MODEL, get_chat_client
from zhisaotong_agent.rag.hybrid_retriever import HybridRetriever
from zhisaotong_agent.rag.vector_store import VectorStoreService
from zhisaotong_agent.utils.config_handler import RagConfig, load_rag_config
from zhisaotong_agent.utils.logger_handler import get_logger
from zhisaotong_agent.utils.prompt_loader import load_rag_prompts

logger = get_logger(__name__)

# 送进模型的参考资料最多多少字符
MAX_CONTEXT_CHARS = 3000
# 单条资料截断到这个长度，避免一本书里的一段霸占全部预算
MAX_SNIPPET_CHARS = 800

ABSTAIN_ANSWER = "知识库中没有找到与这个问题相关的内容。你可以换个说法，或者补充一下具体型号/场景。"


class RagSummarizeService:
    """对外保持 `rag_summarize(query) -> str` 这个原有接口不变。"""

    def __init__(self) -> None:
        self._conf: RagConfig = load_rag_config()
        self._vector_service = VectorStoreService()
        self._retriever = HybridRetriever(self._vector_service.vector_store)
        self._prompt_template = load_rag_prompts()
        self._client = get_chat_client()
        logger.info(
            "RAG 服务就绪：知识库 %d 个分块，对话模型 %s",
            self._retriever.size,
            CHAT_MODEL,
        )

    # === 检索 ===

    def retrieve(self, query: str, k: int | None = None) -> Tuple[List[Tuple[Document, float]], float]:
        top_k = k or self._conf.retrieval.top_k
        return self._retriever.search(query, k=top_k)

    # === 上下文组装 ===

    def build_context(self, hits: List[Tuple[Document, float]]) -> Tuple[str, List[Dict[str, Any]]]:
        """把命中的分块拼成带编号的参考资料，同时返回结构化来源列表。"""
        blocks: List[str] = []
        sources: List[Dict[str, Any]] = []
        used = 0

        for index, (doc, score) in enumerate(hits, start=1):
            text = " ".join(doc.page_content.split())
            if not text:
                continue
            if len(text) > MAX_SNIPPET_CHARS:
                text = text[:MAX_SNIPPET_CHARS] + "…"

            source = str(doc.metadata.get("source", "未知来源"))
            block = f"【资料{index}｜{source}】\n{text}"

            remaining = MAX_CONTEXT_CHARS - used
            if remaining <= 80:
                break
            if len(block) > remaining:
                block = block[:remaining] + "…"

            blocks.append(block)
            used += len(block)
            sources.append(
                {
                    "index": index,
                    "source": source,
                    "score": round(float(score), 4),
                    "preview": text[:120],
                }
            )

        return "\n\n".join(blocks), sources

    # === 生成 ===

    def rag_answer(self, query: str) -> Dict[str, Any]:
        """完整问答：返回答案、来源、是否触发拒答。"""
        try:
            hits, best_score = self.retrieve(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("检索失败，query=%s, error=%s", query, exc, exc_info=True)
            return {"answer": "检索服务暂时不可用，请稍后重试。", "sources": [], "abstained": True}

        if not hits:
            return {"answer": ABSTAIN_ANSWER, "sources": [], "abstained": True}

        # 相关性太低就不硬答（原项目配了这个阈值但代码里没用）
        threshold = float(getattr(self._conf.retrieval, "score_threshold", 0.0) or 0.0)
        if best_score and threshold and best_score < threshold:
            logger.info(
                "触发拒答：最高相关性 %.3f 低于阈值 %.3f，query=%s", best_score, threshold, query
            )
            return {"answer": ABSTAIN_ANSWER, "sources": [], "abstained": True}

        context, sources = self.build_context(hits)
        if not context:
            return {"answer": ABSTAIN_ANSWER, "sources": [], "abstained": True}

        prompt = self._prompt_template.format(input=query, context=context)
        try:
            response = self._client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("模型调用失败，query=%s, error=%s", query, exc, exc_info=True)
            return {"answer": "生成回答时出错，请稍后重试。", "sources": sources, "abstained": True}

        if not answer:
            answer = ABSTAIN_ANSWER
            return {"answer": answer, "sources": [], "abstained": True}

        return {"answer": answer, "sources": sources, "abstained": False}

    def rag_summarize(self, query: str) -> str:
        """兼容原接口：只要文本答案。"""
        return self.rag_answer(query)["answer"]


if __name__ == "__main__":
    # 简单自测：python -m zhisaotong_agent.rag.rag_service
    service = RagSummarizeService()
    question = "扫地机器人多久清理一次尘盒？"
    result = service.rag_answer(question)
    print("Q:", question)
    print("A:", result["answer"])
    print("来源:", [(s["index"], s["source"], s["score"]) for s in result["sources"]])
