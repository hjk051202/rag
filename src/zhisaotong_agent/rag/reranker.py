"""重排（Rerank）：用 cross-encoder 对候选做精排。

为什么需要它
------------
RRF 只是把两路检索的"名次"合起来，它从头到尾没见过问题和文档的内容，
所以它无法判断"这一条到底能不能回答这个问题"。

Cross-encoder 不一样：它把「问题 + 候选块」拼成一条输入，让模型直接输出
一个相关性分数。因为它能看到问题和文档之间的词是怎么对应的，精度显著高于
"各自变向量再比距离"的 bi-encoder。

代价：每个候选都要单独跑一次模型，所以只能用于小批量（20-30 个），
不能用来做全库检索。这就是"召回粗、重排精"的分工。

实现选择
--------
这里调用的是硅基流动的 /v1/rerank 接口（模型 BAAI/bge-reranker-v2-m3），
而不是在本地装 torch + sentence-transformers：
  - 本地方案要拉 2GB 依赖，部署成本高
  - API 方案复用现有的 OPENAI_API_KEY，零新增依赖
如果接口不可用，会自动降级为原来的 RRF 顺序，不影响问答。
"""

from __future__ import annotations

import os
from typing import List, Tuple

import httpx
from dotenv import load_dotenv
from langchain_core.documents import Document

from zhisaotong_agent.utils.logger_handler import get_logger

# 本模块在 import 时就要读环境变量，所以必须先加载 .env，
# 不能依赖别的模块恰好先 import 了 factory。
load_dotenv()

logger = get_logger(__name__)

# 是否启用重排（默认关，避免没配好时影响主流程）
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "0").strip() == "1"

# 重排模型与接口
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3").strip()
RERANK_BASE_URL = os.getenv(
    "RERANK_BASE_URL", os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
).strip()
RERANK_API_KEY = (
    os.getenv("RERANK_API_KEY")
    or os.getenv("EMBEDDING_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

# 送进重排的候选数量：召回要多、重排要精
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "20"))


def is_enabled() -> bool:
    """重排是否启用。"""
    return RERANK_ENABLED


class RerankService:
    """调用 /v1/rerank 做精排。"""

    def __init__(self) -> None:
        if not RERANK_API_KEY:
            raise RuntimeError(
                "未找到重排接口的 API Key。请在 .env 中配置 OPENAI_API_KEY 或 RERANK_API_KEY。"
            )
        self._url = RERANK_BASE_URL.rstrip("/") + "/rerank"
        self._headers = {
            "Authorization": f"Bearer {RERANK_API_KEY}",
            "Content-Type": "application/json",
        }
        logger.info("重排服务已初始化：%s -> %s", RERANK_MODEL, self._url)

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_n: int = 5,
    ) -> List[Tuple[Document, float]]:
        """对候选重新打分排序。

        返回 [(文档, 重排分)]，按分数降序。
        任何异常都会降级为"保持原顺序"，不会让上层崩溃。
        """
        if not docs:
            return []

        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": [doc.page_content for doc in docs],
            "top_n": min(top_n, len(docs)),
        }

        try:
            with httpx.Client(timeout=RERANK_TIMEOUT) as client:
                response = client.post(self._url, json=payload, headers=self._headers)
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("重排接口不可用，降级为 RRF 原顺序：%s", exc)
            return [(doc, 0.0) for doc in docs[:top_n]]

        results = body.get("results") or []
        if not results:
            logger.warning("重排接口返回空结果，降级为 RRF 原顺序")
            return [(doc, 0.0) for doc in docs[:top_n]]

        ranked: List[Tuple[Document, float]] = []
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score", 0.0)
            if not isinstance(index, int) or not (0 <= index < len(docs)):
                continue
            ranked.append((docs[index], float(score)))

        if not ranked:
            return [(doc, 0.0) for doc in docs[:top_n]]

        ranked.sort(key=lambda pair: pair[1], reverse=True)
        logger.info(
            "重排完成：%d 个候选 -> 前 %d 个，最高分 %.3f",
            len(docs),
            len(ranked),
            ranked[0][1],
        )
        return ranked[:top_n]
