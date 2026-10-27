"""模型工厂：统一走 OpenAI 兼容协议。

相比原项目的改动
----------------
原项目把模型写死成 DashScope（通义千问 + text-embedding-v4），换模型要改代码。
这里抽象成「OpenAI 兼容协议」：只要服务商提供 /v1/chat/completions 和
/v1/embeddings，换一家只改 .env，不动代码。

默认配置（复用常见的便宜组合）：
    对话：DeepSeek       CHAT_BASE_URL / CHAT_MODEL / DEEPSEEK_API_KEY
    向量：硅基流动 bge-m3 EMBEDDING_BASE_URL / EMBEDDING_MODEL / OPENAI_API_KEY
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from openai import OpenAI

from zhisaotong_agent.utils.config_handler import RagConfig, load_rag_config
from zhisaotong_agent.utils.logger_handler import get_logger

load_dotenv()

logger = get_logger(__name__)

_rag_conf: RagConfig = load_rag_config()


def _env(*names: str, default: str | None = None) -> str | None:
    """按顺序取第一个非空环境变量。"""
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


# 对话模型
CHAT_BASE_URL = _env("CHAT_BASE_URL", default="https://api.deepseek.com/v1")
CHAT_MODEL = _env("CHAT_MODEL", default=_rag_conf.model.chat_model_name)
CHAT_API_KEY = _env("CHAT_API_KEY", "DEEPSEEK_API_KEY")

# 向量模型
EMBEDDING_BASE_URL = _env("EMBEDDING_BASE_URL", default="https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", default="BAAI/bge-m3")
EMBEDDING_API_KEY = _env("EMBEDDING_API_KEY", "OPENAI_API_KEY")

# 一次请求最多送多少条文本去向量化（服务商普遍有单次上限）
_EMBED_BATCH_SIZE = 32


def get_chat_client() -> OpenAI:
    """对话客户端。"""
    if not CHAT_API_KEY:
        raise RuntimeError(
            "未找到对话模型的 API Key。请在 .env 中配置 DEEPSEEK_API_KEY 或 CHAT_API_KEY。"
        )
    return OpenAI(base_url=CHAT_BASE_URL, api_key=CHAT_API_KEY)


def get_embedding_client() -> OpenAI:
    """向量化客户端。"""
    if not EMBEDDING_API_KEY:
        raise RuntimeError(
            "未找到向量模型的 API Key。请在 .env 中配置 OPENAI_API_KEY 或 EMBEDDING_API_KEY。"
        )
    return OpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)


class OpenAICompatEmbeddings(Embeddings):
    """把 OpenAI 兼容的 /v1/embeddings 接口包成 LangChain 的 Embeddings 接口。

    Chroma 只需要一个实现了 embed_documents / embed_query 的对象，
    自己写这个适配器就不必再引入 langchain-openai 这个包。
    """

    def __init__(self) -> None:
        self._client = get_embedding_client()

    def _embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            response = self._client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
            )
            # 按 index 排序，避免服务端乱序返回
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([item.embedding for item in ordered])
        return vectors

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


_embed_model: OpenAICompatEmbeddings | None = None


def get_embed_model() -> OpenAICompatEmbeddings:
    """懒加载全局 Embedding 实例（建库和检索必须复用同一个）。"""
    global _embed_model
    if _embed_model is None:
        _embed_model = OpenAICompatEmbeddings()
        logger.info("Embedding 模型已初始化：%s", EMBEDDING_MODEL)
    return _embed_model


# 兼容原项目里可能存在的调用写法
get_embedding_model = get_embed_model


def get_chat_model_name() -> str:
    """当前对话模型名（供日志和界面展示）。"""
    return CHAT_MODEL or ""
