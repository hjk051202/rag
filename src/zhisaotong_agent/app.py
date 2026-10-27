"""智扫通 · 扫地机器人知识库问答（Streamlit 前端）。

与原项目的区别：原 app.py 走 ReactAgent（Agent 决策 → 调工具 → 调 RAG），
这里直接调用 RAG 服务。少一层 Agent，链路短、可解释、延迟低，
也更容易在面试里讲清楚每一步在做什么。
"""

from __future__ import annotations

import streamlit as st

from zhisaotong_agent.model.factory import CHAT_MODEL, EMBEDDING_MODEL
from zhisaotong_agent.rag.rag_service import RagSummarizeService
from zhisaotong_agent.utils.logger_handler import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="智扫通 · 扫地机器人问答", layout="centered")


@st.cache_resource(show_spinner="正在加载知识库...")
def get_service() -> RagSummarizeService:
    return RagSummarizeService()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"查看引用的 {len(sources)} 条资料"):
        for item in sources:
            st.markdown(
                f"**[{item['index']}] {item['source']}**  · RRF 融合分 {item['score']}"
            )
            st.caption(item["preview"] + "...")


def main() -> None:
    st.title("智扫通 · 扫地机器人智能客服")
    st.caption("基于 RAG：向量 + BM25 混合检索，回答带引用标注")

    try:
        service = get_service()
    except Exception as exc:  # noqa: BLE001
        st.error(f"初始化失败：{exc}")
        st.info("请检查 .env 里的 API Key 是否配置正确，以及是否已运行 build_index 建库。")
        st.stop()

    with st.sidebar:
        st.subheader("运行信息")
        st.write(f"对话模型：`{CHAT_MODEL}`")
        st.write(f"向量模型：`{EMBEDDING_MODEL}`")
        st.write(f"知识库分块：`{service._retriever.size}`")
        if st.button("清空对话"):
            st.session_state["messages"] = []
            st.rerun()
        st.divider()
        st.caption(
            "提示：问题里带上具体型号或场景，命中率更高。"
            "例如「X10 的尘盒多久清理一次」比「怎么清理」更好。"
        )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("请输入你的问题…")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("检索并生成回答…"):
            try:
                result = service.rag_answer(prompt)
            except Exception as exc:  # noqa: BLE001
                logger.exception("问答失败")
                st.error(f"处理失败：{exc}")
                return

        answer = result["answer"]
        st.markdown(answer)
        render_sources(result.get("sources") or [])

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result.get("sources") or [],
        }
    )


if __name__ == "__main__":
    main()
