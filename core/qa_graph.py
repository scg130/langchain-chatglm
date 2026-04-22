"""
RAG 问答用 LangGraph 编排（替代原 LCEL RunnableMap + get_qa_chain_with_history）。

- resolve：若请求里已有 context（如模板拼好的多源上下文），原样使用；否则从 retriever 拉取文档拼接。
- llm：调用 StableLLM（LangChain BaseLLM 兼容接口）。

流式端仍用同一套 resolve 逻辑后接 llm.astream，以保留按 token 输出；图结构便于后续加分支、重试、工具等。
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

# -------- 状态：与原 RunnableMap 入参、StableLLM.invoke 对齐 --------
class RAGState(TypedDict, total=False):
    query: str
    history: List[Tuple[str, str]]
    context: str
    answer: str


def _resolve_context_from_retriever(
    state: RAGState, retriever: Optional[Any]
) -> RAGState:
    """
    原 build_context 逻辑：优先外部 context；否则用 retriever.invoke(query)。
    """
    existing = state.get("context", "")
    if existing and str(existing).strip():
        return state
    if retriever is None:
        return {**state, "context": ""}
    try:
        docs = retriever.invoke(state.get("query", ""))
        text = "\n".join(getattr(d, "page_content", str(d)) for d in (docs or []))
        return {**state, "context": text}
    except Exception:
        return {**state, "context": ""}


def build_qa_rag_app(llm: Any, retriever: Optional[Any]):
    """编译 LangGraph：resolve -> llm -> END。"""

    def node_resolve(s: RAGState) -> RAGState:
        return _resolve_context_from_retriever(dict(s), retriever)

    def node_llm(s: RAGState) -> RAGState:
        ctx = s.get("context", "")
        ans = llm.invoke(
            {
                "query": s.get("query", ""),
                "history": s.get("history") or [],
                "context": ctx,
            }
        )
        return {**s, "answer": ans}

    g = StateGraph(RAGState)
    g.add_node("resolve", node_resolve)
    g.add_node("llm", node_llm)
    g.add_edge(START, "resolve")
    g.add_edge("resolve", "llm")
    g.add_edge("llm", END)
    return g.compile()


def merge_resolved_rag_state(prompt_input: Dict[str, Any], retriever: Any) -> Dict[str, Any]:
    """与图中 resolve 节点一致，供流式路径在 astream 前使用。"""
    state: RAGState = {
        "query": str(prompt_input.get("query", "")),
        "history": list(prompt_input.get("history") or []),
        "context": str(prompt_input.get("context", "")),
    }
    merged = _resolve_context_from_retriever(state, retriever)
    return {
        "query": merged.get("query", state["query"]),
        "history": merged.get("history", state.get("history") or []),
        "context": merged.get("context", ""),
    }


async def stream_rag_qa(
    llm: Any, retriever: Any, prompt_input: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """与旧 chain.astream 等价的流式：resolve 同图内逻辑，再 token 流式。"""
    merged = merge_resolved_rag_state(prompt_input, retriever)
    base = {
        "query": merged["query"],
        "history": merged.get("history") or [],
        "context": merged.get("context", ""),
    }
    async for chunk in llm.astream(base):
        yield chunk
