"""
GBrain 入口：与示例一致，但记忆层已改为按 user + 当前 query 做语义召回。
若未使用 LangGraph，可只 import Memory + ModelRouter 自行拼管线。
"""

from __future__ import annotations

from typing import Any, Optional

from brain.memory import Memory

# 以下依赖未在仓库中实现时，请按需提供或改为占位
try:
    from brain.router import ModelRouter  # type: ignore
except ImportError:
    ModelRouter = None  # type: ignore

try:
    from agent.graph import build_graph  # type: ignore
except ImportError:
    build_graph = None  # type: ignore


class GBrain:
    def __init__(self) -> None:
        self.memory = Memory()
        self.router = ModelRouter() if ModelRouter else None
        self.graph = build_graph() if build_graph else None

    def run(self, user_id: str, query: str) -> Any:
        # 用「当前问题」在向量库中检索该用户相关历史，而不是用 user_id 当查询词
        context = self.memory.get_context(user_id, current_query=query, n_results=5)

        model: Optional[str] = None
        if self.router is not None:
            model = self.router.route(query)  # type: ignore[union-attr]

        if self.graph is None:
            return {"error": "graph not available", "context": context, "model": model}

        result: Any = self.graph.invoke(  # type: ignore[union-attr]
            {
                "query": query,
                "context": context,
                "model": model,
            }
        )
        self.memory.save(user_id, query, result)
        return result
