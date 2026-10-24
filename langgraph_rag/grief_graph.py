"""
상담 노드를 엮어 그래프로 만든다.

check_safety -> (위기면) END
             -> (아니면) retrieve_docs -> generate -> END

병원 그래프와 달리 분기가 하나 있다. 그 분기가 이 그래프의 존재 이유다.

build_grief_graph() -> 컴파일된 상담 그래프를 만드는 함수
"""

from langgraph.graph import END, StateGraph

from data.config import check_keys

from langgraph_rag.grief_nodes import (
    check_safety,
    generate,
    retrieve_docs,
    route_after_safety,
)
from langgraph_rag.grief_state import GriefState


def build_grief_graph(checkpointer=None):
    """
    checkpointer를 주면 thread_id로 턴 간 상태가 유지된다(멀티턴).

    키 검사를 여기서 하는 이유는 병원 그래프와 같다 - 임베딩 로드와 검색을 다 하고
    generate에서 터지는 것을 막는다.
    """
    check_keys()

    graph = StateGraph(GriefState)

    graph.add_node("check_safety", check_safety)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("generate", generate)

    graph.set_entry_point("check_safety")
    graph.add_conditional_edges(
        "check_safety",
        route_after_safety,
        {"retrieve": "retrieve_docs", "end": END},
    )
    graph.add_edge("retrieve_docs", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)
