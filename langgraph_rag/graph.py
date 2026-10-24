"""
노드를 엮어 그래프로 만든다.

파이프라인의 순서가 함수 호출 순서가 아니라 여기 선언으로 드러난다.
결과가 0건이어도 되돌아가지 않는다. 반경 안에 없으면 없다고 답한다.

build_graph() -> 컴파일된 그래프를 만드는 함수
"""

from langgraph.graph import END, StateGraph

from data.config import check_keys

from langgraph_rag.nodes import generate, resolve_location, search_vets
from langgraph_rag.state import VetSearchState


def build_graph(checkpointer=None):
    """
    resolve_location -> search_vets -> generate

    checkpointer를 주면 thread_id로 턴 간 상태가 유지된다(멀티턴).

    키 검사를 여기서 한다. 그래프를 세우는 시점이 파이프라인의 가장 앞이라,
    임베딩 로드/검색/링크 보강을 다 하고 generate에서 터지는 것을 막는다.
    """
    check_keys()

    graph = StateGraph(VetSearchState)

    graph.add_node("resolve_location", resolve_location)
    graph.add_node("search_vets", search_vets)
    graph.add_node("generate", generate)

    graph.set_entry_point("resolve_location")
    graph.add_edge("resolve_location", "search_vets")
    graph.add_edge("search_vets", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=checkpointer)
