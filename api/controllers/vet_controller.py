"""
그래프 실행을 담당하는 비즈니스 로직.
그래프는 한 번만 컴파일해 재사용한다.

질의 이력은 저장하지 않는다. 멀티턴은 checkpointer가 메모리에서 들고 있을 뿐,
서버가 내려가면 사라진다.

get_graph() -> 컴파일된 그래프를 한 번만 만들어 재사용하는 함수
run_search() -> 그래프를 실행해 결과를 돌려주는 함수
sse_event() -> dict를 SSE 한 블록으로 직렬화하는 함수 (상담 컨트롤러도 같이 쓴다)
stream_search() -> 진행 단계와 답변 토큰을 SSE로 흘려보내는 제너레이터
"""

import json
import uuid

from langchain_core.messages import AIMessageChunk
from threading import RLock

from langgraph.checkpoint.memory import MemorySaver

from langgraph_rag.graph import build_graph
from langgraph_rag.state import initial_state


_graph = None
_graph_lock = RLock()

# 그래프 노드명 -> 사용자에게 보여줄 진행 단계 문구.
# generate는 없다. 답변 토큰이 흐르기 시작하는 것이 곧 "작성 중" 신호다.
_STAGE_MESSAGES = {
    "resolve_location": "위치를 확인하고 있어요.",
    "search_vets": "근처 동물병원을 찾고 있어요.",
}


# sabu: 지연 초기화 — build_graph()가 check_keys()를 부르지만, 이 함수는 첫 요청이 올 때까지
#       그래프를 만들지 않는다. 키가 하나도 없는 채로 서버를 띄우면 기동은 성공한다.
#       운영자가 그 사실을 알게 되는 건 언제인가?
def get_graph():
    """
    컴파일된 그래프를 한 번만 만들어 재사용한다.

    멀티턴을 위해 checkpointer를 붙여 컴파일한다(thread_id로 턴 간 상태 유지).
    InMemory라 서버가 종료되면 대화도 함께 사라진다.
    """
    global _graph

    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_graph(checkpointer=MemorySaver())
    return _graph


def _to_items(vets: list[dict]) -> list[dict]:
    """
    상태의 병원 레코드를 응답 스키마(VetItem)가 받는 형태로 추린다.
    전화번호는 내려보내지 않는다 - 답하지 않기로 한 값이다.
    """
    return [
        {
            "name": v.get("name", ""),
            "address": v.get("address", ""),
            "road_address": v.get("road_address", ""),
            "distance_m": v.get("distance_m"),
            "map_url": v.get("map_url", ""),
        }
        for v in vets
    ]


def run_search(
    question: str,
    lat: float | None = None,
    lng: float | None = None,
    region: str = "",
    thread_id: str | None = None,
) -> dict:
    """
    그래프를 실행하고 결과를 돌려준다. 유료 호출은 그래프 실행당 1회로 제한된다.
    """
    thread_id = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    final = get_graph().invoke(initial_state(question, lat, lng, region), config=config)
    vets = final.get("vets", [])

    return {
        "status": "ok" if vets else "empty",
        "thread_id": thread_id,
        "answer": final.get("answer", ""),
        "sources": final.get("sources", ""),
        "vets": _to_items(vets),
    }


def sse_event(event: dict) -> str:
    """
    dict를 SSE 한 블록(data: ...\\n\\n)으로 직렬화한다.
    """
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def stream_search(
    question: str,
    lat: float | None = None,
    lng: float | None = None,
    region: str = "",
    thread_id: str | None = None,
):
    """
    진행 단계와 답변 토큰을 SSE로 흘려보낸다. 동작은 run_search와 같다.

    custom 스트림으로 노드 "시작"을, messages로 LLM 토큰을, updates로 최종 상태를 모은다.
    updates는 노드가 끝난 뒤에 나오므로 진행 표시에 쓰지 않는다.
    """
    thread_id = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    yield sse_event({"type": "thread", "thread_id": thread_id})

    final: dict = {}
    try:
        for mode, chunk in get_graph().stream(
            initial_state(question, lat, lng, region),
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for node, update in chunk.items():
                    if node in _STAGE_MESSAGES:
                        yield sse_event({"type": "stage", "node": node, "message": _STAGE_MESSAGES[node]})
                    final.update(update or {})
            elif mode == "messages":
                message, _meta = chunk
                # messages 스트림에는 LLM이 흘리는 토큰 조각(AIMessageChunk)뿐 아니라
                # 노드가 상태에 기록한 완성 메시지(history의 HumanMessage/AIMessage)도 섞여 나온다.
                # 걸러내지 않으면 답변이 끝난 뒤 질문과 답변이 통째로 다시 흘러 화면에 두 번 찍힌다.
                if not isinstance(message, AIMessageChunk):
                    continue
                text = getattr(message, "text", None) or getattr(message, "content", "")
                if isinstance(text, str) and text:
                    yield sse_event({"type": "token", "text": text})
    except Exception as exc:
        print(f"[api] stream_search error: {type(exc).__name__}: {exc}")
        yield sse_event(
            {
                "type": "done",
                "status": "error",
                "message": "안내를 불러오는 중 문제가 생겼어요. 잠시 후 다시 시도해 주세요.",
                "sources": "",
                "vets": [],
            }
        )
        return

    vets = final.get("vets", [])
    yield sse_event(
        {
            "type": "done",
            "status": "ok" if vets else "empty",
            "sources": final.get("sources", ""),
            "vets": _to_items(vets),
        }
    )
