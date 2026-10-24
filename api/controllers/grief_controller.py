"""
상담 그래프 실행을 담당하는 비즈니스 로직.
그래프는 한 번만 컴파일해 재사용한다.

대화 내용은 저장하지 않는다. 병원 모드와 같은 이유에 하나가 더 붙는다 -
여기 오가는 말이 훨씬 사적이다. checkpointer가 메모리에서 들고 있을 뿐,
서버가 내려가면 사라진다.

SSE 이벤트 종류는 thread / token / done 세 가지다.
병원 모드에 있는 stage(진행 단계 표시)는 보내지 않는다 - 상담에서 "찾고 있어요"가
말풍선 옆에 뜨는 것은 듣고 있다는 신호가 아니라 처리 중이라는 신호로 읽힌다(2026-08-20 결정).

get_grief_graph() -> 컴파일된 상담 그래프를 한 번만 만들어 재사용하는 함수
run_chat() -> 그래프를 실행해 결과를 돌려주는 함수
stream_chat() -> 진행 단계와 답변 토큰을 SSE로 흘려보내는 제너레이터
"""

import uuid
from threading import RLock

from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver

from api.controllers.vet_controller import sse_event
from langgraph_rag.grief_graph import build_grief_graph
from langgraph_rag.grief_state import initial_grief_state


_graph = None
_graph_lock = RLock()

def get_grief_graph():
    """
    컴파일된 상담 그래프를 한 번만 만들어 재사용한다.
    멀티턴을 위해 checkpointer를 붙여 컴파일한다.
    """
    global _graph

    if _graph is None:
        with _graph_lock:
            if _graph is None:
                _graph = build_grief_graph(checkpointer=MemorySaver())
    return _graph


def run_chat(
    question: str,
    pet_name: str = "",
    pet_gender: str = "",
    rainbow_date: str = "",
    thread_id: str | None = None,
) -> dict:
    """
    그래프를 실행하고 결과를 돌려준다. 유료 호출은 그래프 실행당 1회로 제한된다.
    위기 신호가 걸린 실행은 LLM을 부르지 않았다는 뜻으로 status를 crisis로 돌려준다.
    """
    thread_id = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    final = get_grief_graph().invoke(
        initial_grief_state(question, pet_name, pet_gender, rainbow_date),
        config=config,
    )

    return {
        "status": "crisis" if final.get("crisis") else "ok",
        "thread_id": thread_id,
        "answer": final.get("answer", ""),
        "sources": final.get("sources", ""),
    }


def stream_chat(
    question: str,
    pet_name: str = "",
    pet_gender: str = "",
    rainbow_date: str = "",
    thread_id: str | None = None,
):
    """
    진행 단계와 답변 토큰을 SSE로 흘려보낸다.

    messages로 LLM 토큰을, updates로 최종 상태를 모은다. 진행 단계는 보내지 않는다.

    위기 응답만은 messages 스트림에 흐르지 않는다. LLM을 거치지 않기 때문이다.
    그래서 check_safety의 updates를 보고 고정 문구를 token 이벤트로 직접 실어 보낸다 -
    이걸 빼먹으면 프론트에는 빈 말풍선만 남는다.
    """
    thread_id = thread_id or uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}

    yield sse_event({"type": "thread", "thread_id": thread_id})

    final: dict = {}
    try:
        for mode, chunk in get_grief_graph().stream(
            initial_grief_state(question, pet_name, pet_gender, rainbow_date),
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "updates":
                for node, update in chunk.items():
                    final.update(update or {})
                    if node == "check_safety" and (update or {}).get("crisis"):
                        yield sse_event({"type": "token", "text": update["answer"]})
            elif mode == "messages":
                message, _meta = chunk
                # 노드가 상태에 기록한 완성 메시지(history)가 섞여 나온다. 걸러내지 않으면
                # 답변이 끝난 뒤 질문과 답변이 통째로 다시 흘러 화면에 두 번 찍힌다.
                if not isinstance(message, AIMessageChunk):
                    continue
                text = getattr(message, "text", None) or getattr(message, "content", "")
                if isinstance(text, str) and text:
                    yield sse_event({"type": "token", "text": text})
    except Exception as exc:
        print(f"[api] stream_chat error: {type(exc).__name__}: {exc}")
        yield sse_event(
            {
                "type": "done",
                "status": "error",
                "message": "잠시 문제가 생겼어요. 조금 뒤에 다시 이야기해 주세요.",
                "sources": "",
            }
        )
        return

    yield sse_event(
        {
            "type": "done",
            "status": "crisis" if final.get("crisis") else "ok",
            "sources": final.get("sources", ""),
        }
    )
