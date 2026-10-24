"""
상담 그래프 노드.

각 노드는 GriefState를 받아 자기가 채운 필드만 dict로 돌려준다.

진행 단계(stage)는 흘리지 않는다. 병원 모드와 다른 점이다.

check_safety가 맨 앞이다. 위기 신호가 잡히면 검색도 생성도 하지 않고 바로 끝낸다 -
가장 중요한 순간에 LLM을 거치지 않기 위해 순서를 이렇게 둔다.

check_safety() -> 위기 신호를 판정하고, 걸리면 고정 응답으로 끝내는 노드 (LLM 호출 없음)
retrieve_docs() -> 펫로스 자료를 검색해 근거로 실을 청크를 고르는 노드
generate() -> 답변을 생성하고 출처를 붙이는 노드 (유료 호출 1회)
route_after_safety() -> 위기 여부로 다음 노드를 고르는 분기 함수
"""

from langchain_core.messages import AIMessage, HumanMessage

from langgraph_rag import vectorstore
from langgraph_rag.grief_chain import (
    CITATION_HIDDEN,
    CITATION_SHOWN,
    build_grief_chain,
    build_grief_context,
    build_pet_context,
    render_grief_sources,
    strip_citations,
    wants_sources,
)
from langgraph_rag.grief_state import SEARCH_K, TOP_K, GriefState
from langgraph_rag.safety import CRISIS_RESPONSE, detect_crisis


def check_safety(state: GriefState) -> dict:
    """
    위기 신호를 판정한다. 걸리면 answer를 고정 문구로 채우고 끝낸다.

    여기서 answer를 채우는 것이 이 설계의 핵심이다. 뒤 노드에 맡기면
    언젠가 그 문구가 LLM을 거치게 되고, 그 순간 보장이 사라진다.
    """
    if not detect_crisis(state.get("question", "")):
        return {"crisis": False}

    return {
        "crisis": True,
        "answer": CRISIS_RESPONSE,
        "sources": "",
        # 위기 응답도 대화 이력에는 남긴다. 다음 턴의 LLM이 방금 무슨 말이 오갔는지 알아야 한다.
        "history": [HumanMessage(state["question"]), AIMessage(CRISIS_RESPONSE)],
    }


def route_after_safety(state: GriefState) -> str:
    """
    위기면 끝, 아니면 자료 검색으로.
    """
    return "end" if state.get("crisis") else "retrieve"


def retrieve_docs(state: GriefState) -> dict:
    """
    펫로스 자료를 검색해 근거로 실을 청크를 고른다.

    지역 필터도 거리 재정렬도 없다. 상담에서 '가까움'은 축이 아니다.

    sabu: 임계값 재사용 — filter_by_relevance의 기본값 0.3은 병원 이름 매칭 기준으로 잡은 값이고,
          이 함수는 그 기본값을 그대로 쓴다. 게다가 전부 걸러지면 최고점 하나를 남기도록
          되어 있다(빈손 방지). 상담에서 '아무 자료도 관련 없음'과 '억지로 한 건 끼워 넣음' 중
          어느 쪽이 프롬프트에 더 나쁜가? 그 판단은 어느 계층에서 해야 하나?
    """
    results = vectorstore.search(
        query=state["question"],
        k=SEARCH_K,
        collection_name=vectorstore.GRIEF_COLLECTION_NAME,
    )
    kept = vectorstore.filter_by_relevance(results)[:TOP_K]

    return {"docs": [dict(doc.metadata, text=doc.page_content) for doc, _score in kept]}


def generate(state: GriefState) -> dict:
    """
    답변을 생성하고 출처를 코드가 붙인다. 유료 호출은 실행당 1회로 막는다.

    자료 번호는 보호자가 근거를 물었을 때만 남긴다. 물어보지 않았는데 붙어 나오면
    strip_citations()가 지운다 - 프롬프트는 부탁이고, 이쪽이 보장이다.
    """
    if state.get("paid_call_count", 0) >= 1:
        raise RuntimeError("이번 실행의 유료 호출 한도(1회)를 이미 썼습니다.")

    docs = state.get("docs", [])
    show_sources = wants_sources(state["question"])
    answer = build_grief_chain().invoke(
        {
            "pet": build_pet_context(
                state.get("pet_name", ""),
                state.get("pet_gender", ""),
                state.get("rainbow_date", ""),
            ),
            "context": build_grief_context(docs),
            "citation_rule": CITATION_SHOWN if show_sources else CITATION_HIDDEN,
            "question": state["question"],
            "history": state.get("history", []),
        }
    )
    if not show_sources:
        answer = strip_citations(answer)

    return {
        "answer": answer,
        "sources": render_grief_sources(answer, docs) if show_sources else "",
        "paid_call_count": state.get("paid_call_count", 0) + 1,
        "history": [HumanMessage(state["question"]), AIMessage(answer)],
    }
