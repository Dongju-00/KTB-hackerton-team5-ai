"""
CLI 진입점. 반드시 ai/ 에서 실행한다.

    uv run python -m langgraph_rag.rag index            # 공공데이터 CSV를 벡터 스토어에 적재
    uv run python -m langgraph_rag.rag index-grief      # 펫로스 자료를 벡터 스토어에 적재
    uv run python -m langgraph_rag.rag "강남구 근처 병원"    # 한 번 질의(병원)
    uv run python -m langgraph_rag.rag grief            # 상담 대화 모드
    uv run python -m langgraph_rag.rag                  # 병원 대화 모드

main() -> 인자를 보고 색인/질의/대화를 고르는 함수
run_index_grief() -> 펫로스 자료를 색인하는 함수
run_grief_chat() -> 상담 그래프로 대화하는 함수 (시나리오 수동 검증용)
run_once() -> 그래프를 한 번 실행해 답변을 출력하는 함수
"""

import sys
import uuid

from langgraph.checkpoint.memory import MemorySaver

from data.grief_docs import load_grief_docs
from data.vet_data import load_vets
from langgraph_rag.graph import build_graph
from langgraph_rag.grief_graph import build_grief_graph
from langgraph_rag.grief_state import initial_grief_state
from langgraph_rag.state import initial_state
from langgraph_rag.vectorstore import index_grief_docs, index_vets


def run_index() -> None:
    """
    공공데이터 CSV를 읽어 벡터 스토어에 적재한다.
    """
    vets = load_vets()
    print(f"레코드 {len(vets):,}건 적재 시작")
    count = index_vets(vets)
    print(f"적재 완료: {count:,}건")


def run_index_grief() -> None:
    """
    data_raw/grief 의 마크다운을 읽어 벡터 스토어에 적재한다.
    출처 메타가 빠진 파일이 있으면 여기서 예외로 멈춘다.
    """
    docs = load_grief_docs()
    print(f"청크 {len(docs):,}건 적재 시작")
    count = index_grief_docs(docs)
    print(f"적재 완료: {count:,}건")


def run_grief_chat() -> None:
    """
    상담 그래프로 대화한다. 시나리오를 손으로 돌려보며 어조를 읽기 위한 경로다.
    반려견 정보는 처음 한 번만 묻는다.
    """
    graph = build_grief_graph(checkpointer=MemorySaver())
    thread_id = uuid.uuid4().hex

    pet_name = input("아이 이름(없으면 엔터): ").strip()
    rainbow_date = input("떠난 날 YYYY-MM-DD(없으면 엔터): ").strip()

    print("\n하고 싶은 이야기를 적어 주세요. 종료하려면 빈 줄 또는 Ctrl+C.")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        final = graph.invoke(
            initial_grief_state(question, pet_name, "", rainbow_date),
            config={"configurable": {"thread_id": thread_id}},
        )
        if final.get("crisis"):
            print("\n[위기 경로 - LLM 호출 없음]")
        print("\n" + final.get("answer", ""))
        if final.get("sources"):
            print(final["sources"])


def run_once(graph, question: str, lat=None, lng=None, region="", thread_id=None) -> dict:
    """
    그래프를 한 번 실행하고 최종 상태를 돌려준다.
    checkpointer가 붙어 있어 thread_id가 필요하다. 단발 실행은 매번 새 스레드를 쓴다.
    """
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    final = graph.invoke(initial_state(question, lat, lng, region), config=config)

    print("\n" + final.get("answer", ""))
    if final.get("sources"):
        print(final["sources"])

    return final


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "index":
        run_index()
        return

    if args and args[0] == "index-grief":
        run_index_grief()
        return

    if args and args[0] == "grief":
        run_grief_chat()
        return

    graph = build_graph(checkpointer=MemorySaver())

    if args:
        run_once(graph, " ".join(args))
        return

    # 대화 모드: 같은 thread_id를 유지해 이전 턴을 이어간다.
    thread_id = uuid.uuid4().hex
    print("질문을 입력하세요. 종료하려면 빈 줄 또는 Ctrl+C.")
    while True:
        question = input("\n> ").strip()
        if not question:
            break
        run_once(graph, question, thread_id=thread_id)


if __name__ == "__main__":
    main()
