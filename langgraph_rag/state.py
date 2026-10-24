"""
그래프가 노드 사이에 넘기는 공유 상태 스키마.

지역 변수로 흘려보내던 중간 결과를 명시적 필드로 올려, 어느 노드가 무엇을 채우는지 드러낸다.

VetSearchState -> 그래프 공유 상태
initial_state() -> 첫 실행에 쓸 초기 상태를 만드는 함수
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# 검색 반경. 이 안에 병원이 없으면 넓히지 않고 "없음"으로 답한다.
INITIAL_RADIUS_M = 10_000.0

# 사용자 위치를 아직 입력으로 받지 않아 임시로 고정한 기준점(강남역, 2026-08-19).
# 요청에 lat/lng가 오면 그쪽이 우선이다.
DEFAULT_LAT = 37.4979
DEFAULT_LNG = 127.0276
TOP_N = 5                # 답변에 실을 병원 수 (네이버 링크 보강도 이 수만큼만 호출한다)


class VetSearchState(TypedDict, total=False):
    # 입력
    question: str
    lat: float | None
    lng: float | None
    region: str

    # 검색 제어
    radius_m: float

    # 중간 결과
    vets: list[dict]          # 최종 후보 병원 (거리순)
    candidates: list[str]     # 지역이 모호할 때 제시할 지역 후보

    # 출력
    answer: str
    sources: str

    # 예산 방어: 유료 호출은 그래프 실행당 1회
    paid_call_count: int

    # 멀티턴
    history: Annotated[list[BaseMessage], add_messages]


def initial_state(
    question: str,
    lat: float | None = None,
    lng: float | None = None,
    region: str = "",
) -> VetSearchState:
    """
    첫 실행에 쓸 초기 상태를 만든다.
    좌표가 없으면 고정 기준점(DEFAULT_LAT/LNG)을 쓴다.
    paid_call_count를 여기서 0으로 초기화한다 - 모듈 전역 카운터로 두면 여러 실행이 서로의 예산을 갉아먹는다.
    """
    return {
        "question": question,
        "lat": DEFAULT_LAT if lat is None else lat,
        "lng": DEFAULT_LNG if lng is None else lng,
        "region": region,
        "radius_m": INITIAL_RADIUS_M,
        "vets": [],
        "candidates": [],
        "answer": "",
        "sources": "",
        "paid_call_count": 0,
    }
