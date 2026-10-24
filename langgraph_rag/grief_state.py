"""
상담 그래프가 노드 사이에 넘기는 공유 상태 스키마.

병원 그래프(VetSearchState)와 나눠 둔다. 좌표·반경·지역 후보가 상담에는 하나도 쓰이지 않고,
같은 TypedDict에 두 모드의 필드를 섞으면 어느 노드가 무엇을 채우는지가 다시 흐려진다.

GriefState -> 상담 그래프 공유 상태
initial_grief_state() -> 첫 실행에 쓸 초기 상태를 만드는 함수
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


TOP_K = 4          # 답변 근거로 실을 자료 청크 수
SEARCH_K = 12      # 임계값으로 걸러낼 여지를 남기려고 넉넉히 뽑는다


class GriefState(TypedDict, total=False):
    # 입력
    question: str
    pet_name: str
    pet_gender: str
    rainbow_date: str      # 'YYYY-MM-DD'. 빈 문자열이면 기일을 모르는 상태다.

    # 중간 결과
    crisis: bool           # 위기 신호 감지 여부. True면 검색·생성을 건너뛴다.
    docs: list[dict]       # 근거로 실을 자료 청크 (title/source/url/section/text)

    # 출력
    answer: str
    sources: str

    # 예산 방어: 유료 호출은 그래프 실행당 1회
    paid_call_count: int

    # 멀티턴
    history: Annotated[list[BaseMessage], add_messages]


def initial_grief_state(
    question: str,
    pet_name: str = "",
    pet_gender: str = "",
    rainbow_date: str = "",
) -> GriefState:
    """
    첫 실행에 쓸 초기 상태를 만든다.

    반려견 정보는 없어도 동작해야 한다. 이름을 모르면 이름을 부르지 않을 뿐,
    상담 자체가 막히지는 않는다(프론트가 pet 없이 열 수 있는 경로가 남아 있다).
    paid_call_count를 여기서 0으로 초기화하는 이유는 병원 그래프와 같다.
    """
    return {
        "question": question,
        "pet_name": pet_name,
        "pet_gender": pet_gender,
        "rainbow_date": rainbow_date,
        "crisis": False,
        "docs": [],
        "answer": "",
        "sources": "",
        "paid_call_count": 0,
    }
