"""
POST /grief 요청/응답 스키마.

병원 응답의 vets 배열에 해당하는 것이 없다. 상담에는 카드로 그릴 목록이 없고,
근거 자료는 answer 아래 sources 문자열로만 붙는다.

모든 JSON 응답은 팀 규약(docs/API_SPEC.md 3절)의 ApiResponse envelope로 감싼다.
SSE 스트림은 envelope를 쓰지 않는다.
"""

from typing import Literal

from pydantic import BaseModel


class GriefChatRequest(BaseModel):
    question: str
    pet_name: str | None = None      # 추모 데이터 연계. 없으면 이름을 부르지 않는다
    pet_gender: str | None = None    # 'M' | 'F'
    rainbow_date: str | None = None  # 'YYYY-MM-DD'
    thread_id: str | None = None     # 있으면 후속 턴(같은 대화 이어가기)


class GriefChatResponse(BaseModel):
    status: Literal["ok", "crisis"]  # crisis면 LLM을 거치지 않은 고정 응답이다
    thread_id: str
    answer: str = ""
    sources: str = ""                # 코드가 조립한 '참고한 자료' 섹션
