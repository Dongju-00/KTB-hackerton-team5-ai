"""
POST /vets 요청/응답과 이력 조회 응답 스키마.

병원 목록은 answer 문장 안이 아니라 vets 배열로도 내려보낸다.
프론트가 LLM 문장을 파싱하지 않고 구조화된 값으로 카드를 그릴 수 있어야 하기 때문이다.

모든 JSON 응답은 팀 규약(docs/API_SPEC.md 3절)의 ApiResponse envelope로 감싼다.
SSE 스트림은 envelope를 쓰지 않는다 - 이벤트 스트림이지 단일 JSON 응답이 아니다.
"""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel


class VetSearchRequest(BaseModel):
    question: str
    lat: float | None = None       # 프론트가 브라우저 위치를 넘길 때
    lng: float | None = None
    region: str | None = None      # '서울특별시 강남구' 형식
    thread_id: str | None = None   # 있으면 후속 턴(같은 대화 이어가기)


class VetItem(BaseModel):
    """답변에 실린 병원 한 곳. 링크는 코드가 원본에서 꺼낸 값이다."""

    name: str = ""
    address: str = ""
    road_address: str = ""
    distance_m: float | None = None
    map_url: str = ""              # 네이버 지도 검색 URL. 영업시간·전화번호를 여기서 확인한다


class VetSearchResponse(BaseModel):
    status: Literal["ok", "empty"]
    thread_id: str
    answer: str = ""
    sources: str = ""              # 코드가 조립한 '확인 링크' 섹션
    vets: list[VetItem] = []



T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """
    팀 공통 응답 envelope. 값이 None인 필드는 라우터의 exclude_none 설정으로 생략된다.
    프론트 분기는 message 문자열이 아니라 errorCode를 기준으로 한다.
    """

    success: bool
    data: T | None = None
    message: str
    errorCode: str | None = None
