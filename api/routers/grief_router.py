"""
엔드포인트
- POST /api/grief        : 펫로스 상담 한 턴
- POST /api/grief/stream : 같은 동작의 SSE 스트리밍 버전

대화 이력 조회는 없다. 이력을 저장하지 않기 때문이다.

엔드포인트를 병원 쪽과 나눈 이유: 프론트가 pet.status로 부를 곳을 고른다(2026-08-20 결정).
서버는 어느 반려견이 살아 있는지 모르고, 알 필요도 없다.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.controllers import grief_controller
from api.schemas.grief_schema import GriefChatRequest, GriefChatResponse
from api.schemas.vet_schema import ApiResponse


router = APIRouter(prefix="/api", tags=["grief"])


@router.post(
    "/grief",
    response_model=ApiResponse[GriefChatResponse],
    response_model_exclude_none=True,
)
def create_chat(payload: GriefChatRequest):
    result = grief_controller.run_chat(
        question=payload.question,
        pet_name=payload.pet_name or "",
        pet_gender=payload.pet_gender or "",
        rainbow_date=payload.rainbow_date or "",
        thread_id=payload.thread_id,
    )
    return {"success": True, "data": result, "message": "상담 응답에 성공했습니다."}


@router.post("/grief/stream")
def create_chat_stream(payload: GriefChatRequest):
    """
    진행 단계와 답변 토큰을 SSE로 흘려보낸다.
    """
    return StreamingResponse(
        grief_controller.stream_chat(
            question=payload.question,
            pet_name=payload.pet_name or "",
            pet_gender=payload.pet_gender or "",
            rainbow_date=payload.rainbow_date or "",
            thread_id=payload.thread_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
