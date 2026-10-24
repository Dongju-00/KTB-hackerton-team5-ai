"""
엔드포인트
- POST /api/vets        : 질문 + 위치 -> 근처 동물병원 안내
- POST /api/vets/stream : 같은 동작의 SSE 스트리밍 버전

질의 이력 조회는 없다. 이력을 저장하지 않기 때문이다.

엔드포인트는 동기다. 그래프 invoke는 임베딩/LLM으로 블로킹되는데,
FastAPI가 동기 경로를 스레드풀에서 실행하므로 이벤트 루프를 막지 않는다.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.controllers import vet_controller
from api.schemas.vet_schema import ApiResponse, VetSearchRequest, VetSearchResponse


router = APIRouter(prefix="/api", tags=["vets"])


@router.post(
    "/vets",
    response_model=ApiResponse[VetSearchResponse],
    response_model_exclude_none=True,
)
def create_search(payload: VetSearchRequest):
    result = vet_controller.run_search(
        question=payload.question,
        lat=payload.lat,
        lng=payload.lng,
        region=payload.region or "",
        thread_id=payload.thread_id,
    )
    return {"success": True, "data": result, "message": "병원 검색에 성공했습니다."}


@router.post("/vets/stream")
def create_search_stream(payload: VetSearchRequest):
    """
    진행 단계와 답변 토큰을 SSE로 흘려보낸다.
    """
    return StreamingResponse(
        vet_controller.stream_search(
            question=payload.question,
            lat=payload.lat,
            lng=payload.lng,
            region=payload.region or "",
            thread_id=payload.thread_id,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
