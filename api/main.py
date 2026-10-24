"""
FastAPI 앱 진입점. vets, grief, 사진 라벨링 라우터를 등록하고
오류 응답을 팀 규약 envelope로 맞춘다.

기동 시 라벨링 모델을 한 번 로드하고 백그라운드 워커를 시작한다.
모델 로드나 필수 설정에 실패해도 챗봇 API는 계속 동작한다.

실행: ai/ 에서 `uv run uvicorn api.main:app --reload`
문서: http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routers import dog_labeling_router, grief_router, vet_router
from model import classifier
from worker import photo_labeling_worker


# HTTP 상태 -> 팀 규약 4.1의 오류 코드
_ERROR_CODES = {400: "COMMON_400", 401: "AUTH_401", 403: "AUTH_403",
                404: "COMMON_404", 405: "COMMON_405", 500: "COMMON_500"}


def _error(status: int, message: str) -> JSONResponse:
    """실패 응답. 규약 3.2에 따라 data는 싣지 않는다."""
    return JSONResponse(
        status_code=status,
        content={
            "success": False,
            "message": message,
            "errorCode": _ERROR_CODES.get(status, "COMMON_500"),
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """모델을 한 번 로드하고 워커를 시작한다.

    라벨링이 준비되지 않아도 vets/grief 챗봇은 계속 서비스해야 하므로
    로드 실패를 여기서 흡수한다. 라벨링 경로는 is_ready()로 판단해
    503을 반환한다.
    """
    try:
        classifier.load()
    except Exception as e:
        print(f"[api] 라벨링 모델 로드 실패: {type(e).__name__}: {e}")

    if os.getenv("PHOTO_LABELING_WORKER_ENABLED", "true").lower() == "true":
        await photo_labeling_worker.start()
    else:
        print("[api] 워커가 설정으로 비활성화되었습니다")

    yield

    await photo_labeling_worker.stop()


# 프론트가 브라우저에서 이 서버를 직접 호출한다(2026-08-19 결정).
# Origin은 열거한 것만 허용한다. "*" + allow_credentials 조합은 스펙 위반이자
# 아무 사이트나 인증정보를 실어 호출할 수 있게 되는 구멍이라 쓰지 않는다.
# 이 API는 쿠키·인증을 쓰지 않으므로 allow_credentials는 False로 둔다.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:8443,http://127.0.0.1:8443,"
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]

app = FastAPI(
    title="반려견 추모공간 - 동물병원 안내 / 펫로스 상담 API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(vet_router.router)
app.include_router(grief_router.router)
app.include_router(dog_labeling_router.router)


@app.get("/health")
def health():
    """챗봇 가용성과 라벨링 준비 상태를 구분해 노출한다."""
    return {
        "status": "ok",
        "labelingReady": classifier.is_ready(),
        "workerRunning": photo_labeling_worker.is_running(),
    }


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """
    FastAPI 기본값은 422 + {"detail": [...]}지만,
    팀 규약 4.1은 검증 실패를 400 COMMON_400으로 정의한다.
    """
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(x) for x in first.get("loc", ())[1:]) or "요청"
    return _error(400, f"{field}: {first.get('msg', '잘못된 요청입니다.')}")


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException):
    """
    fastapi.HTTPException이 아니라 Starlette 쪽에 건다.
    라우터가 경로를 못 찾아 내는 404/405는 Starlette의 예외라,
    FastAPI 클래스에 걸면 안 잡힌다.
    (fastapi.HTTPException은 이 클래스의 하위 클래스라 함께 잡힌다.)
    """
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):
    """처리되지 않은 예외. 내부 메시지를 그대로 노출하지 않는다."""
    print(f"[api] 처리되지 않은 예외: {type(exc).__name__}: {exc}")
    return _error(500, "서버 내부 오류가 발생했습니다.")