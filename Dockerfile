# 데이터(data_raw)와 색인(chroma_db)은 이미지에 넣지 않는다.
# 둘 다 .gitignore 대상이라 CI 체크아웃에 없고, 운영에서는 ai-chroma 볼륨이 공급한다.
# 색인을 만드는 최초 1회 절차는 docs/DEPLOYMENT.md 를 따른다.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 의존성만 먼저 설치한다. 코드가 바뀌어도 이 레이어는 캐시에서 재사용된다.
# torch/torchvision은 pyproject의 pytorch-cpu 인덱스에서 받는다(GPU 빌드는 이미지가 몇 배로 커진다).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 디렉터리를 하나씩 나열하면 새로 생긴 패키지가 조용히 빠진다.
# 실제로 worker/, clients/ 가 그렇게 빠져 컨테이너가 ImportError 로 재시작 루프에 빠졌다.
# 제외 목록은 .dockerignore 한 곳에서만 관리한다.
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm

# Pillow가 여는 이미지 포맷과 HTTPS 호출에 필요한 최소 런타임만 남긴다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo libpng16-16 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --gid app --home /app app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

# HF 모델 가중치와 임베딩 모델을 ai-cache 볼륨에 떨어뜨린다.
# 이게 없으면 컨테이너가 재시작할 때마다 수백 MB를 다시 받는다.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Seoul \
    HF_HOME=/app/cache/huggingface

# compose가 볼륨을 붙이는 자리. 볼륨이 없어도 기동은 되게 미리 만든다.
RUN mkdir -p /app/cache /app/chroma_db && chown -R app:app /app/cache /app/chroma_db

USER app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
