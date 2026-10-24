import asyncio
import os
import random
import socket
import uuid

from clients import spring_jobs_client as spring
from model import classifier, s3_loader

WORKER_ID = os.getenv("PHOTO_LABELING_WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}"
CLAIM_LIMIT = int(os.getenv("PHOTO_LABELING_CLAIM_LIMIT", "10"))
IDLE_MIN = float(os.getenv("PHOTO_LABELING_IDLE_MIN_SECONDS", "1"))
IDLE_MAX = float(os.getenv("PHOTO_LABELING_IDLE_MAX_SECONDS", "5"))
BUCKET = os.getenv("AWS_S3_BUCKET", "")
REGION = os.getenv("AWS_REGION", "ap-northeast-2")

_stop_event = asyncio.Event()
_task: asyncio.Task | None = None


def _process_one(job: dict) -> None:
    """7절 처리 규칙 — 한 사진 처리"""
    item_id = job["itemId"]
    object_key = job["objectKey"]
    try:
        data = s3_loader.fetch_image(BUCKET, REGION, object_key)
        tags = classifier.predict_tags(data)
        spring.report_success(item_id, tags)
    except s3_loader.S3NotFound:
        spring.report_failure(item_id, "LABELING_S3_OBJECT_NOT_FOUND", "객체를 찾을 수 없습니다.", retryable=False)
    except s3_loader.S3ReadFailed:
        spring.report_failure(item_id, "LABELING_S3_READ_FAILED", "S3 조회에 실패했습니다.", retryable=True)
    except s3_loader.ImageTooLarge:
        spring.report_failure(item_id, "LABELING_IMAGE_TOO_LARGE", "10MB를 초과합니다.", retryable=False)
    except s3_loader.InvalidImage:
        spring.report_failure(item_id, "LABELING_INVALID_IMAGE", "지원하는 이미지 형식이 아닙니다.", retryable=False)
    except Exception:
        spring.report_failure(item_id, "LABELING_MODEL_FAILED", "모델 추론에 실패했습니다.", retryable=True)


async def _run_loop():
    idle = IDLE_MIN
    while not _stop_event.is_set():
        try:
            jobs = await asyncio.to_thread(spring.claim_jobs, WORKER_ID, CLAIM_LIMIT)
        except spring.SpringJobsError:
            await asyncio.sleep(idle)
            idle = min(idle * 2, IDLE_MAX)
            continue

        if not jobs:
            await asyncio.sleep(idle + random.uniform(0, 0.5))   # 8절 jitter
            idle = min(idle * 2, IDLE_MAX)
            continue

        idle = IDLE_MIN   # 작업 있었으니 초기화
        for job in jobs:
            if _stop_event.is_set():
                break
            await asyncio.to_thread(_process_one, job)


async def start():
    global _task
    if not classifier.is_ready():
        print("[worker] 모델 미준비 — 워커를 시작하지 않습니다")
        return
    if not spring.INTERNAL_KEY or not BUCKET:
        print("[worker] 필수 환경변수 누락 — 워커를 시작하지 않습니다")
        return
    _stop_event.clear()
    _task = asyncio.create_task(_run_loop())
    print(f"[worker] 시작됨 (worker_id={WORKER_ID})")


async def stop():
    _stop_event.set()
    if _task:
        await asyncio.wait_for(_task, timeout=15)
    print("[worker] 정상 종료됨")

def is_running() -> bool:
    return _task is not None and not _task.done()