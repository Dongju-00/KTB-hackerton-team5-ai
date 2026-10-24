import os
import httpx


class SpringJobsError(Exception):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


SPRING_BASE = os.getenv("SPRING_INTERNAL_BASE_URL", "http://backend:8080")
INTERNAL_KEY = os.getenv("FASTAPI_INTERNAL_API_KEY", "")


def claim_jobs(worker_id: str, limit: int = 10, lease_seconds: int = 600) -> list[dict]:
    """6절 Claim — 처리할 작업을 가져온다. 빈 배열도 정상."""
    url = f"{SPRING_BASE}/internal/v1/photo-labeling-jobs/claims"
    try:
        r = httpx.post(url,
            json={"workerId": worker_id, "limit": limit, "leaseSeconds": lease_seconds},
            headers={"X-Internal-Api-Key": INTERNAL_KEY},
            timeout=10.0)
    except Exception as e:
        raise SpringJobsError(f"claim 요청 실패: {e}", retryable=True)

    if r.status_code != 200:
        raise SpringJobsError(f"claim HTTP {r.status_code}", retryable=True)

    data = r.json()
    jobs = data.get("jobs", [])
    # 필수 필드 검증 (5절: "Validate before using")
    valid = []
    for j in jobs:
        if all(k in j for k in ("itemId", "petId", "objectKey")):
            valid.append(j)
    return valid


def report_success(item_id: str, tags: list[str]):
    """6절 결과 보고 — 성공"""
    url = f"{SPRING_BASE}/internal/v1/photo-labeling-jobs/{item_id}/result"
    r = httpx.post(url,
        json={"status": "SUCCEEDED", "tags": tags},
        headers={"X-Internal-Api-Key": INTERNAL_KEY, "Idempotency-Key": item_id},
        timeout=10.0)
    if r.status_code != 200:
        raise SpringJobsError(f"result 보고 실패 HTTP {r.status_code}", retryable=True)


def report_failure(item_id: str, error_code: str, message: str, retryable: bool):
    """6절 결과 보고 — 실패"""
    url = f"{SPRING_BASE}/internal/v1/photo-labeling-jobs/{item_id}/result"
    r = httpx.post(url,
        json={"status": "FAILED", "errorCode": error_code,
              "message": message, "retryable": retryable},
        headers={"X-Internal-Api-Key": INTERNAL_KEY, "Idempotency-Key": item_id},
        timeout=10.0)
    if r.status_code != 200:
        raise SpringJobsError(f"실패 보고 실패 HTTP {r.status_code}", retryable=True)