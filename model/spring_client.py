import httpx


class SpringRegisterError(Exception):
    pass


def register_results(base_url: str, api_key: str, upload_ticket: str,
                     upload_id: str, pet_id: int, photos: list[dict],
                     timeout: float = 30.0) -> dict:
    """명세 8.1.3 — 성공 사진만 한 번에 등록"""
    url = f"{base_url.rstrip('/')}/internal/v1/photo-labeling-results"
    headers = {
        "X-Internal-Api-Key": api_key,
        "Idempotency-Key": upload_id,      # uploadId와 동일해야 함
        "X-Upload-Ticket": upload_ticket,
        "Content-Type": "application/json",
    }
    body = {"uploadId": upload_id, "petId": pet_id, "photos": photos}

    try:
        r = httpx.post(url, json=body, headers=headers, timeout=timeout)
    except Exception as e:
        raise SpringRegisterError(f"요청 실패: {e}")

    if r.status_code != 200:
        raise SpringRegisterError(f"HTTP {r.status_code}")

    data = r.json()
    if not data.get("success") or "data" not in data:
        raise SpringRegisterError("응답 계약 불일치")
    return data["data"]