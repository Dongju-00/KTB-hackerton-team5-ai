import base64, hashlib, hmac, json, time


class TicketError(Exception):
    pass


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def verify_upload_ticket(ticket: str, secret: str) -> dict:
    """명세 8.1.1 — Base64URL 인코딩된 payload 문자열 자체에 서명 검증"""
    parts = ticket.split(".")
    if len(parts) != 2:
        raise TicketError("티켓 형식이 올바르지 않습니다")
    payload_b64, sig_b64 = parts

    expected = hmac.new(secret.encode("utf-8"),
                        payload_b64.encode("utf-8"),
                        hashlib.sha256).digest()
    try:
        actual = _b64url_decode(sig_b64)
    except Exception:
        raise TicketError("서명 디코드 실패")

    if not hmac.compare_digest(expected, actual):
        raise TicketError("서명이 일치하지 않습니다")

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        raise TicketError("payload 파싱 실패")

    for k in ("uploadId", "userId", "petId", "objectKeys", "expiresAt"):
        if k not in payload:
            raise TicketError(f"claim 누락: {k}")
    if not payload["objectKeys"]:
        raise TicketError("objectKeys가 비어 있습니다")
    if int(payload["expiresAt"]) <= int(time.time()):
        raise TicketError("만료된 티켓입니다")

    return payload


def check_scope(payload: dict, upload_id: str, pet_id: int, object_keys: list[str]):
    if payload["uploadId"] != upload_id:
        raise TicketError("uploadId 불일치")
    if int(payload["petId"]) != int(pet_id):
        raise TicketError("petId 불일치")
    allowed = set(payload["objectKeys"])
    for k in object_keys:
        if k not in allowed:
            raise TicketError(f"허용되지 않은 objectKey: {k}")