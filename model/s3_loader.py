import boto3
from botocore.exceptions import ClientError

MAX_BYTES = 10 * 1024 * 1024


class S3NotFound(Exception): pass
class S3ReadFailed(Exception): pass
class ImageTooLarge(Exception): pass
class InvalidImage(Exception): pass


_s3 = None

def _client(region: str):
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=region)   # 기본 credential chain
    return _s3


def detect_format(data: bytes) -> str | None:
    """명세 요구: 확장자가 아닌 파일 signature로 판별"""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def fetch_image(bucket: str, region: str, key: str) -> bytes:
    try:
        obj = _client(region).get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "404"):
            raise S3NotFound(key)
        raise S3ReadFailed(code)

    size = obj.get("ContentLength", 0)
    if size > MAX_BYTES:
        raise ImageTooLarge(key)

    data = obj["Body"].read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ImageTooLarge(key)
    if len(data) == 0:
        raise InvalidImage("빈 파일")
    if detect_format(data) is None:
        raise InvalidImage("지원하지 않는 형식")
    return data