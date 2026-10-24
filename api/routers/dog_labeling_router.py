import os
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas.dog_labeling_schema import LabelingRequest
from model import classifier, s3_loader, spring_client, ticket

router = APIRouter()

SECRET      = os.getenv("PHOTO_UPLOAD_TICKET_SECRET", "")
INTERNAL_KEY = os.getenv("FASTAPI_INTERNAL_API_KEY", "")
SPRING_URL  = os.getenv("SPRING_INTERNAL_BASE_URL", "http://localhost:8080")
BUCKET      = os.getenv("AWS_S3_BUCKET", "pawprint-local")
REGION      = os.getenv("AWS_REGION", "ap-northeast-2")


def fail(status: int, code: str, msg: str):
    return JSONResponse(status_code=status,
                        content={"success": False, "message": msg, "errorCode": code})


@router.post("/v1/photo-labeling-requests")
def label_photos(req: LabelingRequest):
    # 1. 중복 검증
    if len({p.clientFileId for p in req.photos}) != len(req.photos):
        return fail(400, "LABELING_400", "clientFileId가 중복되었습니다")
    if len({p.objectKey for p in req.photos}) != len(req.photos):
        return fail(400, "LABELING_400", "objectKey가 중복되었습니다")

    # 2. 티켓 검증
    try:
        payload = ticket.verify_upload_ticket(req.uploadTicket, SECRET)
        ticket.check_scope(payload, req.uploadId, req.petId,
                           [p.objectKey for p in req.photos])
    except ticket.TicketError as e:
        return fail(401, "LABELING_401", str(e))

    if not classifier.is_ready():
        return fail(503, "LABELING_503", "라벨링 모델을 사용할 수 없습니다")

    # 3~4. 사진별 처리
    succeeded, failed = [], []
    for p in req.photos:
        try:
            data = s3_loader.fetch_image(BUCKET, REGION, p.objectKey)
            tags = classifier.predict_tags(data)
            succeeded.append({"clientFileId": p.clientFileId,
                              "objectKey": p.objectKey,
                              "takenAt": p.takenAt,
                              "tags": tags})
        except s3_loader.S3NotFound:
            failed.append((p, "LABELING_S3_OBJECT_NOT_FOUND", "객체를 찾을 수 없습니다"))
        except s3_loader.S3ReadFailed:
            failed.append((p, "LABELING_S3_READ_FAILED", "S3 조회에 실패했습니다"))
        except s3_loader.ImageTooLarge:
            failed.append((p, "LABELING_IMAGE_TOO_LARGE", "10MB를 초과합니다"))
        except s3_loader.InvalidImage:
            failed.append((p, "LABELING_INVALID_IMAGE", "지원하는 이미지 형식이 아닙니다"))
        except Exception:
            failed.append((p, "LABELING_MODEL_FAILED", "모델 추론에 실패했습니다"))

    failed_out = [{"clientFileId": p.clientFileId, "objectKey": p.objectKey,
                   "errorCode": c, "message": m} for p, c, m in failed]

    # 5. 성공분만 Spring 등록
    succeeded_out = []
    if succeeded:
        try:
            result = spring_client.register_results(
                SPRING_URL, INTERNAL_KEY, req.uploadTicket,
                req.uploadId, req.petId, succeeded)
        except spring_client.SpringRegisterError as e:
            return fail(502, "LABELING_502", f"결과 등록 실패: {e}")

        # 6. photoId 매핑
        id_map = {x["clientFileId"]: x["photoId"] for x in result["photos"]}
        for s in succeeded:
            succeeded_out.append({
                "clientFileId": s["clientFileId"],
                "objectKey": s["objectKey"],
                "photoId": id_map.get(s["clientFileId"]),
                "tags": s["tags"],
            })

    return {"success": True,
            "data": {"uploadId": req.uploadId,
                     "succeeded": succeeded_out,
                     "failed": failed_out},
            "message": "사진 라벨링 요청이 처리되었습니다."}