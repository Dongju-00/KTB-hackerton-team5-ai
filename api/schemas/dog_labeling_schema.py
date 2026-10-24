from pydantic import BaseModel, Field


class PhotoItem(BaseModel):
    clientFileId: str = Field(min_length=1, max_length=100)
    objectKey: str = Field(min_length=1, max_length=1024)
    takenAt: str | None = None


class LabelingRequest(BaseModel):
    uploadId: str
    uploadTicket: str = Field(min_length=1)
    petId: int = Field(gt=0)
    photos: list[PhotoItem] = Field(min_length=1, max_length=10)


class Succeeded(BaseModel):
    clientFileId: str
    objectKey: str
    photoId: int
    tags: list[str]


class Failed(BaseModel):
    clientFileId: str
    objectKey: str
    errorCode: str
    message: str