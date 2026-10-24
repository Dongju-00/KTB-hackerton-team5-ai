import io
import os
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_REPO_ID = os.getenv("HF_MODEL_REPO", "Dongju-00/pawprint-dog-tagger")
HF_FILENAME = os.getenv("HF_MODEL_FILENAME", "dog_model.pt")
LOCAL_CACHE = Path(os.getenv("MODEL_CACHE_DIR", "model/weights"))

ALLOWED_TAGS = {"costume", "playing", "water", "etc",
                "with_friends", "sleeping", "snow", "with_owner"}

_model = None
_tf = None
_classes: list[str] = []


def resolve_model_path() -> Path:
    """Hugging Face Hub에서 가중치 경로를 얻는다."""
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=os.getenv("HF_TOKEN") or None,
        cache_dir=os.getenv("HF_CACHE_DIR") or None,
    )
    return Path(downloaded)


def load(weights_path: str | Path | None = None):
    global _model, _tf, _classes
    torch.set_num_threads(2)

    path = Path(weights_path) if weights_path else resolve_model_path()
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    # 문서 5절: "Validate required checkpoint keys"
    required = ("arch", "classes", "img_size", "state_dict")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise ValueError(f"checkpoint 필수 키 누락: {missing}")

    _classes = ckpt["classes"]
    size = ckpt["img_size"]

    _model = timm.create_model(ckpt["arch"], pretrained=False, num_classes=len(_classes))
    _model.load_state_dict(ckpt["state_dict"])
    _model.to(DEVICE).eval()

    _tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print(f"[labeling] {ckpt['arch']} loaded, {len(_classes)} classes, device={DEVICE}")


def is_ready() -> bool:
    return _model is not None and _tf is not None and bool(_classes)

@torch.no_grad()
def predict_tags(image_bytes: bytes, min_conf: float = 0.35) -> list[str]:
    """명세 8.1.2 — 중복 없는 태그 1~8개 반환"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    probs = F.softmax(_model(x), dim=1)[0]

    idx = probs.argmax().item()
    score = probs[idx].item()
    label = _classes[idx] if score >= min_conf else "etc"

    if label not in ALLOWED_TAGS:
        label = "etc"
    return [label]