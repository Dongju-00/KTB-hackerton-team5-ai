"""
Gemini로 val 셋을 독립 재라벨링해 우리 모델을 교차 평가한다.
CLIP이 만든 라벨의 자기참조 함정을 깨는 것이 목적이다.
"""
import json
import os
import time
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import types

from model import classifier

classifier.load()

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-3.6-flash"

ALLOWED = ["costume", "playing", "water", "with_friends",
           "sleeping", "snow", "with_owner", "etc"]

SYSTEM = """당신은 반려견 사진에 태그를 붙이는 평가자입니다.
아래 목록에서 이 사진에 가장 적합한 태그 하나만 고르세요.

- costume: 개가 옷이나 코스튬을 착용
- playing: 개가 뛰거나 장난감과 놀고 있음
- water: 개가 물에서 수영하거나 물놀이 중
- with_friends: 개가 두 마리 이상 함께 있음
- sleeping: 개가 자거나 누워서 쉬고 있음
- snow: 눈 위 또는 눈 오는 배경
- with_owner: 사람이 개를 안거나 쓰다듬고 있음
- etc: 위 어디에도 명확히 해당하지 않음

여러 개에 해당하면 사진에서 가장 두드러진 것 하나를 고르세요.
개가 없는 사진이면 etc를 고르세요."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "tag": {"type": "STRING", "enum": ALLOWED},
        "alsoApplies": {
            "type": "ARRAY",
            "items": {"type": "STRING", "enum": ALLOWED},
        },
        "reason": {"type": "STRING"},
    },
    "required": ["tag", "alsoApplies", "reason"],
}

MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp"}


def ask_gemini(path: Path) -> dict:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type=MIME.get(path.suffix.lower(), "image/jpeg"),
            ),
            "이 사진의 태그는?",
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0,
        ),
    )
    return json.loads(resp.text)


EXTS = set(MIME)
paths = sorted(p for p in Path("test_photos").glob("*") if p.suffix.lower() in EXTS)

agree = multi_ok = total = 0
disagreements = []
confusion = Counter()

for p in paths:
    try:
        judge = ask_gemini(p)
        ours = classifier.predict_tags(p.read_bytes())[0]
    except Exception as e:
        print(f"{p.name[:30]:30s} 실패: {type(e).__name__}: {e}")
        continue

    total += 1
    also = set(judge.get("alsoApplies") or [])

    if judge["tag"] == ours:
        agree += 1
        mark = "O"
    elif ours in also:
        multi_ok += 1          # 부차 태그로는 맞음
        mark = "~"
    else:
        mark = "X"
        disagreements.append((p.name, judge["tag"], ours, judge["reason"]))

    confusion[(judge["tag"], ours)] += 1
    print(f"{mark} {p.name[:28]:28s} | 평가 {judge['tag']:13s} | 우리 {ours}")
    time.sleep(0.3)            # rate limit 여유

if total:
    print(f"\n완전 일치      : {agree}/{total} ({agree/total:.1%})")
    print(f"부차 태그 포함 : {(agree+multi_ok)}/{total} ({(agree+multi_ok)/total:.1%})")
    print(f"불일치         : {len(disagreements)}건")

print("\n--- 불일치 사례 ---")
for name, j, o, reason in disagreements[:20]:
    print(f"{name[:28]:28s} 평가:{j:13s} 우리:{o:13s}")
    print(f"{'':28s}   → {reason}")

print("\n--- 혼동 패턴 (평가 → 우리) ---")
for (j, o), n in sorted(confusion.items(), key=lambda x: -x[1]):
    if j != o:
        print(f"  {j:13s} → {o:13s} {n}건")