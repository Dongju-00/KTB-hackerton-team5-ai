from pathlib import Path
from model import classifier

classifier.load()

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
folder = Path("test_photos")

paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in EXTS)
print(f"\n{len(paths)}장 처리\n" + "-" * 50)

from collections import Counter
counts = Counter()

for p in paths:
    try:
        tags = classifier.predict_tags(p.read_bytes())
        counts[tags[0]] += 1
        print(f"{p.name[:35]:35s} → {tags[0]}")
    except Exception as e:
        counts["error"] += 1
        print(f"{p.name[:35]:35s} → 실패: {type(e).__name__}")

print("-" * 50)
print("태그 분포:", dict(counts))
etc = counts.get("etc", 0)
if paths:
    print(f"etc 비율: {etc}/{len(paths)} ({etc/len(paths):.1%})")