"""
펫로스 자료를 정규화 레코드로 바꾸는 수집 계층.

원본은 data_raw/grief/*.md 다. 파일마다 앞머리에 출처를 적고, 본문은 마크다운 소제목으로 나눈다.

    ---
    title: 반려동물 상실 후 흔한 애도 반응
    source: 한국심리학회
    url: https://example.org/문서
    license: CC BY 4.0
    phase: acute
    ---

    ## 부정과 충격
    본문...

title/source/url/phase 중 하나라도 없는 파일은 적재하지 않고 예외를 낸다.
병원 모드에서 링크를 LLM이 만들지 않기로 한 것과 같은 규약이다 - 근거를 댈 수 없는 문장은
상담에서도 싣지 않는다. 색인 시점에 막지 않으면 답변 시점에는 막을 방법이 없다.

phase는 '사별 후 어느 시기에 필요한 글인가'를 나눈다(acute=급성 애도 / long=장기화된 애도).
지금은 메타데이터로 싣기만 하고 검색에서 거르지 않는다 - 나중에 필터를 켤 때
전체 재색인을 다시 하지 않기 위해 미리 붙여 두는 것이다(2026-08-20 결정).

정규화 레코드(GriefDoc):
    id / title / source / url / license / phase / section / text

load_grief_docs() -> data_raw/grief 의 문서를 정규화 레코드 리스트로 돌려주는 함수
parse_frontmatter() -> 파일 앞머리의 메타와 본문을 나누는 함수
split_sections() -> 마크다운 본문을 (소제목, 문단들)로 나누는 함수
chunk_section() -> 한 절을 임베딩 크기에 맞게 여러 덩이로 쪼개는 함수
to_document_text() -> 레코드 한 건을 임베딩할 평문 한 덩이로 펴는 함수
"""

import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data_raw"
GRIEF_DIR = DATA_DIR / "grief"

# 청크 목표 길이(문자). KURE-v1의 입력 한도가 아니라 '검색 단위'를 정하는 값이다.
# 너무 길면 한 청크에 여러 주제가 섞여 유사도가 뭉개지고, 너무 짧으면 문맥이 끊긴다.
CHUNK_CHARS = 600
# 이보다 짧은 청크는 앞 청크에 붙인다. 한 문장짜리 조각이 따로 검색되는 것을 막는다.
MIN_CHUNK_CHARS = 120

REQUIRED_META = ("title", "source", "url", "phase")

# sabu: 두 값으로 충분한가 — 애도 단계와 무관하게 언제 읽어도 맞는 글(예: '주변 사람에게
#       어떻게 말할까')은 acute와 long 중 무엇으로 붙나? 지금은 글쓴이가 하나를 골라야 한다.
#       그 선택이 나중에 필터를 켜는 순간 무슨 일을 하는지 짚어 봐라.
#       세 번째 값이 필요하다는 결론이 나오면, 그때 바꿔야 하는 곳은 여기 하나뿐인가?
PHASES = ("acute", "long")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    파일 앞머리의 `---` 블록을 메타 dict로, 나머지를 본문으로 나눈다.

    YAML 파서를 쓰지 않는다. 여기서 필요한 건 `키: 값` 한 줄짜리뿐이고,
    의존성을 하나 더 들이는 값보다 이 형식을 강제하는 값이 크다.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    meta: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1:])
        key, _, value = line.partition(":")
        if value:
            meta[key.strip()] = value.strip()

    # 닫는 --- 가 없는 파일. 메타로 읽은 부분이 본문일 수도 있어 통째로 본문 취급한다.
    return {}, text


def split_sections(body: str) -> list[tuple[str, str]]:
    """
    마크다운 본문을 (소제목, 그 절의 본문) 리스트로 나눈다.
    첫 소제목 앞의 글은 소제목 ""으로 담는다.
    """
    sections: list[tuple[str, list[str]]] = [("", [])]

    for line in body.splitlines():
        if line.lstrip().startswith("#"):
            sections.append((line.lstrip("#").strip(), []))
        else:
            sections[-1][1].append(line)

    return [
        (heading, "\n".join(lines).strip())
        for heading, lines in sections
        if "".join(lines).strip()
    ]


def chunk_section(text: str) -> list[str]:
    """
    한 절을 CHUNK_CHARS에 맞춰 문단 단위로 쪼갠다.

    문단 경계로만 자른다. 문장 중간에서 끊으면 그 청크만 읽는 LLM이 앞뒤를 잃는다.
    한 문단이 CHUNK_CHARS보다 길면 자르지 않고 그대로 둔다 - 억지로 자르는 것보다 낫다.
    """
    chunks: list[str] = []
    current = ""

    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > CHUNK_CHARS:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        chunks.append(current)

    # 마지막 조각이 너무 짧으면 앞에 붙인다.
    if len(chunks) > 1 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] = f"{chunks[-2]}\n\n{chunks.pop()}"

    return chunks


def load_grief_docs(directory: Path | str = GRIEF_DIR) -> list[dict]:
    """
    data_raw/grief 의 마크다운을 읽어 정규화 레코드 리스트로 돌려준다.

    출처 메타가 빠진 파일이 하나라도 있으면 예외를 낸다. 그 파일만 건너뛰면
    색인에는 조용히 근거 없는 청크가 섞이고, 나중에 답변에서 그것을 구별할 수 없다.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"{directory} 가 없습니다. 펫로스 자료를 이 폴더에 마크다운으로 넣어 주세요."
        )

    records: list[dict] = []
    for path in sorted(directory.glob("*.md")):
        meta, body = parse_frontmatter(path.read_text(encoding="utf-8"))

        missing = [key for key in REQUIRED_META if not meta.get(key)]
        if missing:
            raise ValueError(
                f"{path.name}: 메타 누락 ({', '.join(missing)}). "
                "근거를 댈 수 없는 자료는 색인하지 않습니다."
            )

        if meta["phase"] not in PHASES:
            raise ValueError(
                f"{path.name}: phase 값이 잘못됐습니다 ({meta['phase']!r}). "
                f"허용: {', '.join(PHASES)}"
            )

        for section_no, (heading, section_text) in enumerate(split_sections(body)):
            for chunk_no, chunk in enumerate(chunk_section(section_text)):
                records.append(
                    {
                        "id": f"{path.stem}-{section_no}-{chunk_no}",
                        "title": meta["title"],
                        "source": meta["source"],
                        "url": meta["url"],
                        "license": meta.get("license", ""),
                        "phase": meta["phase"],
                        "section": heading,
                        "text": chunk,
                    }
                )

    return records


def to_document_text(record: dict) -> str:
    """
    레코드 한 건을 임베딩할 평문 한 덩이로 편다.

    제목과 소제목을 본문 앞에 붙인다. 청크 하나만 떼어 놓으면 무슨 주제의 글인지가 사라지는데,
    그 문맥이 유사도에 실려야 '안락사 죄책감' 같은 질문이 맞는 절을 찾는다.
    출처 기관명은 넣지 않는다 - 검색어에 기관명이 나올 일이 없고, 유사도만 흐린다.
    """
    parts = [record.get("title", ""), record.get("section", ""), record.get("text", "")]
    return "\n".join(p for p in parts if p)


if __name__ == "__main__":
    # 수집층 단독 검증: 색인하기 전에 이 계층의 값을 직접 센다.
    path = sys.argv[1] if len(sys.argv) > 1 else GRIEF_DIR
    docs = load_grief_docs(path)

    print(f"총 청크            : {len(docs):,}")
    print(f"원본 파일          : {len({d['id'].rsplit('-', 2)[0] for d in docs}):,}")
    print(f"소제목 없는 청크   : {sum(1 for d in docs if not d['section']):,}")
    print(f"라이선스 미기재    : {sum(1 for d in docs if not d['license']):,}")
    for phase in PHASES:
        print(f"phase={phase:<6}     : {sum(1 for d in docs if d['phase'] == phase):,}")

    lengths = sorted(len(d["text"]) for d in docs)
    if lengths:
        print(f"청크 길이(최소/중앙/최대): {lengths[0]} / {lengths[len(lengths) // 2]} / {lengths[-1]}")
        print(f"CHUNK_CHARS 초과   : {sum(1 for n in lengths if n > CHUNK_CHARS):,}")

    print("\n--- 첫 청크 ---")
    if docs:
        print(to_document_text(docs[0])[:400])
