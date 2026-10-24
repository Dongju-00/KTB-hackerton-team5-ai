"""
임베딩 / 색인 / 검색.

벡터 검색이 답할 수 있는 건 병원의 '성격'(이름·지역·도로명에 드러나는 것)뿐이다.
'근처'는 좌표 연산(data/geo.py), '지금 영업중'은 링크 위임으로 처리하고 여기서는 다루지 않는다.

컬렉션은 둘이다. animal_hospitals(병원)와 grief_docs(펫로스 자료).
같은 임베딩 모델과 같은 chroma_db 디렉터리를 쓰고 컬렉션 이름만 다르다.

get_embeddings() -> 임베딩 모델을 한 번만 로드해 재사용하는 함수
get_vectorstore() -> 컬렉션 이름으로 Chroma 벡터 스토어를 가지고 오는 함수
to_documents() -> 정규화 레코드를 Document 리스트로 바꾸는 함수
index_documents() -> Document 리스트를 벡터 스토어에 적재하는 함수 (컬렉션 무관)
index_vets() -> 병원 레코드를 벡터 스토어에 적재하는 함수
index_grief_docs() -> 펫로스 자료 청크를 벡터 스토어에 적재하는 함수
all_vets() -> 색인에 든 모든 병원 레코드를 임베딩 없이 가져오는 함수
dong_to_region() -> 동 이름 -> '시도 시군구' 매핑을 만드는 함수
dong_centroid() -> 동 이름 -> 그 동 병원들의 중심 좌표 매핑을 만드는 함수
search() -> 컬렉션에서 유사 문서를 유사도와 함께 반환하는 함수
search_vets() -> 질의와 지역으로 유사 병원을 유사도와 함께 반환하는 함수
filter_by_relevance() -> 유사도가 임계값 이상인 결과만 고르는 함수
"""

from threading import RLock

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from data.config import PROJECT_ROOT
from data.grief_docs import to_document_text as grief_document_text
from data.vet_data import to_document_text


# 한국어 검색 특화 임베딩(bge-m3 계열). 적재와 검색이 같은 모델이어야 임베딩 공간이 일치한다.
MODEL_NAME = "nlpai-lab/KURE-v1"
# CWD가 아니라 프로젝트 루트 기준. 어느 위치에서 실행해도 같은 색인을 연다.
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "animal_hospitals"
GRIEF_COLLECTION_NAME = "grief_docs"
RELEVANCE_THRESHOLD = 0.3  # cosine 유사도 기준

# Chroma 메타데이터는 None을 받지 않는다. 여기 담을 키만 골라 실어 보낸다.
METADATA_KEYS = ("id", "name", "address", "road_address", "phone", "region", "status")
# 펫로스 자료 쪽. text는 page_content로 들어가므로 메타에 중복해 싣지 않는다.
# phase는 지금 필터에 쓰이지 않는다. 나중에 켤 때 재색인을 피하려고 미리 싣는다.
GRIEF_METADATA_KEYS = ("id", "title", "source", "url", "license", "phase", "section")

_embeddings = None
# 컬렉션 이름별로 따로 캐시한다. 인자를 받는 함수인데 캐시에 키가 없으면
# 먼저 호출된 컬렉션의 인스턴스가 계속 돌아온다(chain.get_llm이 streaming을 키로 둔 것과 같은 이유).
_vectorstores: dict[str, Chroma] = {}
_embeddings_lock = RLock()
_vectorstore_lock = RLock()


def get_embeddings():
    """
    HuggingFaceEmbeddings 인스턴스를 한 번만 만들어 재사용한다.

    normalize_embeddings=True로 L2 정규화한다. KURE-v1은 정규화 + cosine 사용을 전제로 학습됐다.
    """
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = HuggingFaceEmbeddings(
                    model_name=MODEL_NAME,
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embeddings


def get_vectorstore(collection_name: str = COLLECTION_NAME) -> Chroma:
    """
    로컬 chroma_db의 컬렉션을 Chroma 벡터 스토어로 감싸서 가지고 온다.

    컬렉션 이름을 키로 캐시한다. 병원과 펫로스 자료가 같은 디렉터리를 공유하되
    컬렉션으로 분리되므로, 하나만 캐시해 두면 나중에 요청한 쪽이 엉뚱한 컬렉션을 받는다.
    """
    if collection_name not in _vectorstores:
        with _vectorstore_lock:
            if collection_name not in _vectorstores:
                _vectorstores[collection_name] = Chroma(
                    collection_name=collection_name,
                    embedding_function=get_embeddings(),
                    persist_directory=str(CHROMA_PATH),
                    collection_metadata={"hnsw:space": "cosine"},
                )
    return _vectorstores[collection_name]


def to_documents(records: list[dict]) -> list[Document]:
    """
    정규화 레코드를 Document 리스트로 바꾼다.

    좌표는 거리 계산에 쓰이므로 메타데이터에 실어야 하는데 None이 섞이면 Chroma가 거부한다.
    좌표가 없는 레코드는 lat/lng 키를 아예 넣지 않고, 읽는 쪽에서 .get()으로 받는다.
    """
    documents = []
    for record in records:
        metadata = {key: record.get(key) or "" for key in METADATA_KEYS}
        if record.get("lat") is not None and record.get("lng") is not None:
            metadata["lat"] = record["lat"]
            metadata["lng"] = record["lng"]

        documents.append(Document(page_content=to_document_text(record), metadata=metadata))

    return documents


def index_documents(
    documents: list[Document],
    ids: list[str],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 500,
) -> int:
    """
    Document 리스트를 벡터 스토어에 적재하고 적재한 건수를 돌려준다. 컬렉션과 무관하다.

    같은 id로 다시 넣으면 Chroma가 덮어쓰므로 재실행이 안전하다.
    임베딩은 건수가 많아 배치로 나눠 넣는다.
    """
    vectorstore = get_vectorstore(collection_name)
    for start in range(0, len(documents), batch_size):
        chunk = slice(start, start + batch_size)
        vectorstore.add_documents(documents=documents[chunk], ids=ids[chunk])
        print(f"  적재 {min(start + batch_size, len(documents)):,} / {len(documents):,}")

    return len(documents)


def index_vets(records: list[dict], batch_size: int = 500) -> int:
    """
    병원 레코드를 벡터 스토어에 적재하고 적재한 건수를 돌려준다.

    id는 공공데이터의 관리번호를 쓴다.
    """
    documents = to_documents(records)
    ids = [doc.metadata["id"] or f"row-{i}" for i, doc in enumerate(documents)]

    return index_documents(documents, ids, COLLECTION_NAME, batch_size)


def index_grief_docs(records: list[dict], batch_size: int = 500) -> int:
    """
    펫로스 자료 청크를 grief_docs 컬렉션에 적재하고 적재한 건수를 돌려준다.

    id는 '파일명-절번호-청크번호'라 같은 원본을 다시 넣으면 덮어쓴다.
    다만 원본을 고쳐 절 수가 줄면 예전 id가 남는다 - 그때는 컬렉션을 지우고 다시 넣는다.
    """
    documents = [
        Document(
            page_content=grief_document_text(record),
            metadata={key: record.get(key) or "" for key in GRIEF_METADATA_KEYS},
        )
        for record in records
    ]
    ids = [doc.metadata["id"] or f"grief-{i}" for i, doc in enumerate(documents)]

    return index_documents(documents, ids, GRIEF_COLLECTION_NAME, batch_size)


_all_vets = None
_dong_index = None
_dong_centroid = None


def all_vets() -> list[dict]:
    """
    색인에 든 모든 병원 레코드를 메타데이터로 가져온다. 임베딩도 유사도 계산도 하지 않는다.

    좌표가 있는 질의는 '가까움'이 순수 계산이라 벡터 검색을 거칠 이유가 없다.
    오히려 유사도 상위 k건으로 먼저 좁히면 진짜 가까운 병원이 그 k건 밖으로 밀려난다.
    """
    global _all_vets

    if _all_vets is None:
        got = get_vectorstore().get(include=["metadatas"])
        _all_vets = [dict(m) for m in got["metadatas"]]
    return _all_vets


def dong_to_region() -> dict[str, str]:
    """
    동 이름 -> '시도 시군구' 매핑. 지번주소의 세 번째 토큰이 동이다.

    프론트는 '역삼동'처럼 동 단위로 보내는데 색인의 region은 '서울특별시 강남구' 형식이라
    그대로 필터에 넣으면 한 건도 걸리지 않는다.
    같은 동 이름이 여러 구에 있으면 병원이 더 많은 구를 택한다.
    """
    global _dong_index

    if _dong_index is None:
        counts: dict[str, dict[str, int]] = {}
        for record in all_vets():
            parts = (record.get("address") or "").split()
            if len(parts) < 3 or not record.get("region"):
                continue
            counts.setdefault(parts[2], {}).setdefault(record["region"], 0)
            counts[parts[2]][record["region"]] += 1
        _dong_index = {
            dong: max(regions.items(), key=lambda kv: kv[1])[0]
            for dong, regions in counts.items()
        }
    return _dong_index


def dong_centroid() -> dict[str, tuple[float, float]]:
    """
    동 이름 -> 그 동 병원 좌표의 평균. 동 단위 위치의 근사치다.

    행정 경계의 중심이 아니라 '그 동에 있는 병원들의 한가운데'라, 병원이 한쪽에 몰려 있으면
    중심도 쏠린다. 동 하나가 보통 1km 남짓이라 그 오차는 거리 정렬에서 순위를 크게 바꾸지 않는다.
    좌표가 없는 레코드는 평균에서 뺀다.
    """
    global _dong_centroid

    if _dong_centroid is None:
        points: dict[str, list[tuple[float, float]]] = {}
        for record in all_vets():
            parts = (record.get("address") or "").split()
            if len(parts) < 3 or record.get("lat") is None or record.get("lng") is None:
                continue
            points.setdefault(parts[2], []).append((record["lat"], record["lng"]))
        _dong_centroid = {
            dong: (sum(p[0] for p in ps) / len(ps), sum(p[1] for p in ps) / len(ps))
            for dong, ps in points.items()
        }
    return _dong_centroid


def search(
    query: str,
    k: int = 20,
    collection_name: str = COLLECTION_NAME,
    filter: dict | None = None,
) -> list[tuple[Document, float]]:
    """
    컬렉션에서 유사 문서를 (Document, 유사도) 튜플 리스트로 반환한다.

    락을 거는 이유: Chroma 인스턴스를 요청들이 공유한다.
    """
    with _vectorstore_lock:
        return get_vectorstore(collection_name).similarity_search_with_relevance_scores(
            query=query, k=k, filter=filter
        )


def search_vets(query: str, region: str = "", k: int = 20) -> list[tuple[Document, float]]:
    """
    질의와 지역으로 유사 병원을 (Document, 유사도) 튜플 리스트로 반환한다.

    지역은 메타데이터 필터로 건다. 거리 정렬은 여기서 하지 않는다 - 유사도와 거리는 다른 축이라,
    좌표 연산은 호출한 쪽(nodes.search_vets)이 이 결과를 받아 처리한다.
    k를 넉넉히 두는 이유도 거리 재정렬 여지를 남기기 위해서다.
    """
    return search(query, k=k, filter={"region": region} if region else None)


def filter_by_relevance(
    results: list[tuple[Document, float]],
    threshold: float = RELEVANCE_THRESHOLD,
) -> list[tuple[Document, float]]:
    """
    유사도가 임계값 미만인 결과를 걸러낸다.
    전부 걸러지면 유사도가 제일 높은 하나만 남긴다(빈손으로 돌려보내지 않기 위해).
    """
    kept = [(doc, score) for doc, score in results if score >= threshold]

    if not kept and results:
        kept = [max(results, key=lambda r: r[1])]

    return kept
