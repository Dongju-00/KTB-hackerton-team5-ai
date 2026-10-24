"""
공공데이터 동물병원 인허가 데이터를 정규화 레코드로 바꾸는 수집 계층.

원본은 LOCALDATA(지방행정 인허가데이터)의 동물병원 CSV다. 컬럼명이 한글이고
지자체마다 빈 값이 섞여 있어, 이 모듈에서 한 번 정규화한 뒤로는 원본 컬럼명을 쓰지 않는다.

load_vets()는 상세영업상태명이 '정상'인 병원만 싣는다. 폐업·휴업·말소는 로딩 시점에 걸러진다.

정규화 레코드(VetRecord):
    id / name / address / road_address / phone / lat / lng / region / status

load_vets() -> CSV를 읽어 정규화 레코드 리스트로 돌려주는 함수
normalize_row() -> 원본 한 행을 정규화 레코드로 바꾸는 함수
extract_region() -> 주소 문자열에서 '시도 시군구'를 뽑는 함수
to_document_text() -> 레코드 한 건을 임베딩할 평문 한 덩이로 펴는 함수
"""

import csv
import sys
from urllib.parse import quote
from pathlib import Path

from pyproj import Transformer


DATA_DIR = Path(__file__).resolve().parent.parent / "data_raw"
DEFAULT_CSV = DATA_DIR / "animal_hospital.csv"

# LOCALDATA CSV는 대개 CP949로 내려온다. UTF-8로 받은 경우도 있어 순서대로 시도한다.
ENCODINGS = ("cp949", "utf-8-sig", "utf-8")

# LOCALDATA 좌표정보(X)/(Y)의 좌표계. 보정계수 없는 Bessel 중부원점TM.
# 미터 단위 투영좌표라 위경도가 아니다. 그대로 haversine에 넣으면 예외 없이 엉뚱한 거리가 나온다.
SOURCE_CRS = "EPSG:5174"
WGS84 = "EPSG:4326"

_transformer = None


def get_transformer() -> Transformer:
    """
    좌표 변환기를 한 번만 만들어 재사용한다. 레코드마다 새로 만들면 느리다.
    always_xy=True로 입출력 순서를 (경도, 위도)로 고정한다 - 이 축 순서가 뒤집히는 것이
    좌표 변환에서 가장 흔한 조용한 버그다.
    """
    global _transformer

    if _transformer is None:
        _transformer = Transformer.from_crs(SOURCE_CRS, WGS84, always_xy=True)
    return _transformer


# 원본 한글 컬럼 -> 정규화 키
COLUMN_MAP = {
    "관리번호": "id",
    "사업장명": "name",
    "지번주소": "address",
    "도로명주소": "road_address",
    "전화번호": "phone",
    "상세영업상태명": "status",
    "좌표정보(X)": "raw_x",
    "좌표정보(Y)": "raw_y",
}


def extract_region(address: str) -> str:
    """
    '서울특별시 강남구 역삼동 123-4' -> '서울특별시 강남구'
    벡터 검색의 메타데이터 필터로 쓸 지역 키다. 주소가 비면 빈 문자열.
    """
    if not address:
        return ""
    parts = address.split()
    return " ".join(parts[:2])


def normalize_row(row: dict) -> dict:
    """
    원본 CSV 한 행을 정규화 레코드로 바꾼다.

    좌표는 원본 컬럼을 그대로 raw_x / raw_y로 들고 있다가 lat / lng에 옮겨 담는다.
    숫자로 못 바꾸는 값(빈 문자열 등)은 None으로 둔다 - 0.0으로 채우면 아프리카 앞바다에 병원이 생긴다.
    """
    record = {}
    for source, key in COLUMN_MAP.items():
        record[key] = (row.get(source) or "").strip()

    def as_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # 원본은 EPSG:5174 투영좌표다. WGS84 위경도로 바꿔야 haversine이 의미를 갖는다.
    # sabu: 변환 실패 — pyproj는 변환할 수 없는 입력에 예외 대신 inf를 돌려준다.
    #       inf가 lat에 담기면 sort_by_distance는 그걸 '좌표 없음'으로 볼까, 유효한 값으로 볼까?
    x = as_float(record.pop("raw_x"))
    y = as_float(record.pop("raw_y"))

    if x is None or y is None:
        record["lng"] = record["lat"] = None
    else:
        record["lng"], record["lat"] = get_transformer().transform(x, y)

    record["region"] = extract_region(record["address"])

    return record


OPEN_STATUS = "정상"  # 상세영업상태명. 나머지는 폐업 / 휴업 / 말소.


def load_vets(csv_path: Path | str = DEFAULT_CSV) -> list[dict]:
    """
    LOCALDATA 동물병원 CSV를 읽어 정규화 레코드 리스트로 돌려준다.

    영업 중이 아닌 병원은 여기서 걸러낸다. 색인 시점이 아니라 로딩 시점에 거르는 이유는,
    이 함수를 거친 뒤로는 '문 닫은 병원이 섞여 있을 가능성'을 아무도 신경 쓰지 않아도 되게 하기 위해서다.

    CSV는 https://www.localdata.go.kr 에서 '동물병원'으로 내려받아 data_raw/ 에 둔다.
    파일이 없으면 어디에 무엇을 두어야 하는지 알려주고 멈춘다.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} 가 없습니다. LOCALDATA에서 동물병원 CSV를 내려받아 이 경로에 두세요."
        )

    last_error = None
    for encoding in ENCODINGS:
        try:
            with csv_path.open(encoding=encoding, newline="") as f:
                records = [normalize_row(row) for row in csv.DictReader(f)]
                return [r for r in records if r["status"] == OPEN_STATUS]
        except UnicodeDecodeError as e:
            last_error = e

    raise UnicodeDecodeError(*last_error.args)


NAVER_MAP_SEARCH = "https://map.naver.com/p/search/"


def map_search_url(record: dict) -> str:
    """
    병원의 네이버 지도 검색 URL을 만든다. 외부 호출 없이 문자열 조립만 한다.

    place id로 바로 가는 URL이 이상적이지만 지역검색 API 응답에 id가 없다.
    검색 URL은 항상 유효하고, 그 페이지에 영업시간과 전화번호가 있다 - 우리가 답하지 않기로 한 두 값이다.
    검색어에 지역을 붙여 동명 병원이 섞이는 것을 줄인다.
    """
    query = " ".join(x for x in (record.get("region"), record.get("name")) if x)
    return NAVER_MAP_SEARCH + quote(query) if query else ""


def to_document_text(record: dict) -> str:
    """
    레코드 한 건을 임베딩할 평문 한 덩이로 편다.

    거리·영업시간은 여기 넣지 않는다. 벡터 검색이 답할 수 있는 건 병원의 '성격'뿐이고
    거리는 좌표 연산, 영업 여부는 링크로 위임하기 때문이다.
    """
    parts = [record.get("name", ""), record.get("region", "")]
    if record.get("road_address"):
        parts.append(record["road_address"])
    elif record.get("address"):
        parts.append(record["address"])
    return " · ".join(p for p in parts if p)


if __name__ == "__main__":
    # 수집층 단독 검증: 최종 답변이 아니라 이 계층의 값을 직접 센다.
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    vets = load_vets(path)

    print(f"총 레코드          : {len(vets):,}")
    print(f"좌표 없음          : {sum(1 for v in vets if v['lat'] is None):,}")
    print(f"전화 빈 값         : {sum(1 for v in vets if not v['phone']):,}")

    statuses = {}
    for v in vets:
        statuses[v["status"]] = statuses.get(v["status"], 0) + 1
    print("영업상태명 분포     :", statuses)

    print("\n첫 레코드:")
    for k, v in vets[0].items():
        print(f"  {k:14} {v!r}")
    print("\n평문화 예시:", to_document_text(vets[0]))
