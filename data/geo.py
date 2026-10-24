"""
좌표 정규화와 거리 계산.

거리는 벡터 검색이 답할 수 없는 값이라 코드가 직접 계산한다.

haversine() -> 두 좌표 사이의 지표면 거리를 미터로 돌려주는 함수
sort_by_distance() -> 기준 좌표에서 가까운 순으로 레코드를 정렬하는 함수
format_distance() -> 미터 값을 사람이 읽는 문자열로 바꾸는 함수
"""

from math import asin, cos, radians, sin, sqrt


EARTH_RADIUS_M = 6_371_008.8  # 평균 지구 반지름


# 입력은 WGS84 위경도여야 한다. 공공데이터 원본은 EPSG:5174 투영좌표라
# data/vet_data.py에서 변환해 넣는다. 네이버 mapx/mapy도 문서상 WGS84라 같은 기준이다.
def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    두 위경도 사이의 대권 거리를 미터 단위로 돌려준다.
    """
    lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2

    return 2 * EARTH_RADIUS_M * asin(sqrt(a))


def sort_by_distance(records: list[dict], lat: float, lng: float) -> list[dict]:
    """
    기준 좌표에서 가까운 순으로 정렬하고 각 레코드에 distance_m을 채워 돌려준다.

    좌표가 없는 레코드는 거리를 계산할 수 없으므로 distance_m=None으로 두고 맨 뒤로 보낸다.
    (좌표 없음을 0m로 두면 '가장 가까운 병원'으로 올라온다.)
    """
    scored = []
    for record in records:
        if record.get("lat") is None or record.get("lng") is None:
            distance = None
        else:
            distance = haversine(lat, lng, record["lat"], record["lng"])
        scored.append({**record, "distance_m": distance})

    return sorted(scored, key=lambda r: (r["distance_m"] is None, r["distance_m"] or 0))


def format_distance(distance_m: float | None) -> str:
    """
    미터 값을 사람이 읽는 문자열로 바꾼다. 없으면 '거리 미상'.
    """
    if distance_m is None:
        return "거리 미상"
    if distance_m < 1000:
        return f"{round(distance_m):,}m"
    return f"{distance_m / 1000:.1f}km"
