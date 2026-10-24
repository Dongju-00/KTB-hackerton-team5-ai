"""
그래프 노드.

각 노드는 VetSearchState를 받아 자기가 채운 필드만 dict로 돌려준다.

각 노드는 시작하자마자 _emit_stage()로 자기 이름을 custom 스트림에 흘린다.
generate만 예외다. 토큰이 흐르는 동안의 custom 쓰기가 노드 완료 시점까지 밀려서
"작성하고 있어요"가 답변보다 늦게 도착했다. 첫 token 자체가 작성 시작 신호라 stage를 빼기로 했다.
updates 스트림은 노드가 "끝난 뒤" 나오므로 진행 상황 표시에 쓸 수 없다 - 한 박자 늦는다.
사용자에게 보일 문구는 여기서 정하지 않는다. 노드 이름만 넘기고 문구는 api 계층이 붙인다.

반경 안에 병원이 없으면 넓히지 않고 "없음"으로 답한다(2026-08-19 결정).

resolve_location() -> 질문/입력에서 검색에 쓸 지역과 좌표를 확정하는 노드
search_vets() -> 벡터 검색 후 거리로 재정렬하고 반경으로 자르는 노드
generate() -> 답변을 생성하고 출처를 붙이는 노드 (유료 호출 1회)
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_stream_writer

from data.geo import sort_by_distance
from data.vet_data import map_search_url
from langgraph_rag import vectorstore
from langgraph_rag.chain import build_answer_chain, build_context, render_sources
from langgraph_rag.state import TOP_N, VetSearchState


SEARCH_K = 50      # 거리 재정렬 여지를 남기려고 넉넉히 뽑는다


def _emit_stage(name: str) -> None:
    """
    노드 시작을 custom 스트림으로 알린다.
    스트리밍이 아닌 경로(invoke)에서는 writer가 아무 일도 하지 않는다.
    """
    get_stream_writer()({"stage": name})


def _with_links(records: list[dict]) -> list[dict]:
    """
    각 병원에 네이버 지도 검색 URL을 붙인다. 외부 호출이 없어 노드를 따로 두지 않는다.
    """
    return [{**r, "map_url": map_search_url(r)} for r in records]


def resolve_location(state: VetSearchState) -> dict:
    """
    검색에 쓸 지역을 확정한다.

    프론트는 '역삼동'처럼 동 단위로 보내는데 색인의 region은 '서울특별시 강남구' 형식이다.
    동으로 들어오면 구로 바꾸고, 기준 좌표도 그 동의 중심으로 옮긴다.

    sabu: 매핑 실패 — 서울에 없는 동('판교동')은 변환되지 않아 그대로 필터에 들어가고 0건이 된다.
          사용자는 "못 찾았습니다"만 본다. 데이터에 없는 지역인 것과 근처에 병원이 없는 것은
          같은 말인가? 다르다면 그 차이는 상태의 어느 필드에 담겨야 하나?
    """
    _emit_stage("resolve_location")
    region = (state.get("region") or "").strip()

    if region and " " not in region:
        dong = region
        region = vectorstore.dong_to_region().get(dong, dong)
        center = vectorstore.dong_centroid().get(dong)
        if center:
            # 동을 입력한 요청은 그 동에서 가까운 순으로 답해야 한다.
            # 기본 좌표(강남역)를 그대로 두면 노원구를 물어도 강남역 기준으로 반경이 잘린다.
            return {"region": region, "lat": center[0], "lng": center[1]}

    return {"region": region}


def search_vets(state: VetSearchState) -> dict:
    """
    후보 병원을 뽑는다. 좌표가 있으면 거리 계산만, 없으면 벡터 검색을 쓴다.

    좌표가 있을 때 벡터 검색을 거치지 않는 이유: '근처 동물병원 찾기'는 버튼이 보내는 고정 문장이라
    어떤 병원과도 특별히 유사하지 않다. 유사도 상위 k건으로 먼저 좁히면 사실상 무작위 k건이 되고,
    그 안에서 거리를 재봐야 진짜 가까운 병원은 이미 빠져 있다. 실측에서 상위 5곳 중 4곳을 놓쳤다.
    """
    _emit_stage("search_vets")

    region = state.get("region", "")
    lat, lng = state.get("lat"), state.get("lng")

    if lat is not None and lng is not None:
        records = vectorstore.all_vets()
        if region:
            records = [r for r in records if r.get("region") == region]
        ranked = sort_by_distance(records, lat, lng)
        radius = state.get("radius_m", 0)
        within = [r for r in ranked if r["distance_m"] is not None and r["distance_m"] <= radius]
        return {"vets": _with_links(within[:TOP_N])}

    results = vectorstore.search_vets(query=state["question"], region=region, k=SEARCH_K)
    kept = vectorstore.filter_by_relevance(results)
    records = [{**doc.metadata, "similarity": score} for doc, score in kept]

    return {"vets": _with_links(records[:TOP_N])}





def generate(state: VetSearchState) -> dict:
    """
    답변을 생성하고 출처(확인 링크)를 코드가 붙인다. 유료 호출은 실행당 1회로 막는다.
    """
    if state.get("paid_call_count", 0) >= 1:
        raise RuntimeError("이번 실행의 유료 호출 한도(1회)를 이미 썼습니다.")

    vets = state.get("vets", [])
    answer = build_answer_chain().invoke(
        {
            "context": build_context(vets),
            "question": state["question"],
            "history": state.get("history", []),
        }
    )

    return {
        "answer": answer,
        "sources": render_sources(answer, vets),
        "paid_call_count": state.get("paid_call_count", 0) + 1,
        "history": [HumanMessage(state["question"]), AIMessage(answer)],
    }
