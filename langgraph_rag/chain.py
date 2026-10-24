"""
프롬프트 / LLM / 답변 생성.

이 프로젝트의 유일한 유료 API가 쓰이는 곳이다. 예산 방어를 위해 그래프 실행당 1회만 호출한다.

핵심 규약: 링크는 LLM이 쓰지 않는다.
LLM은 (병원 n) 표기만 하고, 실제 URL은 render_sources()가 원본 메타데이터에서 역참조해 붙인다.
응급 상황에서 잘못된 링크는 잘못된 문장보다 비싸다.

build_context() -> 병원 목록을 프롬프트용 문자열로 조립하는 함수 (LLM 호출 없음)
render_sources() -> 답변이 인용한 병원만 모아 출처 섹션을 만드는 함수 (LLM 호출 없음)
get_llm() -> ChatOpenAI 인스턴스를 한 번만 만들어 재사용하는 함수 (streaming 항상 켬)
get_prompt() -> 답변 생성용 ChatPromptTemplate을 만드는 함수
build_answer_chain() -> 프롬프트 -> LLM -> 문자열로 이어지는 LCEL 체인을 만드는 함수
"""

import re

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from data.geo import format_distance


ANSWER_MODEL = "gpt-5.6-luna"
MAX_TOKENS = 2048

_llms: dict[bool, ChatOpenAI] = {}

SYSTEM_PROMPT = """\
너는 반려동물 보호자에게 근처 동물병원을 안내하는 도우미다.
위급한 상황에서 읽는 사람이 많다. 짧고 분명하게 쓴다.

병원 목록은 네가 쓰지 않는다. 이름·주소·거리·링크는 화면이 [병원 목록]을 그대로 카드로 그린다.
너는 그 목록을 보고 짧은 안내 문장만 쓴다. 세 문장을 넘기지 않는다.

지켜야 할 것:
1. 병원 이름·주소·거리·전화번호·링크를 문장에 쓰지 않는다. 중복이고, 틀릴 여지만 생긴다.
2. 마크다운을 쓰지 않는다. 굵게, 목록, 제목 기호 없이 평문으로만 쓴다.
3. 전화번호와 영업시간은 절대 쓰지 않는다. 우리에게 그 데이터가 없다.
   대신 "영업 여부와 전화번호는 각 병원 링크에서 확인해 주세요"라고 안내한다.
4. [병원 목록]이 비었으면 "주어진 자료로는 근처 동물병원을 찾지 못했습니다"라고 밝히고,
   지역을 넓혀 다시 물어보도록 안내한다. 지어내지 않는다.
5. 위급 증상(호흡 곤란, 의식 저하, 심한 출혈, 경련, 중독)이 의심되면 즉시 이동하라고 덧붙인다.

좋은 예: "가까운 순으로 아래에 정리했습니다. 영업 여부와 전화번호는 각 병원 링크에서 확인해 주세요. 호흡이 힘들거나 의식이 흐리면 지체 없이 가장 가까운 곳으로 이동하세요."
"""

USER_PROMPT = """\
[병원 목록]
{context}

[질문]
{question}
"""


def build_context(vets: list[dict]) -> str:
    """
    병원 레코드 리스트를 프롬프트용 문자열로 조립한다. LLM 호출은 없다.

    전화번호는 컨텍스트에 넣지 않는다. 넣으면 프롬프트로 금지해도 새어 나올 여지가 생기고,
    애초에 이 값은 갱신이 느린 인허가 데이터라 답변에 실을 정확도가 아니다.
    """
    if not vets:
        return "(조건에 맞는 병원 없음)"

    lines = []
    for i, vet in enumerate(vets, start=1):
        address = vet.get("road_address") or vet.get("address") or "주소 미상"
        lines.append(
            f"병원 {i}: {vet.get('name', '이름 미상')}\n"
            f"  주소: {address}\n"
            f"  거리: {format_distance(vet.get('distance_m'))}"
        )

    return "\n".join(lines)


# 답변에 표기된 (병원 n)을 다시 읽어 출처로 실을 병원을 고른다.
_CITATION = re.compile(r"\(병원\s*(\d+)\)")


def render_sources(answer: str, vets: list[dict]) -> str:
    """
    답변이 인용한 병원만 모아 출처 섹션 문자열을 만든다. LLM 호출은 없다.

    URL은 원본 메타데이터에서 그대로 꺼내 붙인다. LLM이 만든 문자열을 쓰지 않는다.
    링크가 없는 병원은 없다고 밝힌다 - 빈 링크를 감추면 사용자가 클릭할 곳을 잃는다.
    """
    cited = sorted({int(n) for n in _CITATION.findall(answer)})
    if not cited:
        return ""

    lines = ["\n## 확인 링크"]
    for number in cited:
        if not 1 <= number <= len(vets):
            continue  # 목록에 없는 번호를 인용한 경우. 없는 링크를 만들어 붙이지 않는다.
        vet = vets[number - 1]
        link = vet.get("map_url")
        if link:
            lines.append(f"- 병원 {number}: {vet.get('name', '')} - {link}")
        else:
            lines.append(f"- 병원 {number}: {vet.get('name', '')} - 등록된 링크 없음")

    return "\n".join(lines) if len(lines) > 1 else ""


def get_llm(streaming: bool = True) -> ChatOpenAI:
    """
    ChatOpenAI 인스턴스를 한 번만 만들어 재사용한다.

    streaming은 항상 켠다(2026-08-20 결정). POST /api/vets 와 /api/vets/stream 이 같은 그래프
    싱글턴을 공유해서, 노드는 자기가 스트리밍 요청 안에서 도는지 알 수 없다.
    켜 두면 SSE 경로는 토큰이 잘게 흐르고, 비스트리밍 경로는 .invoke()가 내부에서 모아 돌려준다.

    캐시 키가 streaming인 이유: 인자를 받는 함수인데 캐시에 키가 없으면
    먼저 호출된 쪽의 인스턴스가 계속 돌아온다.

    temperature는 넘기지 않는다(기본값 사용).
    max_tokens는 LangChain 쪽 인자 이름이고, gpt-5 계열로는 max_completion_tokens로 번역돼 나간다.
    """
    if streaming not in _llms:
        _llms[streaming] = ChatOpenAI(
            model=ANSWER_MODEL,
            max_tokens=MAX_TOKENS,
            streaming=streaming,
        )
    return _llms[streaming]


def get_prompt() -> ChatPromptTemplate:
    """
    답변 생성용 프롬프트. history는 멀티턴에서 이전 대화를 끼워 넣는 자리다.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("user", USER_PROMPT),
        ]
    )


def build_answer_chain(streaming: bool = True) -> Runnable:
    """
    프롬프트 -> LLM -> 문자열로 이어지는 LCEL 체인.
    입력은 {"context": str, "question": str, "history": list}.
    """
    return get_prompt() | get_llm(streaming) | StrOutputParser()
