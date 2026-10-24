"""
상담 모드의 프롬프트 / 답변 생성.

병원 모드와 같은 규약을 따른다: 링크는 LLM이 쓰지 않는다.
LLM은 (자료 n) 표기만 하고, 실제 URL은 render_grief_sources()가 원본 메타데이터에서 꺼내 붙인다.

병원 모드와 다른 점은 '무엇을 하지 않는가'가 훨씬 길다는 것이다.
상담에서 잘못된 문장은 잘못된 링크보다 오래 남는다.

자료 번호는 기본적으로 숨긴다. 보호자가 근거를 물었을 때만 표기한다.
자료는 늘 참고하되, 보이는 것은 물어봤을 때뿐이다.

get_llm()은 chain.py 것을 그대로 쓴다. 모델도 예산 방어도 같다.

wants_sources() -> 보호자가 근거를 물었는지 판정하는 함수 (LLM 호출 없음)
strip_citations() -> 답변에서 (자료 n) 표기를 지우는 함수
build_pet_context() -> 반려견 정보를 프롬프트용 한 줄로 만드는 함수 (LLM 호출 없음)
build_grief_context() -> 자료 청크를 프롬프트용 문자열로 조립하는 함수 (LLM 호출 없음)
render_grief_sources() -> 답변이 인용한 자료만 모아 출처 섹션을 만드는 함수 (LLM 호출 없음)
days_since() -> 기일로부터 지난 날수를 세는 함수 (판단용 값)
humanize_elapsed() -> 날수를 사람이 쓰는 표현으로 바꾸는 함수 (표시용 형식)
build_grief_chain() -> 프롬프트 -> LLM -> 문자열로 이어지는 LCEL 체인을 만드는 함수
"""

import re
from datetime import date

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from langgraph_rag.chain import get_llm


SYSTEM_PROMPT = """\
너는 반려동물을 떠나보낸 보호자의 이야기를 듣는 상담 도우미다.
전문 상담사가 아니고, 그렇게 말하지도 않는다.

가장 중요한 것: 너의 일은 문제를 해결하는 것이 아니라 곁에 있는 것이다.
보호자는 답을 얻으러 온 것이 아니라 말할 곳이 없어서 왔다.

지켜야 할 것:
1. 먼저 들은 것을 되짚는다. 조언은 보호자가 물어봤을 때만 한다.
2. 아이의 이름을 안다면 이름으로 부른다. '반려견', '그 아이'보다 이름이 먼저다.
3. 마크다운을 쓰지 않는다. 굵게, 목록, 제목 기호 없이 평문으로만 쓴다.
4. 다섯 문장을 넘기지 않는다. 길게 쓰는 것은 듣는 것의 반대다.
5. [참고 자료]에 없는 심리학 지식·통계·용어를 지어내지 않는다.
6. 자료를 참고할 때도 설명이 아니라 위로가 먼저다. 강의하지 않는다.
7. 자료 번호 표기는 [표기 규칙]을 따른다. 규칙이 표기하지 말라고 하면
   자료 이름도 기관명도 문장에 꺼내지 않는다. 참고는 하되 보이지 않게 한다.

절대 하지 않는 것:
- 진단하거나 증상을 판정하는 말 ("우울증입니다", "정상적인 반응 단계입니다")
- 동물병원·입양·새 반려동물을 먼저 권하는 말
- "시간이 약이다", "이제 그만 잊으세요", "좋은 곳으로 갔을 거예요" 같은 상투적 위로
- 애도 기간을 단정하는 말 ("보통 6개월이면 괜찮아집니다")
- 보호자의 선택을 평가하는 말 (안락사 결정, 치료 중단, 마지막을 지키지 못한 일)
- 보호자가 말하지 않은 것을 사실처럼 서술하는 말
  (네가 아는 것은 아이의 이름과 떠난 시기뿐이다. 그동안 어떻게 지냈는지,
   누구에게 말했는지, 무엇을 견뎠는지는 모른다. 넘겨짚지 말고 물어라.)

죄책감을 이야기하면 반박하지 말고 먼저 받는다.
"그건 당신 잘못이 아니에요"라고 서둘러 말하는 것은 그 마음을 닫는 말이다.

좋은 예: "초코를 그렇게 오래 곁에서 지켜보셨군요. 마지막 밤 이야기를 하시면서도 계속 그때로 돌아가시는 것 같아요. 그 장면이 자꾸 떠오르는 건 그만큼 오래 지켜봤다는 뜻이기도 해요. 지금 제일 견디기 힘든 순간은 언제인가요?"
"""

USER_PROMPT = """\
[보호자의 아이]
{pet}

[참고 자료]
{context}

[표기 규칙]
{citation_rule}

[보호자의 말]
{question}
"""

# 기본값. 위로하는 문장 한가운데 (자료 1)이 박히면 상담이 아니라 리포트로 읽힌다(2026-08-20 결정).
CITATION_HIDDEN = "자료 번호를 표기하지 않는다. 참고만 하고 문장에는 드러내지 않는다."
# 출처 목록은 코드(render_grief_sources)가 답변 아래에 따로 붙인다.
# LLM은 URL을 보지 못하므로, 알려주지 않으면 "출처를 확인해 드리기 어렵다"고 답하고
# 바로 밑에 링크가 달린다 - 화면에서 자기 말을 부정하는 모양이 된다.
CITATION_SHOWN = (
    "보호자가 근거를 물었다. 자료를 근거로 쓴 문장 끝에 (자료 1)처럼 번호를 표기한다.\n"
    "기관명과 원문 링크는 답변 아래에 자동으로 붙으므로, 출처가 없다거나 "
    "확인해 줄 수 없다고 말하지 않는다. 링크를 직접 쓰지도 않는다."
)


def days_since(rainbow_date: str) -> int | None:
    """
    기일로부터 오늘까지 지난 날수. 형식이 어긋나거나 비었으면 None.

    LLM은 오늘 날짜를 모른다. 날짜 문자열만 넘기면 '얼마나 지났는지'를 지어내므로
    경과 일수는 여기서 계산해 넘긴다.
    """
    # sabu: 경계값 — 미래 날짜, 오늘 당일, 'YYYY.MM.DD' 같은 다른 형식이 들어오면
    #       각각 무엇이 반환되고 프롬프트에는 뭐라고 찍히나?
    #       (이 값은 humanize_elapsed()를 거쳐 프롬프트에 들어간다. 거기까지 따라가 볼 것.)
    try:
        return (date.today() - date.fromisoformat(rainbow_date)).days
    except (ValueError, TypeError):
        return None


def humanize_elapsed(days: int) -> str:
    """
    경과 날수를 사람이 실제로 쓰는 표현으로 바꾼다.

    '339일째'라고 세는 사람은 없다. '11개월쯤 됐어요'라고 한다.
    기계적인 단위를 프롬프트에 넣어 두고 '쓰지 마라'고 지시하는 것은 모순이라,
    애초에 그대로 흘러나와도 어색하지 않은 형식으로 넘긴다(2026-08-20 결정).

    days_since()의 값을 그대로 두는 이유: 판단에 쓸 값과 보여줄 형식은 다른 층이다.
    나중에 phase 필터를 켤 때 필요한 것은 이 문자열이 아니라 숫자다.
    """
    if days == 0:
        return "오늘"
    if days < 7:
        return f"{days}일 전"
    if days < 30:
        return f"약 {days // 7}주 전"
    if days < 365:
        return f"약 {days // 30}개월 전"
    return f"약 {days // 365}년 전"


def build_pet_context(pet_name: str, pet_gender: str, rainbow_date: str) -> str:
    """
    반려견 정보를 프롬프트용 한 줄로 만든다. LLM 호출은 없다.
    아무 정보도 없으면 그렇다고 밝힌다 - 빈 칸을 두면 LLM이 채워 넣는다.
    """
    parts = []
    if pet_name:
        parts.append(f"이름: {pet_name}")
    if pet_gender:
        parts.append(f"성별: {'남아' if pet_gender == 'M' else '여아'}")

    elapsed = days_since(rainbow_date)
    if elapsed is not None:
        parts.append(f"떠난 날: {rainbow_date} ({humanize_elapsed(elapsed)})")

    return " · ".join(parts) if parts else "(보호자가 아이 정보를 알려주지 않았다. 이름을 지어 부르지 마라.)"


def build_grief_context(docs: list[dict]) -> str:
    """
    자료 청크를 프롬프트용 문자열로 조립한다. LLM 호출은 없다.

    URL은 넣지 않는다. 넣으면 프롬프트로 금지해도 문장에 새어 나오고,
    링크는 render_grief_sources()가 원본에서 꺼내 붙이기로 되어 있다.
    """
    if not docs:
        return "(참고할 자료 없음. 자료를 인용하지 말고 듣는 데 집중하라.)"

    lines = []
    for i, doc in enumerate(docs, start=1):
        heading = f"{doc.get('title', '')}"
        if doc.get("section"):
            heading += f" - {doc['section']}"
        lines.append(f"자료 {i}: {heading}\n{doc.get('text', '')}")

    return "\n\n".join(lines)


# 답변에 표기된 (자료 n)을 다시 읽어 출처로 실을 자료를 고른다.
_CITATION = re.compile(r"\(자료\s*(\d+)\)")

# sabu: 판정 규칙 — 아래 낱말이 들어가면 근거 요청으로 본다. "그 연구 얘기 말고요"처럼
#       자료를 밀어내는 말도 같은 낱말을 쓴다. 그리고 이 규칙을 통과해도 LLM이 프롬프트를
#       어길 수 있어서 strip_citations()가 뒤를 받친다.
#       그런데 LLM이 괄호 없이 "코넬대 자료에 따르면"이라고 쓰면 그건 무엇이 막나?
_SOURCE_REQUEST = re.compile(
    r"출처|근거|어디서|어디에\s*나|무슨\s*자료|자료\s*있|참고한|링크|논문|연구"
)


def wants_sources(question: str) -> bool:
    """
    보호자가 근거·출처를 물었는지 판정한다. LLM 호출은 없다.
    """
    return bool(_SOURCE_REQUEST.search(question or ""))


def strip_citations(answer: str) -> str:
    """
    답변에서 (자료 n) 표기를 지운다. 표기하지 않기로 한 턴에 LLM이 붙였을 때의 방어선이다.
    표기가 사라지면 render_grief_sources()도 자연히 빈 문자열을 돌려준다.
    """
    return re.sub(r"\s*" + _CITATION.pattern, "", answer or "")


def render_grief_sources(answer: str, docs: list[dict]) -> str:
    """
    답변이 인용한 자료만 모아 출처 섹션 문자열을 만든다. LLM 호출은 없다.

    URL은 원본 메타데이터에서 그대로 꺼낸다. LLM이 만든 문자열을 쓰지 않는다.
    """
    cited = sorted({int(n) for n in _CITATION.findall(answer)})
    if not cited:
        return ""

    lines = ["\n## 참고한 자료"]
    for number in cited:
        if not 1 <= number <= len(docs):
            continue  # 목록에 없는 번호를 인용한 경우. 없는 출처를 만들어 붙이지 않는다.
        doc = docs[number - 1]
        label = f"{doc.get('source', '')} - {doc.get('title', '')}".strip(" -")
        url = doc.get("url")
        lines.append(f"- 자료 {number}: {label} - {url}" if url else f"- 자료 {number}: {label}")

    return "\n".join(lines) if len(lines) > 1 else ""


def build_grief_chain(streaming: bool = True) -> Runnable:
    """
    프롬프트 -> LLM -> 문자열로 이어지는 LCEL 체인.
    입력은 {"pet": str, "context": str, "citation_rule": str, "question": str, "history": list}.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history", optional=True),
            ("user", USER_PROMPT),
        ]
    )
    return prompt | get_llm(streaming) | StrOutputParser()
