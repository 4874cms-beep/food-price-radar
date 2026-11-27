"""Gemini 연동. 계산된 통계(facts)만 근거로 설명·요약·추천하게 한다."""
from __future__ import annotations

SYSTEM = """너는 'Food Price Radar(저렴이 레이더)'의 장보기 도우미야.
규칙:
1. 가격·비율·달 같은 숫자는 반드시 [데이터]에 있는 값만 사용해. 없는 숫자를 지어내지 마.
2. [데이터]에 없는 품목이나 기간을 물으면 "이 사이트 데이터에는 없어요"라고 말하고, 일반 상식은 '일반적으로'라고 구분해서 짧게만 말해.
3. '평년 패턴상 경향'은 과거 평균일 뿐 미래 보장이 아니라는 점을 필요할 때 한 번 언급해.
4. 한국어, 친근하지만 간결하게. 마크다운 목록 사용 가능.
5. 데이터 출처는 KAMIS 소매가격(전국 평균)이야."""

DEFAULT_MODELS = ["gemini-4-hydrogen", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash"]


class LLM:
    def __init__(self, api_key: str | None, model: str | None = None):
        self.ok = bool(api_key)
        self.models = [model] + DEFAULT_MODELS if model else DEFAULT_MODELS
        self.client = None
        if self.ok:
            from google import genai

            self.client = genai.Client(api_key=api_key)

    def ask(self, prompt: str, facts: str, temperature: float = 0.4) -> str:
        if not self.ok:
            return "⚠️ Gemini API 키가 설정되지 않았어요. (Streamlit secrets에 GEMINI_API_KEY 추가)"
        from google.genai import types

        contents = f"[데이터]\n{facts}\n\n[요청]\n{prompt}"
        last = None
        for m in self.models:
            try:
                r = self.client.models.generate_content(
                    model=m,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM, temperature=temperature
                    ),
                )
                if r.text:
                    return r.text
            except Exception as e:  # 모델 이름 변경·한도 초과 시 다음 모델로
                last = e
        return f"⚠️ AI 응답을 받지 못했어요. 잠시 후 다시 시도해 주세요. ({type(last).__name__ if last else '빈 응답'})"


def briefing_prompt(table_text: str) -> str:
    return (
        "아래 '이번 주 저렴이 순위'를 보고 장보기 브리핑을 써줘.\n"
        "- 지금 사기 좋은 품목 3개와 이유(평년 대비 %, 4주 경향)\n"
        "- 지금은 비싸니 미루면 좋은 품목 1~2개\n"
        "- 5~8줄 이내\n\n" + table_text
    )


def recipe_prompt(cheap_items: list[str], people: int, extra: str) -> str:
    return (
        f"지금 평년보다 싼 재료: {', '.join(cheap_items)}.\n"
        f"{people}인분 기준으로 이 재료를 최대한 활용한 집밥 메뉴 3개를 추천해줘. "
        "각 메뉴마다 쓰는 싼 재료, 추가로 필요한 재료, 간단한 조리 순서(3~5단계)를 적어줘. "
        "가격은 [데이터]에 있는 값만 언급해."
        + (f"\n추가 조건: {extra}" if extra else "")
    )
