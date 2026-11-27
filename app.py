"""Food Price Radar (저렴이 레이더) — 식재료 저가 시즌 대시보드 + 생성형 AI 장보기 도우미"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import analysis as A
from llm import LLM, briefing_prompt, recipe_prompt

st.set_page_config(page_title="Food Price Radar · 저렴이 레이더", page_icon="🛒", layout="wide")

DATA_PATH = Path(__file__).parent / "prices.csv"
MAX_AI_CALLS = 30  # 한 사람이 한 번 접속해서 쓸 수 있는 AI 호출 수 (무료 한도 보호)


def secret(name: str, default=None):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ---------------- 데이터 ----------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_data() -> pd.DataFrame | None:
    if DATA_PATH.exists():
        return A.load_prices(DATA_PATH)
    key, cid = secret("KAMIS_CERT_KEY"), secret("KAMIS_CERT_ID")
    if key and cid:  # CSV가 없으면 서버에서 직접 수집 (첫 접속 시 1~3분)
        import kamis

        df = kamis.collect(key, cid, years=int(secret("KAMIS_YEARS", 3)), log=lambda *_: None)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_radar(df: pd.DataFrame) -> pd.DataFrame:
    return A.radar_table(df)


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def cached_facts(df: pd.DataFrame) -> str:
    return A.facts_for_llm(df)


@st.cache_resource
def get_llm() -> LLM:
    return LLM(secret("GEMINI_API_KEY"), secret("GEMINI_MODEL"))


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_ask(prompt: str, facts: str) -> str:
    return get_llm().ask(prompt, facts)


def ai(prompt: str, facts: str, use_cache: bool = True) -> str:
    n = st.session_state.get("ai_calls", 0)
    if n >= MAX_AI_CALLS:
        return "⚠️ 이번 접속에서 AI 사용 횟수를 모두 썼어요. 새로고침 후 다시 이용해 주세요."
    st.session_state["ai_calls"] = n + 1
    return cached_ask(prompt, facts) if use_cache else get_llm().ask(prompt, facts)


def pct(x) -> str:
    return "–" if x != x else f"{x * 100:+.0f}%"


# ---------------- 화면 ----------------
st.title("🛒 Food Price Radar · 저렴이 레이더")
st.caption("공공 가격 데이터(KAMIS)로 식재료가 **해마다 반복해서 싸지는 시기**를 찾고, 생성형 AI(Gemini)가 장보기 타이밍을 설명해 주는 서비스")

with st.spinner("가격 데이터를 불러오는 중... (처음 접속이면 1~3분 걸릴 수 있어요)"):
    try:
        df = load_data()
    except Exception as e:
        df = None
        st.error(f"데이터 수집 중 오류: {e}")

if df is None or df.empty:
    st.warning(
        "아직 가격 데이터가 없어요.\n\n"
        "- 방법 1: 로컬에서 `python kamis.py --years 5` 실행 후 `prices.csv`를 GitHub에 올리기\n"
        "- 방법 2: Streamlit secrets에 `KAMIS_CERT_KEY`, `KAMIS_CERT_ID` 넣기 (서버가 직접 수집)"
    )
    st.stop()

llm = get_llm()
radar = cached_radar(df)
facts = cached_facts(df)
asof = df["date"].max().date()

with st.sidebar:
    st.subheader("데이터 정보")
    st.write(f"기준일: **{asof}**")
    st.write(f"기간: {df['date'].min().date()} ~ {asof}")
    st.write(f"품목: {df['item'].nunique()}개 · {len(df):,}건")
    st.caption("출처: KAMIS 농산물유통정보 소매가격(전국 평균)")
    st.divider()
    st.write("🤖 AI:", "Gemini 연결됨 ✅" if llm.ok else "키 없음 ❌")
    st.caption(f"이번 접속 AI 사용 {st.session_state.get('ai_calls', 0)}/{MAX_AI_CALLS}")

tab_radar, tab_item, tab_chat, tab_recipe, tab_about = st.tabs(
    ["🏠 저렴이 레이더", "📈 품목 분석", "🤖 AI에게 물어보기", "🍳 저렴이 레시피", "ℹ️ 서비스 소개"]
)

# ---- 1. 저렴이 레이더 ----
with tab_radar:
    if radar.empty:
        st.info("최근 가격이 있는 품목이 없어요.")
    else:
        cheap = radar.dropna(subset=["vs_normal"])
        below = cheap[cheap["vs_normal"] < 0]
        if below.empty:
            st.subheader("이번 주는 평년보다 싼 품목이 없어요 · 그나마 덜 오른 품목")
            below = cheap
        else:
            st.subheader(f"이번 주 평년보다 싼 품목 ({len(below)}개)")
        cols = st.columns(4)
        for c, (_, r) in zip(cols, below.head(4).iterrows()):
            c.metric(
                f"{r['item']} ({r['unit']})",
                f"{r['price']:,.0f}원",
                f"평년 대비 {pct(r['vs_normal'])}",
                delta_color="inverse",
            )

        show = radar.assign(
            가격=radar["price"].map(lambda v: f"{v:,.0f}원"),
            평년대비=radar["vs_normal"].map(pct),
            한달전대비=radar["vs_month"].map(pct),
            앞으로4주_평년경향=radar["trend4"].map(pct),
            이상신호=radar["z"].map(lambda z: "⚠️ 급변" if z == z and abs(z) >= 2.5 else ""),
        ).rename(columns={"item": "품목", "unit": "단위", "cheapest_months": "보통 가장 싼 달"})
        st.dataframe(
            show[["품목", "단위", "가격", "평년대비", "한달전대비", "앞으로4주_평년경향", "보통 가장 싼 달", "이상신호"]],
            hide_index=True,
            width="stretch",
        )
        st.caption("평년대비: 최근 5년 같은 시기(±1주) 가격 중 최고·최저를 뺀 평균(KAMIS 평년 방식)과 비교 · 앞으로 4주 경향: 과거 해들의 평균적인 움직임(예측 보장 아님)")

        fig = px.bar(
            cheap,
            x="item",
            y=cheap["vs_normal"] * 100,
            color=cheap["vs_normal"] < 0,
            color_discrete_map={True: "#2a9d8f", False: "#e76f51"},
            labels={"item": "", "y": "평년 대비 (%)"},
        )
        fig.update_layout(showlegend=False, height=330, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        st.subheader("🤖 AI 장보기 브리핑")
        if st.button("이번 주 브리핑 생성", type="primary"):
            table_text = "\n".join(
                f"{r['item']}: {r['price']:,.0f}원/{r['unit']}, 평년대비 {pct(r['vs_normal'])}, "
                f"4주전대비 {pct(r['vs_month'])}, 4주 평년경향 {pct(r['trend4'])}"
                for _, r in radar.iterrows()
            )
            with st.spinner("Gemini가 브리핑을 쓰는 중..."):
                st.session_state["briefing"] = ai(briefing_prompt(table_text), facts)
        if "briefing" in st.session_state:
            st.markdown(st.session_state["briefing"])

# ---- 2. 품목 분석 ----
with tab_item:
    items = sorted(df["item"].unique())
    default = items.index("배추") if "배추" in items else 0
    item = st.selectbox("품목 선택", items, index=default)
    g = df[df["item"] == item]
    sp = A.seasonal_profile(g)
    stt = A.current_status(g, df["date"].max())

    c1, c2, c3, c4 = st.columns(4)
    if stt:
        c1.metric("최근 가격", f"{stt['price']:,.0f}원", help=stt["unit"])
        c2.metric("평년 대비", pct(stt["vs_normal"]), delta_color="off")
        c3.metric("4주 전 대비", pct(stt["vs_month"]), delta_color="off")
        c4.metric("앞으로 4주 평년 경향", pct(stt["trend4"]), delta_color="off")
    else:
        st.info("최근 3주 안에 조사된 가격이 없어요. 출하 시기가 아닐 수 있어요.")

    st.markdown("#### 연도별 가격 겹쳐보기")
    w = A.weekly(g).to_frame("price")
    w["연도"] = w.index.year.astype(str)
    w["x"] = pd.to_datetime("2000-" + w.index.strftime("%m-%d"), errors="coerce")
    w = w.dropna(subset=["x"])
    fig = px.line(w, x="x", y="price", color="연도", labels={"x": "", "price": "원"})
    fig.update_xaxes(tickformat="%m월")
    fig.update_layout(height=380, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### 월별 가격 지수 (1.0 = 그 해 평균)")
        mi = sp["month_index"]
        if len(mi):
            bar = pd.DataFrame({"월": [f"{m}월" for m in mi.index], "지수": mi.values})
            low = set(c["month"] for c in sp["cheapest"])
            bar["구분"] = ["싼 달" if m in low else "" for m in mi.index]
            fig2 = px.bar(bar, x="월", y="지수", color="구분",
                          color_discrete_map={"싼 달": "#2a9d8f", "": "#adb5bd"})
            fig2.add_hline(y=1.0, line_dash="dot")
            fig2.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10))
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info("계절 패턴을 계산할 만큼 데이터가 충분하지 않아요.")
    with right:
        st.markdown("#### 자동 검출한 저가 시즌")
        for c in sp["cheapest"]:
            st.write(
                f"**{c['month']}월** · 연평균보다 {abs(c['index'] - 1) * 100:.0f}% "
                f"{'저렴' if c['index'] < 1 else '비쌈'} · 반복성 {c['repeat']}/{c['years']}년"
            )
        st.caption("반복성 = 분석한 해 중 그 달이 '그 해 가장 싼 3개월'에 든 횟수")

    if st.button(f"🤖 {item} 언제 사면 좋을지 AI 설명", key="item_ai"):
        with st.spinner("Gemini가 분석 결과를 설명하는 중..."):
            line = A.item_fact_line(g, df["date"].max())
            st.markdown(ai(f"{item}을(를) 언제 사면 가장 저렴한지, 지금 사도 되는지 설명해줘. 5줄 이내.", line))

# ---- 3. AI 질문 ----
with tab_chat:
    st.caption("예) 꽃게 언제 사는 게 제일 싸? · 이번 주 싼 채소 뭐 있어? · 김장 재료 지금 사도 돼?")
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
    for role, msg in st.session_state["chat"]:
        st.chat_message(role).markdown(msg)
    q = st.chat_input("식재료 가격에 대해 물어보세요")
    if q:
        st.chat_message("user").markdown(q)
        history = "\n".join(f"{'사용자' if r == 'user' else 'AI'}: {m}" for r, m in st.session_state["chat"][-6:])
        prompt = (f"[이전 대화]\n{history}\n\n" if history else "") + f"질문: {q}"
        with st.chat_message("assistant"), st.spinner("생각 중..."):
            ans = ai(prompt, facts, use_cache=not history)
            st.markdown(ans)
        st.session_state["chat"] += [("user", q), ("assistant", ans)]

# ---- 4. 레시피 ----
with tab_recipe:
    cheap_items = radar[radar["vs_normal"] < 0]["item"].tolist() if not radar.empty else []
    if not cheap_items:
        st.info("지금 평년보다 싼 품목이 없어요.")
    else:
        sel = st.multiselect("사용할 싼 재료", cheap_items, default=cheap_items[:4])
        people = st.slider("인분", 1, 6, 2)
        extra = st.text_input("추가 조건 (선택)", placeholder="예) 자취생, 30분 이내, 매운 거 싫어함")
        if st.button("🍳 레시피 추천받기", type="primary", disabled=not sel):
            with st.spinner("Gemini가 메뉴를 고르는 중..."):
                st.markdown(ai(recipe_prompt(sel, people, extra), facts))

# ---- 5. 소개 ----
with tab_about:
    st.markdown(
        """
### 어떤 서비스인가요?
꽃게는 금어기가 풀린 직후, 배추는 출하가 몰리는 시기처럼 **식재료마다 해마다 반복해서 싸지는 시기**가 있습니다.
Food Price Radar는 공공 가격 데이터로 이 시기를 자동으로 찾아내고, 생성형 AI가 "지금 사도 되는지"를 쉬운 말로 알려줍니다.

### 어떻게 동작하나요?
1. **데이터 수집**: KAMIS(한국농수산식품유통공사) Open-API로 품목별 일별 소매가격(전국 평균) 수집
2. **통계 분석 (Python)**: 연도별 평균으로 나눠 물가 상승을 걷어낸 뒤 월별 가격 지수, 반복성, 평년 대비 비율, 급변(robust z-score) 계산
3. **생성형 AI (Gemini)**: 계산된 숫자만 프롬프트에 넣고, AI는 **설명·요약·추천만** 담당 → 가격을 지어내는 환각(hallucination) 방지

| 기능 | 담당 |
|---|---|
| 저가 시즌 검출, 평년 대비, 4주 경향 | Python 통계 |
| 장보기 브리핑, 질문 답변, 레시피 추천 | Gemini |

### 한계
- '앞으로 4주 경향'은 과거 평균적인 움직임이며 미래 가격을 보장하지 않습니다.
- KAMIS 조사 품목·등급 기준이라 동네 마트 가격과 다를 수 있습니다.
"""
    )
