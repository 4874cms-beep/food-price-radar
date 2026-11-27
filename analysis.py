"""가격 통계 계산. 숫자는 전부 여기서 계산하고, LLM은 이 결과를 설명만 한다."""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_KO = [f"{m}월" for m in range(1, 13)]


def load_prices(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    for c in ("category", "kind", "rank", "unit"):
        if c in df:
            df[c] = df[c].fillna("").astype(str)
    return df.sort_values(["item", "date"])


def weekly(df_item: pd.DataFrame) -> pd.Series:
    s = df_item.set_index("date")["price"].resample("W-SUN").mean().dropna()
    return s


def _year_normalized(w: pd.Series) -> pd.DataFrame:
    """연도별 평균으로 나눠 물가 상승(추세)을 걷어낸 주간 지수."""
    d = w.to_frame("price")
    d["year"] = d.index.year
    d["week"] = d.index.isocalendar().week.astype(int).clip(upper=52)
    d["month"] = d.index.month
    d["norm"] = d["price"] / d.groupby("year")["price"].transform("mean")
    return d


def seasonal_profile(df_item: pd.DataFrame) -> dict:
    """월별 가격 지수(1.0=그 해 평균), 가장 싼 달, 반복성."""
    w = weekly(df_item)
    d = _year_normalized(w)
    # 6개월 이상 관측된 해만 사용
    good_years = d.groupby("year")["month"].nunique()
    good_years = good_years[good_years >= 6].index
    d = d[d["year"].isin(good_years)]
    if d.empty:
        return {"month_index": pd.Series(dtype=float), "cheapest": [], "years": 0}

    ym = d.groupby(["year", "month"])["norm"].mean().unstack()  # 연 x 월
    month_index = ym.median(axis=0)  # 연도별 중앙값 → 이상치에 덜 민감
    n_years = ym.shape[0]

    # 반복성: 해마다 그 달이 '그 해 가장 싼 3개월' 안에 들었는가
    bottom3 = ym.rank(axis=1, method="min") <= 3
    repeat = bottom3.sum(axis=0)
    valid = ym.notna().sum(axis=0)

    order = month_index.sort_values().index[:3]
    cheapest = [
        {
            "month": int(m),
            "index": float(month_index[m]),
            "repeat": int(repeat.get(m, 0)),
            "years": int(valid.get(m, 0)),
        }
        for m in order
    ]
    return {"month_index": month_index, "cheapest": cheapest, "years": n_years, "year_month": ym}


def week_profile(df_item: pd.DataFrame) -> pd.Series:
    d = _year_normalized(weekly(df_item))
    prof = d.groupby("week")["norm"].median()
    return prof.reindex(range(1, 53)).interpolate(limit_direction="both")


def current_status(df_item: pd.DataFrame, asof: pd.Timestamp | None = None) -> dict | None:
    """현재 가격, 평년 동기 대비, 한 달 전 대비, 앞으로 4주 평년 경향, 이상 여부."""
    w = weekly(df_item)
    if w.empty:
        return None
    asof = asof or w.index.max()
    recent = w[w.index <= asof]
    if recent.empty or (asof - recent.index.max()).days > 21:
        return None  # 최근 3주 안에 가격이 없음 = 출하 안 되는 시기
    last_date = recent.index.max()
    last = float(recent.iloc[-1])

    # 평년(KAMIS 방식): 최근 5년의 같은 주(±1주) 연도별 평균 중 최고·최저를 뺀 나머지의 평균
    hist = w[(w.index.year < last_date.year) & (w.index.year >= last_date.year - 5)]
    wk = int(min(last_date.isocalendar().week, 52))
    hw = hist[hist.index.isocalendar().week.astype(int).clip(upper=52).between(wk - 1, wk + 1)]
    per_year = hw.groupby(hw.index.year).mean().sort_values()
    if len(per_year) >= 5:
        per_year = per_year.iloc[1:-1]
    baseline = float(per_year.mean()) if len(per_year) >= 2 else np.nan
    vs_normal = last / baseline - 1 if baseline == baseline else np.nan

    month_ago = recent[recent.index <= last_date - pd.Timedelta(days=28)]
    vs_month = last / float(month_ago.iloc[-1]) - 1 if not month_ago.empty else np.nan

    prof = week_profile(df_item)
    ahead = [((wk + k - 1) % 52) + 1 for k in range(1, 5)]
    trend4 = float(prof.loc[ahead].mean() / prof.loc[wk] - 1) if prof.notna().all() else np.nan

    # 이상치: (가격/평년) 비율의 과거 분포 대비 robust z-score
    z = np.nan
    if len(hist) > 52:
        d = _year_normalized(w)
        resid = np.log(d["norm"]) - np.log(d["week"].map(week_profile(df_item)))
        med = resid.median()
        mad = (resid - med).abs().median() * 1.4826
        if mad > 0:
            z = float((resid.iloc[-1] - med) / mad)

    first = df_item.iloc[0]
    return {
        "item": first["item"],
        "category": first.get("category", ""),
        "unit": str(first.get("unit", "")) or str(first.get("kind", "")),
        "date": last_date.date(),
        "price": last,
        "baseline": baseline,
        "vs_normal": vs_normal,
        "vs_month": vs_month,
        "trend4": trend4,
        "z": z,
    }


def radar_table(df: pd.DataFrame) -> pd.DataFrame:
    asof = df["date"].max()
    rows = []
    for item, g in df.groupby("item"):
        st = current_status(g, asof)
        if st:
            sp = seasonal_profile(g)
            st["cheapest_months"] = ", ".join(f"{c['month']}월" for c in sp["cheapest"])
            rows.append(st)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    return t.sort_values("vs_normal", na_position="last").reset_index(drop=True)


def _pct(x) -> str:
    return "정보 없음" if x != x else f"{x * 100:+.0f}%"


def item_fact_line(df_item: pd.DataFrame, asof=None) -> str:
    """LLM에 넣을 한 줄 요약 (숫자는 전부 계산된 값)."""
    name = df_item["item"].iloc[0]
    sp = seasonal_profile(df_item)
    st = current_status(df_item, asof)
    season = "; ".join(
        f"{c['month']}월(연평균 대비 {(c['index'] - 1) * 100:+.0f}%, {c['years']}년 중 {c['repeat']}년 하위3개월)"
        for c in sp["cheapest"]
    ) or "계절 패턴 계산 불가"
    if st:
        now = (
            f"최근 {st['date']} 가격 {st['price']:,.0f}원/{st['unit']}, 평년 같은 시기 대비 {_pct(st['vs_normal'])}, "
            f"4주 전 대비 {_pct(st['vs_month'])}, 평년 패턴상 앞으로 4주 경향 {_pct(st['trend4'])}"
        )
        if st["z"] == st["z"] and abs(st["z"]) >= 2.5:
            now += f", 평소 패턴에서 크게 벗어남(z={st['z']:+.1f})"
    else:
        now = "최근 3주 안에 조사된 가격 없음(출하 시기가 아닐 수 있음)"
    return f"- {name}: {now}. 가장 싼 달 Top3: {season} (분석 {sp['years']}년)"


def facts_for_llm(df: pd.DataFrame) -> str:
    asof = df["date"].max()
    lines = [item_fact_line(g, asof) for _, g in df.groupby("item")]
    return f"[데이터 기준일 {asof.date()}, KAMIS 소매가격(전국 평균)]\n" + "\n".join(lines)
