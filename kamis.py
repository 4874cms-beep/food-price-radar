"""KAMIS(농산물유통정보) Open-API 수집기.

사용법 (로컬 PC 또는 GitHub Actions):
    export KAMIS_CERT_KEY=발급받은_인증키
    export KAMIS_CERT_ID=KAMIS_아이디
    python kamis.py --years 5

결과: prices.csv  (date, item, category, kind, rank, unit, price)

품목코드를 직접 외우지 않아도 되도록,
1) '일별 부류별 가격정보(dailyPriceByCategoryList)'로 품목/품종/등급 코드 목록을 먼저 받아오고
2) TARGETS에 적힌 품목 이름과 맞는 코드를 골라
3) '기간별 품목 가격정보(periodProductList)'로 과거 소매가격을 1년씩 끊어서 받습니다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"
DATA_PATH = Path(__file__).parent / "prices.csv"

CATEGORIES = {
    "100": "식량작물",
    "200": "채소류",
    "400": "과일류",
    "500": "축산물",
    "600": "수산물",
}

# (KAMIS 품목명, 품종에 포함될 단어 또는 None, 화면 표시 이름, 부분일치 허용 여부)
TARGETS = [
    ("배추", None, "배추", False),
    ("양배추", None, "양배추", False),
    ("무", None, "무", False),
    ("양파", None, "양파", False),
    ("파", "대파", "대파", False),
    ("감자", None, "감자", False),
    ("고구마", None, "고구마", False),
    ("오이", None, "오이", False),
    ("호박", "애호박", "애호박", False),
    ("당근", None, "당근", False),
    ("시금치", None, "시금치", False),
    ("상추", None, "상추", False),
    ("토마토", None, "토마토", False),
    ("마늘", None, "마늘", True),
    ("사과", None, "사과", False),
    ("배", None, "배", False),
    ("감귤", None, "감귤", False),
    ("딸기", None, "딸기", False),
    ("수박", None, "수박", False),
    ("참외", None, "참외", False),
    ("복숭아", None, "복숭아", False),
    ("포도", None, "포도", False),
    ("고등어", None, "고등어", False),
    ("갈치", None, "갈치", False),
    ("오징어", None, "오징어", True),
    ("꽃게", None, "꽃게", True),
    ("계란", None, "계란", True),
]


def _get(params: dict, cert_key: str, cert_id: str, retries: int = 3) -> dict:
    q = {**params, "p_cert_key": cert_key, "p_cert_id": cert_id, "p_returntype": "json"}
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(BASE_URL, params=q, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # 네트워크/JSON 오류는 잠시 쉬고 재시도
            last_err = e
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"KAMIS 요청 실패: {last_err}")


def _items(js: dict) -> list[dict]:
    """KAMIS 응답에서 item 리스트만 꺼낸다. (데이터가 없으면 data가 ['001'] 같은 리스트로 옴)"""
    data = js.get("data") if isinstance(js, dict) else None
    if isinstance(data, dict):
        items = data.get("item", [])
        if isinstance(items, dict):
            items = [items]
        return items or []
    return []


def _to_price(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").strip()
    if s in ("", "-", "0"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_catalog(cert_key: str, cert_id: str, ref_dates: list[dt.date]) -> pd.DataFrame:
    """여러 날짜의 부류별 소매가격 목록에서 품목/품종/등급 코드표를 만든다.
    (제철에만 나오는 딸기·수박 같은 품목을 잡으려고 1년에 걸쳐 여러 날짜를 본다)"""
    rows = []
    for cat in CATEGORIES:
        for d in ref_dates:
            for back in range(0, 7):  # 주말·공휴일이면 하루씩 앞으로
                day = d - dt.timedelta(days=back)
                try:
                    js = _get(
                    {
                        "action": "dailyPriceByCategoryList",
                        "p_product_cls_code": "01",
                        "p_item_category_code": cat,
                        "p_country_code": "",
                        "p_regday": day.isoformat(),
                        "p_convert_kg_yn": "N",
                    },
                    cert_key,
                    cert_id,
                    )
                except Exception as e:
                    print(f"    ! 코드표 {cat} {day} 건너뜀: {e}", flush=True)
                    continue
                items = _items(js)
                if items:
                    for it in items:
                        rows.append(
                            {
                                "category_code": cat,
                                "item_name": str(it.get("item_name", "")).strip(),
                                "item_code": str(it.get("item_code", "")).strip(),
                                "kind_name": str(it.get("kind_name", "")).strip(),
                                "kind_code": str(it.get("kind_code", "")).strip(),
                                "rank": str(it.get("rank", "")).strip(),
                                "rank_code": str(it.get("rank_code", "")).strip(),
                                "unit": str(it.get("unit", "")).strip(),
                            }
                        )
                    break
                time.sleep(0.3)
    cat_df = pd.DataFrame(rows).drop_duplicates(["item_code", "kind_code", "rank_code"])
    return cat_df


def pick_targets(catalog: pd.DataFrame) -> list[dict]:
    """품목마다 사용할 (품종, 등급) 목록을 고른다.
    배추(봄·고랭지·가을·월동)처럼 품종이 계절별로 나뉜 품목은 단위가 같은 품종을 모두 받아
    날짜별 평균으로 이어 붙여야 1년 내내 가격이 이어진다."""
    picked = []
    for name, kind_kw, display, contains in TARGETS:
        if contains:
            cand = catalog[catalog["item_name"].str.contains(name, na=False)]
        else:
            cand = catalog[catalog["item_name"] == name]
        if kind_kw:
            cand = cand[cand["kind_name"].str.contains(kind_kw, na=False)]
        if cand.empty:
            print(f"  - {display}: 코드표에서 찾지 못함 (건너뜀)")
            continue
        # 같은 품목코드(첫 번째)만, 품종마다 '상품' 등급 우선
        cand = cand[cand["item_code"] == cand["item_code"].iloc[0]]
        rows = []
        for _, kg in cand.groupby("kind_code", sort=False):
            top = kg[kg["rank"].str.contains("상", na=False)]
            rows.append((top if not top.empty else kg).iloc[0])
        unit = rows[0]["unit"]
        rows = [r for r in rows if r["unit"] == unit]  # 단위가 다른 품종은 섞지 않음
        kinds = [{**r.to_dict(), "display": display} for r in rows]
        picked.append({"display": display, "kinds": kinds})
        print(f"  - {display}: {rows[0]['item_name']} / " + ", ".join(r["kind_name"] for r in rows) + f" ({unit})")
    return picked


def fetch_history(t: dict, start: dt.date, end: dt.date, cert_key: str, cert_id: str) -> pd.DataFrame:
    out = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=182), end)  # 반년씩 (응답이 커서 1년은 느림)
        try:
            js = _get(
            {
                "action": "periodProductList",
                "p_productclscode": "01",
                "p_startday": cur.isoformat(),
                "p_endday": chunk_end.isoformat(),
                "p_itemcategorycode": t["category_code"],
                "p_itemcode": t["item_code"],
                "p_kindcode": t["kind_code"],
                "p_productrankcode": t["rank_code"],
                "p_countrycode": "",
                "p_convert_kg_yn": "N",
            },
            cert_key,
            cert_id,
            )
        except Exception as e:  # 한 구간 실패해도 전체를 멈추지 않고 건너뜀
            print(f"    ! {t['display']} {cur}~{chunk_end} 건너뜀: {e}", flush=True)
            cur = chunk_end + dt.timedelta(days=1)
            continue
        for it in _items(js):
            price = _to_price(it.get("price"))
            if price is None:
                continue
            yyyy = str(it.get("yyyy", "")).strip()
            regday = str(it.get("regday", "")).strip()  # 'MM/DD'
            try:
                if "/" in regday and yyyy:
                    m, d = regday.split("/")[:2]
                    date = dt.date(int(yyyy), int(m), int(d))
                else:
                    date = pd.to_datetime(regday).date()
            except Exception:
                continue
            out.append({"date": date, "county": str(it.get("countyname", "")).strip(), "price": price})
        cur = chunk_end + dt.timedelta(days=1)
        time.sleep(0.3)

    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    avg = df[df["county"] == "평균"]
    if avg.empty:  # '평균' 행이 없으면 지역 가격의 평균을 사용
        avg = df[~df["county"].isin(["평년"])].groupby("date", as_index=False)["price"].mean()
    else:
        avg = avg.groupby("date", as_index=False)["price"].mean()
    avg["item"] = t["display"]
    avg["category"] = CATEGORIES.get(t["category_code"], t["category_code"])
    avg["kind"] = t["kind_name"]
    avg["rank"] = t["rank"]
    avg["unit"] = t["unit"]
    return avg[["date", "item", "category", "kind", "rank", "unit", "price"]]


def collect(cert_key: str, cert_id: str, years: int = 5, log=print) -> pd.DataFrame:
    today = dt.date.today()
    ref_dates = [today - dt.timedelta(days=30 * k) for k in (0, 3, 6, 9)]
    log("⓪ KAMIS 연결 확인 중...")
    try:
        _get({"action": "dailyPriceByCategoryList", "p_product_cls_code": "01",
              "p_item_category_code": "200", "p_country_code": "", "p_regday": today.isoformat(),
              "p_convert_kg_yn": "N"}, cert_key, cert_id)
    except Exception as e:
        raise RuntimeError(f"KAMIS 서버에 연결하지 못했어요 (해외 서버 차단 가능성). PC에서 수집해 주세요. [{e}]")
    log("① 품목 코드표 받는 중...")
    catalog = fetch_catalog(cert_key, cert_id, ref_dates)
    if catalog.empty:
        raise RuntimeError("코드표가 비어 있어요. 인증키(cert_key)와 아이디(cert_id)를 확인해 주세요.")
    log(f"   코드표 {len(catalog)}행")
    targets = pick_targets(catalog)
    start = dt.date(today.year - years, 1, 1)
    frames = []
    for i, t in enumerate(targets, 1):
        log(f"② [{i}/{len(targets)}] {t['display']} {start}~{today} 가격 받는 중...")
        parts = [fetch_history(k, start, today, cert_key, cert_id) for k in t["kinds"]]
        parts = [p for p in parts if not p.empty]
        if not parts:
            continue
        h = pd.concat(parts)
        kinds = "·".join(dict.fromkeys(h["kind"]))
        h = h.groupby("date", as_index=False).agg(
            item=("item", "first"), category=("category", "first"), rank=("rank", "first"),
            unit=("unit", "first"), price=("price", "mean"),
        )
        h["kind"] = kinds
        frames.append(h[["date", "item", "category", "kind", "rank", "unit", "price"]])
    if not frames:
        raise RuntimeError("가격 데이터를 하나도 받지 못했어요.")
    return pd.concat(frames, ignore_index=True).sort_values(["item", "date"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5, help="몇 년 치를 받을지 (기본 5년)")
    args = ap.parse_args()
    key, cid = os.environ.get("KAMIS_CERT_KEY"), os.environ.get("KAMIS_CERT_ID")
    if not key or not cid:
        sys.exit("환경변수 KAMIS_CERT_KEY, KAMIS_CERT_ID를 먼저 설정해 주세요.")
    df = collect(key, cid, args.years)
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"완료: {len(df):,}행, {df['item'].nunique()}개 품목 → {DATA_PATH}")


if __name__ == "__main__":
    main()
