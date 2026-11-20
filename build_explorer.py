#!/usr/bin/env python3
"""조회 사이트(docs/)용 데이터 만들기.

analyze.py 를 먼저 실행해 results/first_purchase_customers.csv 를 만든 뒤 실행:
    .venv/bin/python build_explorer.py
출력: docs/data/explorer.js (window.OLIST = {...}; 파일을 바로 열어도 동작하도록 JSON 대신 스크립트)
  - cube: 조건 조합별 고객 수·재구매 수 (w=1: 상품 도착 전에 리뷰 작성) (고객 단위 원본 대신 집계본만 공개)
  - repeaters: 재구매 고객의 조건과 재구매까지 걸린 일수
  - reviews: 글이 있는 부정 리뷰(1~2점) 원문과 주제
"""
import json
import re
import os

import numpy as np
import pandas as pd

from analyze import RAW, THEMES, norm_text

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "docs", "data")
os.makedirs(OUT, exist_ok=True)

f = pd.read_csv(os.path.join(BASE, "results", "first_purchase_customers.csv"),
                parse_dates=["order_delivered_customer_date"])
cust = pd.read_csv(os.path.join(RAW, "olist_customers_dataset.csv"))
rv = pd.read_csv(os.path.join(RAW, "olist_order_reviews_dataset.csv"), parse_dates=["review_creation_date"])

f["state"] = f.customer_id.map(cust.set_index("customer_id").customer_state)
f["pay"] = pd.cut(f.payment, [-1, 50, 100, 200, np.inf], labels=["50 미만", "50~100", "100~200", "200 이상"])
f["bucket"] = f.delay_bucket.fillna("기록 없음")

# 리뷰: analyze.py 와 같은 리뷰(답변 시각이 가장 늦은 것)
r = rv.sort_values("review_answer_timestamp").groupby("order_id").last()
f["created"] = f.order_id.map(r.review_creation_date)
f["before"] = f.created.dt.normalize() < f.order_delivered_customer_date.dt.normalize()

MONTHS = sorted(f.month.unique())
CATS = f.cat.value_counts().index.tolist()
STATES = f.state.value_counts().index.tolist()
STATE_KO = {
    "SP": "상파울루", "RJ": "리우데자네이루", "MG": "미나스제라이스", "RS": "히우그란지두술", "PR": "파라나",
    "SC": "산타카타리나", "BA": "바이아", "ES": "이스피리투산투", "GO": "고이아스", "DF": "연방구(브라질리아)",
    "PE": "페르남부쿠", "CE": "세아라", "PA": "파라", "MT": "마투그로수", "MA": "마라냥", "MS": "마투그로수두술",
    "PB": "파라이바", "PI": "피아우이", "RN": "히우그란지두노르치", "AL": "알라고아스", "SE": "세르지피",
    "TO": "토칸칭스", "RO": "혼도니아", "AM": "아마조나스", "AC": "아크리", "AP": "아마파", "RR": "호라이마",
}
PAYS = ["50 미만", "50~100", "100~200", "200 이상"]
BUCKETS = ["정시", "1~3일", "4~7일", "8일 이상", "기록 없음"]
REVIEWS = ["긍정", "중립", "부정", "리뷰 없음"]
THEME_NAMES = list(THEMES)

idx = lambda col, values: f[col].astype(str).map({v: i for i, v in enumerate(values)})  # noqa: E731
codes = pd.DataFrame({
    "m": idx("month", MONTHS), "c": idx("cat", CATS), "s": idx("state", STATES),
    "p": idx("pay", PAYS), "b": idx("bucket", BUCKETS), "r": idx("review_group", REVIEWS),
    "w": f.before.astype(int),
})
assert not codes.isna().any().any(), codes.isna().sum()
dims = list(codes.columns)

cube = codes.assign(n=1, rep=f.repeat.astype(int)).groupby(dims).sum().reset_index()
repeaters = codes[f.repeat].assign(d=f.loc[f.repeat, "days_to_repeat"].round().astype(int))

# 부정 리뷰 원문: analyze.py 와 같은 주제 사전
f["text"] = f.order_id.map(r.review_comment_message).fillna("").astype(str).str.strip()
neg = (f.review_group == "부정") & f.text.ne("")
nt = f.loc[neg, "text"].map(norm_text)
mask = sum(nt.map(lambda s, ws=ws: any(w in s for w in ws)).astype(int) * (1 << i)
           for i, ws in enumerate(THEMES.values()))
reviews = codes[neg].assign(sc=f.loc[neg, "score"].astype(int), t=mask, x=f.loc[neg, "text"].str.replace(r"\s+", " ", regex=True))
# 한국어 번역: translate_reviews.py 가 만든 results/review_ko.json (없으면 빈 칸)
KO = os.path.join(BASE, "results", "review_ko.json")
ko = json.load(open(KO, encoding="utf-8")) if os.path.exists(KO) else {}
reviews["ko"] = reviews.x.map(ko).fillna("")
# 공개 사이트에 실명이 나가지 않도록 리뷰 속 사람 이름을 가림 (번역 중 발견된 것)
NAMES = ["Lourdes Marli Barichello Conceição", "Diego Machado Lopes", "Cyro Dias Braga", "Suzana Costa", "Hélder", "mateus"]
# 한글이 바로 붙는 번역문에서도 걸리도록 \b 대신 라틴 문자 경계만 확인
name_re = re.compile(r"(?<![A-Za-zÀ-ÿ])(" + "|".join(map(re.escape, NAMES)) + r")(?![A-Za-zÀ-ÿ])", re.IGNORECASE)
for col in ("x", "ko"):
    reviews[col] = reviews[col].str.replace(name_re, "○○○", regex=True)
print(f"번역된 리뷰 {(reviews.ko != '').sum()}/{len(reviews)}건")

# 확인용: 사후 발견 수치 재현
late = f.late & f.delivery_known & f.created.notna()
print("지연 고객 중 받기 전 작성:", round(f.loc[late, "before"].mean() * 100, 1), "%")
for b in (True, False):
    g = f[late & (f.before == b)]
    print(" 받기 전" if b else " 받은 뒤", "부정 비율:", round((g.review_group == "부정").mean() * 100, 1), "%", len(g))

data = {
    "dims": {"m": MONTHS, "c": CATS, "s": STATES, "p": PAYS, "b": BUCKETS, "r": REVIEWS},
    "themes": THEME_NAMES,
    "cube": cube[dims + ["n", "rep"]].values.tolist(),
    "repeaters": repeaters[dims + ["d"]].values.tolist(),
    "reviews": reviews[dims + ["sc", "t", "x", "ko"]].values.tolist(),
    "stateKo": [STATE_KO.get(x, x) for x in STATES],
}
path = os.path.join(OUT, "explorer.js")
with open(path, "w", encoding="utf-8") as fp:
    fp.write("window.OLIST=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
print(f"cube {len(cube)}행, 재구매 {len(repeaters)}명, 부정 리뷰 {len(reviews)}건, 고객 {cube.n.sum()}명")
print(f"{path}: {os.path.getsize(path) / 1024:.0f} KB")
