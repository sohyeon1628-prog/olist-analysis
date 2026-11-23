#!/usr/bin/env python3
"""상품의 초기 평점이 이후 판매와 관계가 있는가 (탐색 + 검증). 규칙은 results/rating_sales/rules.md 에
검증 구간 결과를 보기 전에 확정했다.

    .venv/bin/python rating_sales.py   # → results/rating_sales/summary.md
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analyze import RAW

BASE = os.path.dirname(os.path.abspath(__file__))
END = pd.Timestamp("2018-08-31 23:59:59")
EXPLORE_END = pd.Timestamp("2017-08-31 23:59:59")
CONFIRM = (pd.Timestamp("2017-09-01"), pd.Timestamp("2018-04-30 23:59:59"))
rd = lambda n, **k: pd.read_csv(os.path.join(RAW, n + ".csv"), **k)

o = rd("olist_orders_dataset", parse_dates=["order_purchase_timestamp"])
it = rd("olist_order_items_dataset")
pr = rd("olist_products_dataset", usecols=["product_id", "product_category_name"])
tr = rd("product_category_name_translation")
rv = rd("olist_order_reviews_dataset", parse_dates=["review_answer_timestamp"]).sort_values("review_answer_timestamp").groupby("order_id").last().review_score

d = it[["order_id", "product_id"]].drop_duplicates().merge(o[["order_id", "order_purchase_timestamp"]], on="order_id")
d = d[d.order_purchase_timestamp <= END].sort_values(["product_id", "order_purchase_timestamp", "order_id"])
d["n"] = d.groupby("product_id").cumcount() + 1
d["single"] = d.order_id.map(it.groupby("order_id").product_id.nunique()) == 1
d["score"] = d.order_id.map(rv)
price = it.groupby("product_id").price.mean()
cat = pr.merge(tr, on="product_category_name", how="left").set_index("product_id").product_category_name_english

rows = []
for pid, g in d.groupby("product_id"):
    if len(g) < 10:
        continue
    first = g.iloc[:10]
    t10 = first.order_purchase_timestamp.iloc[-1]
    rated = first[first.single & first.score.notna()]
    if len(rated) < 5 or t10 + pd.Timedelta(days=120) > END:
        continue
    win = g[(g.order_purchase_timestamp >= t10 + pd.Timedelta(days=30)) & (g.order_purchase_timestamp <= t10 + pd.Timedelta(days=120))]
    rows.append({"product": pid, "t10": t10, "score": rated.score.mean(), "after": len(win),
                 "lprice": np.log(price.get(pid, np.nan)), "lspeed": np.log1p((t10 - first.order_purchase_timestamp.iloc[0]).days),
                 "cat": cat.get(pid, np.nan), "month": t10.strftime("%Y-%m")})
s = pd.DataFrame(rows).dropna(subset=["lprice"])


def fit(df, rhs):
    df = df.copy()
    vc = df.cat.value_counts()
    df["cat"] = df.cat.where(df.cat.map(vc) >= 15, "기타").fillna("기타")
    return smf.glm("after ~ " + rhs, data=df, family=sm.families.Poisson()).fit(cov_type="HC1")


def row(m, label):
    b, ci = m.params["score"], m.conf_int().loc["score"]
    return f"| {label} | {np.exp(b):.2f} | {np.exp(ci[0]):.2f} ~ {np.exp(ci[1]):.2f} | {m.pvalues['score']:.3f} |"


def tercile(df):
    q = pd.qcut(df.score.rank(method="first"), 3, labels=["낮은 1/3", "중간", "높은 1/3"])
    t = df.groupby(q, observed=True).agg(상품=("after", "size"), 평균별점=("score", "mean"), 이후주문=("after", "mean"))
    return [f"| {i} | {int(r.상품)} | {r.평균별점:.2f} | {r.이후주문:.1f} |" for i, r in t.iterrows()]


full = "score + lprice + lspeed + C(cat) + C(month)"
out = ["# 상품 평점과 이후 판매 (탐색 + 검증)", "", "규칙: `rules.md` (검증 구간 결과를 보기 전에 확정)",
       "Olist 별점은 만족도 설문이라 상품 페이지에 보이는 값인지 확인할 수 없음. '함께 움직이는가'만 본다.", ""]
for title, df, verdict in [("탐색 구간 (T10 ≤ 2017-08-31) — 판정하지 않음", s[s.t10 <= EXPLORE_END], False),
                           ("검증 구간 (T10 2017-09-01 ~ 2018-04-30) — **판정**", s[(s.t10 >= CONFIRM[0]) & (s.t10 <= CONFIRM[1])], True)]:
    m = fit(df, full)
    out += [f"## {title}", "", f"상품 {len(df):,}개, 이후 90일 평균 주문 {df.after.mean():.1f}건", "",
            "| 모델 | 평점 +1점당 이후 판매 비율 | 95% 구간 | p |", "| --- | --- | --- | --- |", row(m, "통제 변수 포함 (주 비교)"),
            row(fit(df, "score"), "단순 (참고)"), row(fit(df, "score + lprice + C(cat) + C(month)"), "초기 판매 속도 제외 (참고)"), "",
            "**초기 평점 3분위별 이후 판매 (참고)**", "", "| 구간 | 상품 | 평균 별점 | 이후 90일 평균 주문 |", "| --- | --- | --- | --- |", *tercile(df), ""]
    if verdict:
        b = m.params["score"]; p = m.pvalues["score"]
        out += ["**판정**: " + ("**지지** — 초기 평점이 낮은 상품이 이후 판매가 적음" if (b > 0 and p < 0.05) else "**확인되지 않음** (영향이 없다는 증명은 아님)"),
                f"- 비율 {np.exp(b):.2f}, p = {p:.3f}, 상품 {len(df)}개", ""]
out += ["참고: 관계일 뿐 원인이 아니다. 10번째 주문에 도달한 상품만 포함."]
open(os.path.join(BASE, "results", "rating_sales", "summary.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
