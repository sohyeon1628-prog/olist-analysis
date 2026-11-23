#!/usr/bin/env python3
"""판매자 지속 분석 (탐색 + 검증). 규칙은 results/seller/rules.md 에 결과를 보기 전에 확정했다.

    .venv/bin/python seller_retention.py   # → results/seller/summary.md
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analyze import RAW

BASE = os.path.dirname(os.path.abspath(__file__))
END = pd.Timestamp("2018-08-31 23:59:59")
EXPLORE_END = pd.Timestamp("2017-10-31 23:59:59")
CONFIRM = (pd.Timestamp("2017-11-01"), pd.Timestamp("2018-02-28 23:59:59"))

rd = lambda n, **k: pd.read_csv(os.path.join(RAW, n + ".csv"), **k)
o = rd("olist_orders_dataset", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
o = o[o.order_purchase_timestamp <= END]
rv = rd("olist_order_reviews_dataset", parse_dates=["review_answer_timestamp"]).sort_values("review_answer_timestamp").groupby("order_id").last()
o["score"] = o.order_id.map(rv.review_score)
o["late"] = np.where(o.order_delivered_customer_date.notna(),
                     (o.order_delivered_customer_date.dt.normalize() > o.order_estimated_delivery_date.dt.normalize()).astype(float), np.nan)
it = rd("olist_order_items_dataset")[["order_id", "seller_id"]].drop_duplicates()
d = it.merge(o[["order_id", "order_purchase_timestamp", "score", "late"]], on="order_id")
d = d.sort_values(["seller_id", "order_purchase_timestamp", "order_id"])
d["n"] = d.groupby("seller_id").cumcount() + 1

rows = []
for sid, g in d.groupby("seller_id"):
    if len(g) < 10:
        continue
    first = g.iloc[:10]
    t10 = first.order_purchase_timestamp.iloc[-1]
    rev, dl = first.score.dropna(), first.late.dropna()
    if len(rev) < 5 or len(dl) < 5:
        continue
    win = g[(g.order_purchase_timestamp >= t10 + pd.Timedelta(days=90)) & (g.order_purchase_timestamp <= t10 + pd.Timedelta(days=180))]
    rows.append({"seller": sid, "t10": t10, "neg": (rev <= 2).mean(), "late": dl.mean(),
                 "speed": np.log1p((t10 - first.order_purchase_timestamp.iloc[0]).days), "active": int(len(win) > 0),
                 "end_ok": t10 + pd.Timedelta(days=180) <= END})
s = pd.DataFrame(rows)
s = s[s.end_ok]


def fit(df, cols):
    X = sm.add_constant(df[cols].astype(float))
    m = sm.Logit(df.active, X).fit(disp=0)
    return m


def line(m, col, scale, label):
    b, ci = m.params[col] * scale, m.conf_int().loc[col] * scale
    return f"| {label} | {np.exp(b):.2f} | {np.exp(ci[0]):.2f} ~ {np.exp(ci[1]):.2f} | {m.pvalues[col]:.3f} |"


def tercile(df, col):
    q = pd.qcut(df[col].rank(method="first"), 3, labels=["낮은 1/3", "중간", "높은 1/3"])
    t = df.groupby(q, observed=True).agg(판매자=("active", "size"), 지속률=("active", "mean"), 평균=(col, "mean"))
    return [f"| {i} | {int(r.판매자)} | {r.평균:.1%} | {r.지속률:.1%} |" for i, r in t.iterrows()]


out = ["# 판매자 지속 분석 결과 (탐색 + 검증)", "", "규칙: `rules.md` (검증 구간 결과를 보기 전에 확정)", ""]
for title, df, verdict in [("탐색 구간 (T10 ≤ 2017-10-31) — 판정하지 않음", s[s.t10 <= EXPLORE_END], False),
                           ("검증 구간 (T10 2017-11-01 ~ 2018-02-28) — **판정**", s[(s.t10 >= CONFIRM[0]) & (s.t10 <= CONFIRM[1])], True)]:
    out += [f"## {title}", "", f"판매자 {len(df):,}명, 이후 지속 {df.active.mean():.1%}", ""]
    m = fit(df, ["neg"]); df = df.assign(neg10=df.neg * 10)
    m = fit(df, ["neg10"])
    out += ["| 변수 (지속 ~) | 오즈비 | 95% 구간 | p |", "| --- | --- | --- | --- |", line(m, "neg10", 1, "부정 리뷰율 +10%p (주 비교)")]
    m2 = fit(df.assign(late10=df.late * 10), ["late10"]); out.append(line(m2, "late10", 1, "지연율 +10%p (참고)"))
    m3 = fit(df.assign(late10=df.late * 10), ["neg10", "speed"]); out.append(line(m3, "neg10", 1, "부정 리뷰율 +10%p, 속도 통제 (참고)"))
    out += ["", "**부정 리뷰율 3분위별 지속률 (참고)**", "", "| 구간 | 판매자 | 평균 부정 리뷰율 | 지속률 |", "| --- | --- | --- | --- |", *tercile(df, "neg"), ""]
    if verdict:
        p, orr = m.pvalues["neg10"], np.exp(m.params["neg10"])
        sd = df.neg.std() * 10
        # 검출 가능한 최소 효과 (80% 검정력, 양측 5%) — 단순 근사
        pbar = df.active.mean(); mde = (1.96 + 0.84) / np.sqrt(len(df) * pbar * (1 - pbar) * (sd / 10) ** 2) * 10 / 10
        out += [f"**판정**: " + ("**지지** — 초기 부정 리뷰가 많은 판매자가 덜 지속" if (orr < 1 and p < 0.05) else "**확인되지 않음** (차이가 없다는 증명은 아님)"),
                f"- 오즈비 {orr:.2f}, p = {p:.3f}, 표본 {len(df)}명",
                f"- 80% 검정력으로 잡을 수 있는 최소 효과: 부정 리뷰율 1표준편차(약 {df.neg.std():.0%}p) 증가당 로그 오즈 약 {mde * df.neg.std():.2f} (오즈비 약 {np.exp(-abs(mde * df.neg.std())):.2f} 이하)", ""]
out += ["참고: 관계일 뿐 원인이 아니다. 10번째 주문에 도달한 판매자만 포함."]
p_ = os.path.join(BASE, "results", "seller", "summary.md")
open(p_, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
