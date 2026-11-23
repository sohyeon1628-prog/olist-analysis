#!/usr/bin/env python3
"""묶음 주문(여러 상품·여러 판매자)의 부정 리뷰 글에서 어떤 불만이 많은가 (탐색, 계획서 v1.2 밖).

가설 (결과를 보기 전에 적음): 묶음 주문은 정시에 도착해도 부정 리뷰가 많은데, 그 이유는 '일부만 받았다'는 불만이다.
  - 단어 사전은 analyze.py의 THEMES를 그대로 쓴다 (새로 만들지 않음).
  - 비교: 부정 리뷰(1~2점) 중 글이 있는 것에서, 주문 유형별 '일부 누락' 주제 비율.
  - 가설을 뒷받침한다고 보는 기준: 묶음 주문 유형의 '일부 누락' 비율이 상품 1개 주문보다 높고, 95% 구간이 겹치지 않을 때.
    그 외에는 '뒷받침하지 못함'. 다른 주제 비율은 참고로만 보고 판정하지 않는다.
    .venv/bin/python multi_order_themes.py   # → results/9_multi_order_themes.md
"""
import os

import numpy as np
import pandas as pd

from analyze import RAW, THEMES, theme_hits

BASE = os.path.dirname(os.path.abspath(__file__))
rd = lambda n, **k: pd.read_csv(os.path.join(RAW, n + ".csv"), **k)

rv = rd("olist_order_reviews_dataset", parse_dates=["review_answer_timestamp"]).sort_values("review_answer_timestamp").groupby("order_id").last()
o = rd("olist_orders_dataset", parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"])
it = rd("olist_order_items_dataset")
a = it.groupby("order_id").agg(items=("order_item_id", "size"), products=("product_id", "nunique"), sellers=("seller_id", "nunique"))
x = o.join(a, on="order_id").dropna(subset=["items"])
x["score"] = x.order_id.map(rv.review_score)
x["text"] = x.order_id.map(rv.review_comment_message).fillna("").str.strip()
x["late"] = x.order_delivered_customer_date.dt.normalize() > x.order_estimated_delivery_date.dt.normalize()


def kind(r):
    if r["items"] == 1:
        return "1) 상품 1개"
    if r.sellers > 1:
        return "4) 여러 판매자"
    return "2) 같은 상품 여러 개" if r.products == 1 else "3) 한 판매자, 다른 상품 여러 개"


x["kind"] = x.apply(kind, axis=1)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


out = ["# 묶음 주문의 부정 리뷰 불만 주제 (탐색)", "",
       "부정 리뷰(1~2점) 중 글이 있는 것, analyze.py의 단어 사전(THEMES) 그대로. 한 리뷰가 여러 주제에 걸릴 수 있음.", ""]
for label, df in [("전체", x), ("정시 도착 주문만", x[~x.late.fillna(False) & x.order_delivered_customer_date.notna()])]:
    neg = df[(df.score <= 2) & df.text.ne("")].copy()
    hits = theme_hits(neg.text)
    neg = pd.concat([neg.reset_index(drop=True), hits.reset_index(drop=True)], axis=1)
    out += [f"## {label}", "", "| 주문 유형 | 글 있는 부정 리뷰 | " + " | ".join(THEMES) + " |", "| --- | --- | " + " | ".join(["---"] * len(THEMES)) + " |"]
    rows = {}
    for k, g in neg.groupby("kind"):
        cells = []
        for th in THEMES:
            c, n = int(g[th].sum()), len(g); lo, hi = wilson(c, n)
            cells.append(f"{c / n:.1%} ({lo:.0%}~{hi:.0%})"); rows[(k, th)] = (c / n, lo, hi)
        out.append(f"| {k} | {len(g):,} | " + " | ".join(cells) + " |")
    out.append("")
    if label == "전체":
        base = rows[("1) 상품 1개", "일부 누락")]
        out.append("**가설 판정 (일부 누락)**")
        for k in ["2) 같은 상품 여러 개", "3) 한 판매자, 다른 상품 여러 개", "4) 여러 판매자"]:
            r = rows[(k, "일부 누락")]
            ok = r[0] > base[0] and r[1] > base[2]
            out.append(f"- {k}: {r[0]:.1%} vs 상품 1개 {base[0]:.1%} → " + ("뒷받침 (구간이 겹치지 않음)" if ok else "뒷받침하지 못함"))
        out.append("")
out += ["참고: 단어 사전은 포르투갈어 원문에 적용한 단순 규칙이라 놓치거나 잘못 잡는 경우가 있음(결과 요약의 한계 참고). 관계일 뿐 원인이 아님."]
open(os.path.join(BASE, "results", "9_multi_order_themes.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
