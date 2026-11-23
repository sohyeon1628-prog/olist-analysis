#!/usr/bin/env python3
"""분석 이야기 01·06·08에 쓰인 숫자를 재현한다 (탐색, 계획서 v1.2 밖).

  01 질문 재정의: 리뷰 글의 마켓플레이스 이름 언급, 재구매 시 같은 판매자 비율
  06 묶음 주문: 주문 유형별 부정 리뷰 비율 (전체 / 배송 완료 + 정시 도착만)
  08 세일 시기: 블랙프라이데이 주간 비교, 월별 지연율·부정 리뷰율

기준 (analyze.py와 같은 날짜 기준)
  - 리뷰: 주문마다 가장 늦게 답한 리뷰, 부정 = 1~2점
  - 지연: 도착일(날짜)이 예정일(날짜)보다 늦음. 지연율·'정시' 비교는 도착일이 있는 주문만
  - 부정 리뷰 비율은 리뷰가 있는 주문 기준
    .venv/bin/python structure_evidence.py   # → results/10_structure_evidence.md
"""
import os

import numpy as np
import pandas as pd

from analyze import RAW

BASE = os.path.dirname(os.path.abspath(__file__))
rd = lambda n, **k: pd.read_csv(os.path.join(RAW, n + ".csv"), **k)

o = rd("olist_orders_dataset", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
rv_all = rd("olist_order_reviews_dataset", parse_dates=["review_answer_timestamp"])
rv = rv_all.sort_values("review_answer_timestamp").groupby("order_id").last()
o["score"] = o.order_id.map(rv.review_score)
o["has_d"] = o.order_delivered_customer_date.notna()
o["late"] = o.has_d & (o.order_delivered_customer_date.dt.normalize() > o.order_estimated_delivery_date.dt.normalize())
it = rd("olist_order_items_dataset")
out = ["# 분석 이야기 01·06·08 숫자 재현 (탐색)", ""]

# ---------- 01 ----------
t = rv.review_comment_message.fillna("").str.strip().str.lower()
t = t[t != ""]
named = t.str.contains("lannister|stark|targaryen|baratheon")
cust = rd("olist_customers_dataset", usecols=["customer_id", "customer_unique_id"])
oo = o.merge(cust, on="customer_id")
oo = oo[oo.order_id.isin(it.order_id)].sort_values(["order_purchase_timestamp", "order_id"])
oo["rk"] = oo.groupby("customer_unique_id").cumcount()
sel = oo.groupby("customer_unique_id").filter(lambda d: len(d) >= 2).groupby("customer_unique_id").head(2)
first, second = sel[sel.rk == 0].set_index("customer_unique_id").order_id, sel[sel.rk == 1].set_index("customer_unique_id").order_id
sellers = it.groupby("order_id").seller_id.agg(set)
same = np.mean([len(sellers[first[u]] & sellers[second[u]]) > 0 for u in first.index])
n_cust = oo.customer_unique_id.nunique()
out += ["## 01 질문 재정의", "",
        f"- 글이 있는 리뷰(주문당 1건) {len(t):,}건: 'olist' 언급 {int(t.str.contains('olist').sum())}건, 마켓플레이스 이름 언급 {int(named.sum()):,}건 ({named.mean():.1%})",
        f"- 주문이 있는 고객 {n_cust:,}명 중 2회 이상 주문 {len(first):,}명 ({len(first) / n_cust:.1%})",
        f"- 그 고객들의 두 번째 주문이 첫 주문과 같은 판매자를 포함: {same:.1%}", ""]

# ---------- 06 ----------
a = it.groupby("order_id").agg(items=("order_item_id", "size"), products=("product_id", "nunique"), sellers=("seller_id", "nunique"))
x = o.join(a, on="order_id").dropna(subset=["items"])


def kind(r):
    if r["items"] == 1:
        return "1) 상품 1개"
    if r.sellers > 1:
        return "4) 여러 판매자"
    return "2) 같은 상품 여러 개" if r.products == 1 else "3) 한 판매자, 다른 상품 여러 개"


x["kind"] = x.apply(kind, axis=1)
xr = x.dropna(subset=["score"]).copy()
xr["neg"] = xr.score <= 2
g = xr.groupby("kind").agg(주문=("neg", "size"), 부정=("neg", "mean"))
g["주문 비중"] = g.주문 / g.주문.sum()
g["부정 리뷰 중 비중"] = xr.groupby("kind").neg.sum() / xr.neg.sum()
xd = xr[xr.has_d]
g["지연율(도착한 주문)"] = xd.groupby("kind").late.mean()
ont = xd[~xd.late]
g["부정(정시 도착만)"] = ont.groupby("kind").neg.mean()
out += ["## 06 묶음 주문", "", f"리뷰가 있는 주문 {len(xr):,}건 (배송 완료 {len(xd):,}건, 그중 정시 {len(ont):,}건)", "",
        "| 유형 | 주문 | 주문 비중 | 부정 리뷰 비율 | 부정 리뷰 중 비중 | 지연율(도착한 주문) | 부정(정시 도착만) |", "| --- | --- | --- | --- | --- | --- | --- |"]
out += [f"| {k} | {int(r.주문):,} | {r['주문 비중']:.1%} | {r.부정:.1%} | {r['부정 리뷰 중 비중']:.1%} | {r['지연율(도착한 주문)']:.1%} | {r['부정(정시 도착만)']:.1%} |" for k, r in g.iterrows()]
multi = ~xr.kind.str.startswith("1")
out += ["", f"- 상품이 여러 개인 주문: 주문의 {multi.mean():.1%}, 부정 리뷰의 {xr[multi].neg.sum() / xr.neg.sum():.1%}", ""]

# ---------- 08 ----------
d = o[o.has_d & o.score.notna()].copy()
d["neg"] = d.score <= 2
w = lambda s, e: d[(d.order_purchase_timestamp >= s) & (d.order_purchase_timestamp < e)]
b, bf = w("2017-11-01", "2017-11-20"), w("2017-11-20", "2017-11-28")
out += ["## 08 세일 시기", "", "배송 완료된 주문 중 리뷰가 있는 주문 기준.", "",
        "| 기간 | 주문 | 늦게 도착 | 부정 리뷰 |", "| --- | --- | --- | --- |",
        f"| 2017-11-01 ~ 19 | {len(b):,} | {b.late.mean():.1%} | {b.neg.mean():.1%} |",
        f"| 2017-11-20 ~ 27 (블랙프라이데이 주간) | {len(bf):,} | {bf.late.mean():.1%} | {bf.neg.mean():.1%} |", "",
        "| 월 | 주문 | 늦게 도착 | 부정 리뷰 |", "| --- | --- | --- | --- |"]
d["m"] = d.order_purchase_timestamp.dt.to_period("M")
mm = d.groupby("m").agg(n=("late", "size"), late=("late", "mean"), neg=("neg", "mean"))
out += [f"| {m} | {int(r.n):,} | {r.late:.1%} | {r.neg:.1%} |" for m, r in mm[(mm.index >= "2017-10") & (mm.index <= "2018-06")].iterrows()]
open(os.path.join(BASE, "results", "10_structure_evidence.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
