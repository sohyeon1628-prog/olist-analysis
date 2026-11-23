#!/usr/bin/env python3
"""사후 발견(탐색): 늦은 고객은 상품을 받기 전에 리뷰를 썼나 + 리뷰 요청 시점을 바꾸면 줄어드는 부정 리뷰 규모.

정의 (2026-11-23 바로잡음)
  - 리뷰를 쓴 시점 = review_answer_timestamp (고객이 답한 시각) < order_delivered_customer_date (도착 시각)
  - 참고: review_creation_date 는 Olist가 설문을 보낸 날. 예전 숫자(76.5%·77.2%·27.2%)는 이 날짜를 기준으로 계산돼
    '설문을 받은 시점'이었지 '리뷰를 쓴 시점'이 아니었음.
규모 추정은 '도착 전에 답한 리뷰가 도착 후에 답한 리뷰와 같은 부정 비율을 보였다면'이라는 단순 가정. 상한에 가까운 값.
    .venv/bin/python posthoc_review_timing.py   # → results/8_posthoc_review_timing.md
"""
import os

import pandas as pd

from analyze import RAW

BASE = os.path.dirname(os.path.abspath(__file__))
rv = pd.read_csv(os.path.join(RAW, "olist_order_reviews_dataset.csv"), parse_dates=["review_creation_date", "review_answer_timestamp"])
r = rv.sort_values("review_answer_timestamp").groupby("order_id").last()   # analyze.py 와 같은 규칙: 주문마다 가장 늦게 답한 리뷰


def timing(df, label):
    """df: order_id, delivered, late, score 가 있는 표 (도착일 있고 리뷰 있는 주문)."""
    d = df.join(r[["review_creation_date", "review_answer_timestamp"]], on="order_id")
    late = d[d.late]
    pre = late.review_answer_timestamp < late.delivered
    sent_pre = late.review_creation_date < late.delivered.dt.normalize()
    neg_pre, neg_post = (late[pre].score <= 2).mean(), (late[~pre].score <= 2).mean()
    avoid = pre.sum() * (neg_pre - neg_post)
    neg_all = (d.score <= 2).sum()
    avg_now = d.score.mean()
    # 도착 전 답한 리뷰의 별점 분포를 도착 후 분포로 바꿨을 때의 평균 별점
    avg_new = (d.score.sum() - late[pre].score.sum() + pre.sum() * late[~pre].score.mean()) / len(d)
    return [f"## {label}", "",
            f"- 대상: 도착일과 리뷰가 있는 주문 {len(d):,}건 중 늦게 도착 {len(late):,}건",
            f"- 늦은 주문 중 **고객이 도착 전에 답함 {pre.mean():.1%}** ({pre.sum():,}건) · 설문이 도착 전에 발송됨 {sent_pre.mean():.1%}",
            f"- 늦은 주문의 부정 리뷰(1~2점) 비율: 도착 전 답함 **{neg_pre:.1%}** vs 도착 후 답함 **{neg_post:.1%}**",
            f"- 규모 추정: 도착 전 답한 리뷰가 도착 후와 같은 부정 비율이었다면 부정 리뷰 약 **{avoid:,.0f}건** 감소 "
            f"(전체 부정 리뷰 {neg_all:,}건의 {avoid / neg_all:.1%}, 부정 비율 {neg_all / len(d):.1%} → {(neg_all - avoid) / len(d):.1%})",
            f"- 평균 별점: {avg_now:.2f} → 약 {avg_new:.2f}", ""]


# 1) 첫 구매 고객 (분석 대상, 2017-01 ~ 2018-02)
f = pd.read_csv(os.path.join(BASE, "results", "first_purchase_customers.csv"), parse_dates=["order_delivered_customer_date"])
f = f[f.delivery_known & f.score.notna()].rename(columns={"order_delivered_customer_date": "delivered"})
out = ["# 사후 발견: 리뷰를 쓴 시점과 부정 리뷰 (탐색적 분석)", "",
       "리뷰를 쓴 시점 = 고객이 답한 시각(review_answer_timestamp). 설문 발송일(review_creation_date)과 구분함.", ""]
out += timing(f[["order_id", "delivered", "late", "score"]], "첫 구매 고객 (2017-01 ~ 2018-02)")

# 2) 전체 주문 (Olist 데이터 전체 기간, 리뷰 요청은 모든 주문에 해당하는 운영 프로세스)
o = pd.read_csv(os.path.join(RAW, "olist_orders_dataset.csv"), parse_dates=["order_delivered_customer_date", "order_estimated_delivery_date"])
o = o.dropna(subset=["order_delivered_customer_date"]).rename(columns={"order_delivered_customer_date": "delivered"})
o["late"] = o.delivered.dt.normalize() > o.order_estimated_delivery_date.dt.normalize()   # analyze.py 와 같은 날짜 기준
o["score"] = o.order_id.map(r.review_score)
o = o.dropna(subset=["score"])
out += timing(o[["order_id", "delivered", "late", "score"]], "전체 주문 (2016-09 ~ 2018-10)")
out += ["주의: 도착 후에 답한 고객이 도착 전에 답한 고객과 원래 다를 수 있어(선택 편향) 실제 효과는 이보다 작을 수 있음. "
        "원인 확인은 리뷰 요청 시점 A/B 테스트로."]
p = os.path.join(BASE, "results", "8_posthoc_review_timing.md")
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out))
