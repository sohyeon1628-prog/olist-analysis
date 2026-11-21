#!/usr/bin/env python3
"""감성 모델 결과 정리 (탐색적 분석, 계획서 v1.2 밖).

입력: results/sentiment/{gold_model.csv, olist_text_sentiment.csv, b2w_val.json}, results/first_purchase_customers.csv
출력: results/sentiment/summary.md
    .venv/bin/python sentiment_analysis.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from analyze import RAW

BASE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(BASE, "results", "sentiment")
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


out = ["# 리뷰 글 감성 분석 결과 (탐색적 분석)", "",
       "계획서 v1.2에 없던 분석. 설정은 `model_setup.md`, 정답 기준은 `guideline.md`에 결과를 보기 전에 정했다.", ""]

# 1) 모델 정확도
b2w = json.load(open(os.path.join(SD, "b2w_val.json")))
g = pd.read_csv(os.path.join(SD, "gold_model.csv"))
e = g[g.gold.isin(["부정", "긍정"])]
e_no3 = e[e.star != 3]
star_pred = np.where(e_no3.star >= 4, "긍정", "부정")
out += ["## 1. 모델 정확도", "",
        "| 기준 | 대상 | 정확도 |", "| --- | --- | --- |",
        f"| B2W 검증 (별점 기준, 참고) | {b2w['n_val']:,}건 | {b2w['b2w_val_accuracy']:.1%} |",
        f"| 정답 세트, 글이 부정·긍정인 리뷰 | {len(e)}건 | **{(e.model == e.gold).mean():.1%}** |",
        f"| 같은 리뷰에서 ★3 제외 — 모델 | {len(e_no3)}건 | {(e_no3.model == e_no3.gold).mean():.1%} |",
        f"| 같은 리뷰에서 ★3 제외 — 별점(1~2 부정, 4~5 긍정) | {len(e_no3)}건 | {(star_pred == e_no3.gold).mean():.1%} |",
        f"| ★3 리뷰만 — 모델 (별점으로는 판정 불가) | {(e.star == 3).sum()}건 | {(e[e.star == 3].model == e[e.star == 3].gold).mean():.1%} |", "",
        f"- 부정 재현율 {(e[e.gold == '부정'].model == '부정').mean():.1%}, 긍정 재현율 {(e[e.gold == '긍정'].model == '긍정').mean():.1%}",
        f"- 혼합 {int((g.gold == '혼합').sum())}건 중 모델 부정 {(g[g.gold == '혼합'].model == '부정').mean():.0%}, "
        f"중립 {int((g.gold == '중립').sum())}건 중 모델 부정 {(g[g.gold == '중립'].model == '부정').mean():.0%} — 두 가지로만 나누는 모델의 한계",
        "- 정답 세트는 Claude가 별점을 보지 않고 판정. 사람(박소현) 50건 확인 일치율 86.7%, 카파 0.77", ""]

# 2) Olist 전체: 별점과 글이 어긋나는 비율
sent = pd.read_csv(os.path.join(SD, "olist_text_sentiment.csv"))
p_of = dict(zip(sent.text, sent.p_pos))
s = open(os.path.join(BASE, "docs", "data", "reviews.js"), encoding="utf-8").read()
rows = pd.DataFrame(json.loads(s[s.index("["):s.rindex("]") + 1]), columns=["order", "star", "text", "ko"])
rows["p"] = rows.text.map(lambda t: p_of.get(norm(t), np.nan))
rows = rows.dropna(subset=["p"])
rows["neg"] = rows.p < .5
mm = rows.groupby("star").neg.agg(["size", "mean"])
out += ["## 2. 별점과 글 감성이 어긋나는 비율 (Olist 글 리뷰 전체)", "",
        "| 별점 | 글 리뷰 | 모델이 부정으로 판정 |", "| --- | --- | --- |"]
out += [f"| ★{st} | {int(r['size']):,} | {r['mean']:.1%} |" for st, r in mm.iterrows()]
hi, lo = rows[rows.star >= 4], rows[rows.star <= 2]
out += ["", f"- ★4~5인데 글이 부정: **{hi.neg.mean():.1%}** ({int(hi.neg.sum()):,} / {len(hi):,}건)",
        f"- ★1~2인데 글이 긍정: **{(~lo.neg).mean():.1%}** ({int((~lo.neg).sum()):,} / {len(lo):,}건)",
        "- 모델 오차(정답 세트 기준 약 2%)가 섞인 추정치", ""]

# 3) 첫 구매 고객: 글 감성별 재구매율 (기존 결과 ③을 글 기준으로 다시)
f = pd.read_csv(os.path.join(BASE, "results", "first_purchase_customers.csv"))
rv = pd.read_csv(os.path.join(RAW, "olist_order_reviews_dataset.csv"))
r = rv.sort_values("review_answer_timestamp").groupby("order_id").last()
f["text"] = f.order_id.map(r.review_comment_message).fillna("").map(norm)
f["p"] = f.text.map(lambda t: p_of.get(t, np.nan) if t else np.nan)
t = f.dropna(subset=["p"]).copy()
t["text_group"] = np.where(t.p < .5, "부정", "긍정")
t["star_group"] = t.review_group


def rate_table(col, groups):
    lines = []
    for gname in groups:
        x = t[t[col] == gname].repeat
        lo_, hi_ = wilson(int(x.sum()), len(x))
        lines.append(f"| {gname} | {len(x):,} | {int(x.sum())} | {x.mean():.2%} | {lo_:.2%} ~ {hi_:.2%} |")
    return lines


def pval(col, a, b):
    sub = t[t[col].isin([a, b])]
    return chi2_contingency(pd.crosstab(sub[col], sub.repeat), correction=False)[1]


out += ["## 3. 첫 구매 고객: 글 감성별 180일 재구매율", "",
        f"대상: 첫 구매 고객 중 리뷰에 글을 남긴 {len(t):,}명 (전체 {len(f):,}명). 기존 결과 ③은 별점 기준 전체 고객.", "",
        "**글 감성 기준 (모델)**", "", "| 그룹 | 고객 | 재구매 | 재구매율 | 95% 구간 |", "| --- | --- | --- | --- | --- |"]
out += rate_table("text_group", ["부정", "긍정"])
out += ["", f"부정 vs 긍정: p = {pval('text_group', '부정', '긍정'):.3f}", "",
        "**같은 고객을 별점 기준으로 (비교용)**", "", "| 그룹 | 고객 | 재구매 | 재구매율 | 95% 구간 |", "| --- | --- | --- | --- | --- |"]
out += rate_table("star_group", ["부정", "중립", "긍정"])
out += ["", f"부정 vs 긍정: p = {pval('star_group', '부정', '긍정'):.3f}", "",
        "- 판정 규칙은 계획서와 같게: p < 0.05 이고 그룹마다 재구매 고객 10명 이상일 때만 '차이 있음'",
        "- 관계일 뿐 원인은 아님", ""]

open(os.path.join(SD, "summary.md"), "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
