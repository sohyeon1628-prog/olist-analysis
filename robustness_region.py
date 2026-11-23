#!/usr/bin/env python3
"""보조 확인: 지역(주)을 넣어도 ②③ 결론이 같은가 (2026-11-23, 계획서 v1.2 밖).

analyze.py ⑥ 로지스틱 회귀에 고객 주를 추가한다.
  A: 기존 식 + 주
  B: 리뷰 그룹을 뺀 식 + 주 — 리뷰는 지연의 결과(매개)이기도 해서, 지연의 '전체' 관계를 보려면 빼야 함
고객 300명 미만인 주는 '기타'로 묶는다.
    .venv/bin/python robustness_region.py   # → results/7_robustness_region.md
"""
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analyze import RAW

BASE = os.path.dirname(os.path.abspath(__file__))
f = pd.read_csv(os.path.join(BASE, "results", "first_purchase_customers.csv"))
cust = pd.read_csv(os.path.join(RAW, "olist_customers_dataset.csv"), usecols=["customer_id", "customer_state"])
f = f.merge(cust, on="customer_id", how="left")
dk = f[f.delivery_known].copy()
big = dk.customer_state.value_counts()
dk["state"] = dk.customer_state.where(dk.customer_state.map(big) >= 300, "기타")
reg = dk.assign(y=dk.repeat.astype(int), late_i=dk.late.astype(int), lpay=np.log1p(dk.payment.fillna(0)))

base = "C(cat, Treatment('기타')) + C(month) + lpay"
models = {
    "기존 (analyze.py ⑥)": f"y ~ late_i + C(review_group, Treatment('긍정')) + {base}",
    "A: 기존 + 지역": f"y ~ late_i + C(review_group, Treatment('긍정')) + {base} + C(state, Treatment('SP'))",
    "B: 리뷰 빼고 + 지역": f"y ~ late_i + {base} + C(state, Treatment('SP'))",
}
rows = []
for name, fml in models.items():
    m = smf.logit(fml, data=reg).fit(disp=0, maxiter=200)
    ci = m.conf_int()
    for k, label in [("late_i", "배송 지연"), ("C(review_group, Treatment('긍정'))[T.부정]", "부정 리뷰")]:
        if k in m.params:
            rows.append(f"| {name} | {label} | {np.exp(m.params[k]):.2f} | {np.exp(ci.loc[k, 0]):.2f} ~ {np.exp(ci.loc[k, 1]):.2f} | {m.pvalues[k]:.3f} |")
    nobs = int(m.nobs)

by_state = dk.groupby("state").agg(고객=("late", "size"), 지연율=("late", "mean"), 재구매율=("repeat", "mean")).sort_values("지연율")
md = ["# 보조 확인: 지역을 넣은 로지스틱 회귀 (계획서 v1.2 밖)", "",
      "재구매 ~ 배송 지연 (+ 리뷰 그룹) + 첫 구매 카테고리 + 첫 구매 월 + 결제 금액(로그) (+ 고객 주, 기준 SP)", "",
      "| 모델 | 변수 | 오즈비 | 95% 신뢰구간 | p |", "| --- | --- | --- | --- | --- |", *rows, "",
      f"관측 {nobs:,}명. 고객 300명 미만인 주는 '기타'로 묶음.", "",
      "## 주별 지연율과 재구매율", "", "| 주 | 고객 | 지연율 | 재구매율 |", "| --- | --- | --- | --- |"]
md += [f"| {s} | {int(r['고객']):,} | {r['지연율']:.1%} | {r['재구매율']:.2%} |" for s, r in by_state.iterrows()]
out = os.path.join(BASE, "results", "7_robustness_region.md")
open(out, "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md[:12]))
