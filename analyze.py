#!/usr/bin/env python3
"""Olist 첫 구매 고객 재구매 분석 — 계획서 규칙 v1.2를 그대로 코드로 옮긴 것.

실행: ../shortform-analysis/.venv/bin/python analyze.py
입력: data/raw/ (캐글 olistbr/brazilian-ecommerce CSV 9개)
출력: results/ (summary.md, 표 CSV, 그래프 PNG, 불만 주제 표본)
"""
import os
import re
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.stats import chi2_contingency, fisher_exact  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

RULES = "v1.2"
BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

# ---- 규칙 (계획서와 같아야 함) ----
START, END = pd.Timestamp("2017-01-01"), pd.Timestamp("2018-03-01")   # 첫 구매 기간 [START, END)
REPEAT_MIN, REPEAT_MAX = pd.Timedelta(hours=24), pd.Timedelta(days=180)
MIN_GROUP, MIN_REPEATERS, MIN_THEME = 200, 10, 30
CAT_MIN_FIRST = 2000
ALPHA, SEED, THEME_SAMPLE = 0.05, 42, 20
THEMES = {
    "배송 지연": ["atras", "demor"],
    "미수령": ["nao recebi", "nao chegou", "nao foi entregue", "nao entregaram"],
    "품질·파손": ["pessima qualidade", "baixa qualidade", "ma qualidade", "qualidade ruim",
                "quebrad", "defeito", "danificad", "rasgad"],
    "설명과 다름·오배송": ["diferente", "errad", "nao corresponde", "veio outro", "trocad"],
    "일부 누락": ["faltando", "falt", "incomplet", "so recebi", "apenas um"],
}
CAT_KO = {
    "bed_bath_table": "침구·욕실", "sports_leisure": "스포츠·레저", "health_beauty": "뷰티·헬스",
    "computers_accessories": "컴퓨터·액세서리", "furniture_decor": "가구·장식", "housewares": "생활용품",
    "watches_gifts": "시계·선물", "toys": "장난감", "telephony": "휴대폰", "cool_stuff": "쿨스터프",
    "garden_tools": "정원용품",
}
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
C_BAD, C_OK, C_MID, C_GRID, C_INK, C_MUTED = "#D9480F", "#1C7ED6", "#ADB5BD", "#E9ECEF", "#212529", "#868E96"


def norm_text(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip()


def build():
    rd = lambda f, **k: pd.read_csv(os.path.join(RAW, f), **k)  # noqa: E731
    o = rd("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                                                     "order_estimated_delivery_date"])
    c = rd("olist_customers_dataset.csv")
    it = rd("olist_order_items_dataset.csv")
    p = rd("olist_products_dataset.csv")
    t = rd("product_category_name_translation.csv")
    rv = rd("olist_order_reviews_dataset.csv", parse_dates=["review_answer_timestamp"])
    pay = rd("olist_order_payments_dataset.csv")

    d = o[o.order_status == "delivered"].merge(c[["customer_id", "customer_unique_id"]], on="customer_id")
    d = d.sort_values(["order_purchase_timestamp", "order_id"])
    first = d.groupby("customer_unique_id").first().reset_index()
    first = first[(first.order_purchase_timestamp >= START) & (first.order_purchase_timestamp < END)].copy()

    # 재구매: 첫 주문 + 24시간 이후 ~ 180일 이내의 배송 완료 주문
    m = d[["customer_unique_id", "order_purchase_timestamp"]].merge(
        first[["customer_unique_id", "order_purchase_timestamp"]], on="customer_unique_id", suffixes=("", "_first"))
    gap = m.order_purchase_timestamp - m.order_purchase_timestamp_first
    rep = m[(gap >= REPEAT_MIN) & (gap <= REPEAT_MAX)]
    rep_first_gap = (rep.assign(gap=gap[rep.index]).groupby("customer_unique_id").gap.min())
    first["repeat"] = first.customer_unique_id.isin(rep_first_gap.index)
    first["days_to_repeat"] = first.customer_unique_id.map(rep_first_gap).dt.total_seconds() / 86400

    # 배송 지연 (날짜 기준)
    dl = first.order_delivered_customer_date.dt.normalize()
    est = first.order_estimated_delivery_date.dt.normalize()
    first["delivery_known"] = dl.notna()
    first["late"] = dl > est
    first["delay_days"] = (dl - est).dt.days
    first["delay_bucket"] = pd.cut(first.delay_days, [-10**4, 0, 3, 7, 10**4],
                                   labels=["정시", "1~3일", "4~7일", "8일 이상"])

    # 리뷰: 답변 시각이 가장 늦은 것
    r = rv.sort_values("review_answer_timestamp").groupby("order_id").last()
    first["score"] = first.order_id.map(r.review_score)
    first["review_group"] = np.select([first.score <= 2, first.score == 3, first.score >= 4],
                                      ["부정", "중립", "긍정"], default="리뷰 없음")
    first["text"] = first.order_id.map(r.review_comment_message).fillna("").astype(str).str.strip()
    first["review_id"] = first.order_id.map(r.review_id)

    # 카테고리: 첫 주문의 첫 번째 상품
    items = it[it.order_item_id == 1].merge(p[["product_id", "product_category_name"]], on="product_id")
    en = dict(zip(t.product_category_name, t.product_category_name_english))
    cat = items.set_index("order_id").product_category_name.map(lambda x: en.get(x, x))
    first["cat_en"] = first.order_id.map(cat)
    first["cat"] = first.cat_en.map(lambda x: "카테고리 없음" if pd.isna(x) else CAT_KO.get(x, "기타"))
    first["payment"] = first.order_id.map(pay.groupby("order_id").payment_value.sum())
    first["month"] = first.order_purchase_timestamp.dt.strftime("%Y-%m")
    return first


def rate_row(label, s):
    s = s.dropna().astype(bool)
    n, k = len(s), int(s.sum())
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return {"그룹": label, "n": n, "해당": k, "비율": k / n if n else np.nan, "CI_low": lo, "CI_high": hi}


def two_group_test(a, b):
    """a, b: bool Series. 카이제곱(연속성 보정 없음), 기대빈도 <5면 Fisher."""
    a, b = a.dropna().astype(bool), b.dropna().astype(bool)
    tab = np.array([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]])
    chi2, p, _, exp = chi2_contingency(tab, correction=False)
    method = "카이제곱"
    if (exp < 5).any():
        _, p = fisher_exact(tab)
        method = "Fisher"
    return p, method


def verdict(rows, p, repeat_outcome):
    if any(r["n"] < MIN_GROUP for r in rows):
        return "결론 없음 (그룹 200명 미만)"
    if repeat_outcome and any(r["해당"] < MIN_REPEATERS for r in rows):
        return "표본 부족 (재구매 10명 미만)"
    return "의미 있는 차이" if p < ALPHA else "차이 없음 (우연 범위)"


def pct(x, d=1):
    return "-" if pd.isna(x) else f"{x * 100:.{d}f}%"


def md_rates(rows):
    out = ["| 그룹 | 고객 | 해당 | 비율 | 95% 신뢰구간 |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r['그룹']} | {r['n']:,} | {r['해당']:,} | {pct(r['비율'], 2)} | "
                   f"{pct(r['CI_low'], 2)} ~ {pct(r['CI_high'], 2)} |")
    return "\n".join(out)


def compare_two(df, flag, outcome, labels, repeat_outcome):
    a, b = df[df[flag]], df[~df[flag]]
    rows = [rate_row(labels[0], a[outcome]), rate_row(labels[1], b[outcome])]
    p, method = two_group_test(a[outcome], b[outcome])
    diff = rows[0]["비율"] - rows[1]["비율"]
    ratio = rows[0]["비율"] / rows[1]["비율"] if rows[1]["비율"] else np.nan
    return rows, {"p": p, "method": method, "diff": diff, "ratio": ratio,
                  "verdict": verdict(rows, p, repeat_outcome)}


def theme_hits(texts):
    nt = texts.map(norm_text)
    return pd.DataFrame({th: nt.map(lambda s, ws=ws: any(w in s for w in ws)) for th, ws in THEMES.items()})


# ---------------- 그래프 ----------------
def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)


def chart_flow(neg, rep, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=200)
    for ax, rows, title in [(axes[0], neg, "부정 리뷰(1~2점) 비율"), (axes[1], rep, "180일 안 재구매율")]:
        vals = [r["비율"] * 100 for r in rows]
        err = [[(r["비율"] - r["CI_low"]) * 100 for r in rows], [(r["CI_high"] - r["비율"]) * 100 for r in rows]]
        bars = ax.bar([r["그룹"] for r in rows], vals, color=[C_BAD, C_OK], width=0.55, yerr=err,
                      error_kw={"ecolor": C_INK, "elinewidth": 1, "capsize": 4})
        for b, v, r in zip(bars, vals, rows):
            ax.text(b.get_x() + b.get_width() / 2, r["CI_high"] * 100 + max(vals) * 0.04, f"{v:.1f}%\n(n={r['n']:,})",
                    ha="center", fontsize=8.5, color=C_INK)
        ax.set_title(title, fontsize=10, color=C_INK, loc="left")
        ax.set_ylim(0, max(vals) * 1.45)
        style(ax)
    fig.suptitle("첫 주문 배송이 늦은 고객 vs 제때 받은 고객", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.01, "막대 위 선 = 95% 신뢰구간 · 2017-01~2018-02 첫 구매 고객 · Olist", fontsize=7.5, color=C_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def chart_rates(rows, title, path, colors):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    vals = [r["비율"] * 100 for r in rows]
    err = [[(r["비율"] - r["CI_low"]) * 100 for r in rows], [(r["CI_high"] - r["비율"]) * 100 for r in rows]]
    bars = ax.bar([r["그룹"] for r in rows], vals, color=colors, width=0.55, yerr=err,
                  error_kw={"ecolor": C_INK, "elinewidth": 1, "capsize": 4})
    for b, v, r in zip(bars, vals, rows):
        ax.text(b.get_x() + b.get_width() / 2, r["CI_high"] * 100 + max(vals) * 0.04, f"{v:.2f}%\n(n={r['n']:,})",
                ha="center", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.45)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_themes(tab, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    y = np.arange(len(tab))[::-1]
    ax.barh(y + 0.18, tab["지연 고객"] * 100, height=0.34, color=C_BAD, label="첫 주문 지연 고객")
    ax.barh(y - 0.18, tab["정시 고객"] * 100, height=0.34, color=C_OK, label="제때 받은 고객")
    ax.set_yticks(y)
    ax.set_yticklabels(tab.index, fontsize=9)
    ax.set_xlabel("부정 리뷰(글 있음) 중 해당 주제 비율 (%)", fontsize=8.5, color=C_MUTED)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("부정 리뷰는 무엇에 불만인가 (미리 정한 단어 사전 기준)", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    f = build()
    md = [f"# Olist 첫 구매 고객 분석 결과 (규칙 {RULES})", "",
          f"분석 대상: 2017-01 ~ 2018-02 첫 구매 고객 {len(f):,}명. "
          f"배송일 기록 없음 {int((~f.delivery_known).sum())}명은 배송 지연 비교에서 제외.", ""]
    dk = f[f.delivery_known]
    base_rep = rate_row("전체", f.repeat)
    md += [f"전체 180일 재구매율: {pct(base_rep['비율'], 2)} ({base_rep['해당']:,}명)", ""]

    # ① 지연 → 부정 리뷰 (리뷰가 있는 고객)
    rv = dk[dk.review_group != "리뷰 없음"].assign(neg=lambda x: x.review_group == "부정")
    neg_rows, neg_t = compare_two(rv, "late", "neg", ["지연", "정시"], False)
    md += ["## ① 배송 지연 → 부정 리뷰 (주 비교)", "", md_rates(neg_rows), "",
           f"차이 {neg_t['diff'] * 100:+.1f}%p, {neg_t['ratio']:.1f}배 · {neg_t['method']} p={neg_t['p']:.2g} → **{neg_t['verdict']}**", ""]

    # ② 지연 → 재구매
    rep_rows, rep_t = compare_two(dk, "late", "repeat", ["지연", "정시"], True)
    md += ["## ② 배송 지연 → 재구매 (주 비교)", "", md_rates(rep_rows), "",
           f"차이 {rep_t['diff'] * 100:+.2f}%p, {rep_t['ratio']:.2f}배 · {rep_t['method']} p={rep_t['p']:.2g} → **{rep_t['verdict']}**", ""]
    chart_flow(neg_rows, rep_rows, os.path.join(OUT, "1_flow_late.png"))

    # ③ 리뷰 → 재구매
    groups = ["부정", "중립", "긍정", "리뷰 없음"]
    rg_rows = [rate_row(g, f.loc[f.review_group == g, "repeat"]) for g in groups]
    main3 = f[f.review_group.isin(["부정", "중립", "긍정"])]
    tab3 = pd.crosstab(main3.review_group, main3.repeat)
    _, p3, _, _ = chi2_contingency(tab3, correction=False)
    np_rows, np_t = compare_two(f[f.review_group.isin(["부정", "긍정"])].assign(isneg=lambda x: x.review_group == "부정"),
                                "isneg", "repeat", ["부정", "긍정"], True)
    md += ["## ③ 리뷰 긍정·부정 → 재구매 (주 비교)", "", md_rates(rg_rows), "",
           f"세 그룹(부정·중립·긍정) 카이제곱 p={p3:.2g}. "
           f"핵심 비교 부정 vs 긍정: 차이 {np_t['diff'] * 100:+.2f}%p, {np_t['ratio']:.2f}배 · p={np_t['p']:.2g} → **{np_t['verdict']}**",
           "", "'리뷰 없음'은 참고로만 보여줍니다.", ""]
    chart_rates(rg_rows, "첫 주문 리뷰 그룹별 180일 재구매율", os.path.join(OUT, "2_repeat_by_review.png"),
                [C_BAD, C_MID, C_OK, C_GRID])

    # ④ 불만 주제
    txt = f[f.text != ""]
    hits = theme_hits(txt.text)
    neg_txt = txt.review_group == "부정"
    th = pd.DataFrame({
        "부정 리뷰 전체": hits[neg_txt].mean(),
        "지연 고객": hits[neg_txt & txt.late].mean(),
        "정시 고객": hits[neg_txt & ~txt.late & txt.delivery_known].mean(),
        "건수(부정)": hits[neg_txt].sum(),
    }).sort_values("부정 리뷰 전체", ascending=False)
    th.to_csv(os.path.join(OUT, "4_themes.csv"), encoding="utf-8-sig")
    md += ["## ④ 부정 리뷰의 불만 주제 (보조, 검정 없음)", "",
           f"글이 있는 부정 리뷰 {int(neg_txt.sum()):,}건 (지연 고객 {int((neg_txt & txt.late).sum()):,}건, "
           f"정시 고객 {int((neg_txt & ~txt.late & txt.delivery_known).sum()):,}건). 한 리뷰가 여러 주제에 걸릴 수 있음.", "",
           "| 주제 | 부정 리뷰 전체 | 지연 고객 | 정시 고객 | 건수 |", "| --- | --- | --- | --- | --- |"]
    for name, r in th.iterrows():
        flag = "" if r["건수(부정)"] >= MIN_THEME else " (30건 미만, 해석 안 함)"
        md.append(f"| {name} | {pct(r['부정 리뷰 전체'])} | {pct(r['지연 고객'])} | {pct(r['정시 고객'])} | "
                  f"{int(r['건수(부정)'])}{flag} |")
    md.append("")
    chart_themes(th[["지연 고객", "정시 고객"]], os.path.join(OUT, "3_complaint_themes.png"))
    samples = []
    for name in THEMES:
        pool = txt[hits[name]].sort_values("review_id")
        k = min(THEME_SAMPLE, len(pool))
        for _, r in pool.sample(k, random_state=SEED).iterrows():
            samples.append({"주제": name, "review_id": r.review_id, "점수": r.score, "리뷰 원문": r.text})
    pd.DataFrame(samples).to_csv(os.path.join(OUT, "4_theme_samples.csv"), index=False, encoding="utf-8-sig")

    # ⑤ 지연 일수 구간
    b_rows = []
    for b in ["정시", "1~3일", "4~7일", "8일 이상"]:
        g = dk[dk.delay_bucket == b]
        nr = rate_row(b, g.loc[g.review_group != "리뷰 없음", "review_group"].eq("부정"))
        rr = rate_row(b, g.repeat)
        b_rows.append({"구간": b, "고객": len(g), "부정 리뷰 비율": nr["비율"], "재구매율": rr["비율"], "재구매 고객": rr["해당"]})
    bt = pd.DataFrame(b_rows)
    bt.to_csv(os.path.join(OUT, "5_delay_buckets.csv"), index=False, encoding="utf-8-sig")
    md += ["## ⑤ 지연 일수 구간별 (보조)", "", "| 구간 | 고객 | 부정 리뷰 비율 | 재구매율 | 재구매 고객 |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {r['구간']} | {r['고객']:,} | {pct(r['부정 리뷰 비율'])} | {pct(r['재구매율'], 2)} | {r['재구매 고객']} |"
           for _, r in bt.iterrows()] + [""]

    # ⑥ 로지스틱 회귀
    reg = dk.assign(y=dk.repeat.astype(int), late_i=dk.late.astype(int), lpay=np.log1p(dk.payment.fillna(0)))
    model = smf.logit("y ~ late_i + C(review_group, Treatment('긍정')) + C(cat, Treatment('기타')) + C(month) + lpay",
                      data=reg).fit(disp=0)
    ci = model.conf_int()
    keep = [k for k in model.params.index if k.startswith(("late_i", "C(review_group", "lpay"))]
    md += ["## ⑥ 함께 보기: 로지스틱 회귀 (보조)", "",
           "재구매 ~ 배송 지연 + 리뷰 그룹(기준: 긍정) + 첫 구매 카테고리(기준: 기타) + 첫 구매 월 + 결제 금액(로그)", "",
           "| 변수 | 오즈비 | 95% 신뢰구간 | p |", "| --- | --- | --- | --- |"]
    for k in keep:
        name = k.replace("C(review_group, Treatment('긍정'))[T.", "리뷰 ").replace("]", "").replace("late_i", "배송 지연").replace("lpay", "결제 금액(로그)")
        md.append(f"| {name} | {np.exp(model.params[k]):.2f} | {np.exp(ci.loc[k, 0]):.2f} ~ {np.exp(ci.loc[k, 1]):.2f} | {model.pvalues[k]:.2g} |")
    md += ["", f"관측 {int(model.nobs):,}명. 카테고리·월 계수는 results/6_logit_full.txt 참고.", ""]
    with open(os.path.join(OUT, "6_logit_full.txt"), "w", encoding="utf-8") as fh:
        fh.write(model.summary().as_text())

    # ⑦ 재구매까지 걸린 일수
    dtr = f.days_to_repeat.dropna()
    md += ["## ⑦ 재구매까지 걸린 일수 (보조)", "",
           f"재구매 고객 {len(dtr):,}명: 25% 지점 {dtr.quantile(.25):.0f}일, 중앙값 {dtr.median():.0f}일, 75% 지점 {dtr.quantile(.75):.0f}일", ""]

    # ⑧ 카테고리별
    cat_rows = []
    for cname in list(CAT_KO.values()) + ["기타"]:
        g = f[(f.cat == cname)]
        gd = g[g.delivery_known]
        late_r = rate_row("지연", gd.loc[gd.late, "repeat"])
        on_r = rate_row("정시", gd.loc[~gd.late, "repeat"])
        ok = late_r["n"] >= MIN_GROUP and late_r["해당"] >= MIN_REPEATERS and on_r["해당"] >= MIN_REPEATERS
        p = two_group_test(gd.loc[gd.late, "repeat"], gd.loc[~gd.late, "repeat"])[0] if ok else np.nan
        negr = g[g.review_group != "리뷰 없음"]
        cat_rows.append({"카테고리": cname, "고객": len(g), "지연 비율": gd.late.mean(),
                         "부정 리뷰 비율": negr.review_group.eq("부정").mean(),
                         "재구매율": g.repeat.mean(), "재구매율_지연": late_r["비율"], "재구매율_정시": on_r["비율"],
                         "지연 고객": late_r["n"], "지연 재구매": late_r["해당"], "p": p,
                         "판단": ("표본 부족" if not ok else ("의미 있는 차이" if p < ALPHA else "차이 없음"))})
    ct = pd.DataFrame(cat_rows).sort_values("고객", ascending=False)
    ct.to_csv(os.path.join(OUT, "8_by_category.csv"), index=False, encoding="utf-8-sig")
    md += ["## ⑧ 카테고리별 (보조)", "",
           "| 카테고리 | 고객 | 지연 비율 | 부정 리뷰 비율 | 재구매율 | 재구매율 지연/정시 | 지연 고객(재구매) | 판단 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r['카테고리']} | {r['고객']:,} | {pct(r['지연 비율'])} | {pct(r['부정 리뷰 비율'])} | {pct(r['재구매율'], 2)} | "
           f"{pct(r['재구매율_지연'], 2)} / {pct(r['재구매율_정시'], 2)} | {r['지연 고객']} ({r['지연 재구매']}) | {r['판단']} |"
           for _, r in ct.iterrows()] + [""]

    md += ["## 읽을 때 주의", "",
           "- 관계(패턴)일 뿐 원인은 아닙니다. 원인은 A/B 테스트로 확인해야 합니다.",
           "- 불만 주제는 미리 정한 단어 사전 기준이며, 주제별 표본 리뷰는 results/4_theme_samples.csv 에 있습니다."]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    f.drop(columns=["text"]).to_csv(os.path.join(OUT, "first_purchase_customers.csv"), index=False, encoding="utf-8-sig")
    print("\n".join(md))


if __name__ == "__main__":
    main()
