#!/usr/bin/env python3
"""탐색 사이트(docs/)용 데이터 만들기 — Olist 전체 주문·고객 대상 (특정 관점 없이 있는 그대로).

    .venv/bin/python build_explorer.py

출력 (파일을 바로 열어도 동작하도록 JSON 대신 스크립트로 저장):
  docs/data/orders.js   window.OLIST = 주문 99,441건의 조건 값(열 단위 배열)과 이름표
  docs/data/reviews.js  window.OLIST_REVIEWS = 글이 있는 리뷰 [주문 번호, 별점, 원문, 한국어 번역]

고객·주문·판매자의 원래 ID는 내보내지 않고, 0부터 새로 매긴 번호만 씁니다.
"""
import json
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data", "raw")
OUT = os.path.join(BASE, "docs", "data")
os.makedirs(OUT, exist_ok=True)
rd = lambda f, **k: pd.read_csv(os.path.join(RAW, f), **k)  # noqa: E731

CAT_KO = {
    "health_beauty": "뷰티·헬스", "computers_accessories": "컴퓨터 액세서리", "auto": "자동차 용품", "bed_bath_table": "침구·욕실",
    "furniture_decor": "가구·인테리어", "sports_leisure": "스포츠·레저", "perfumery": "향수", "housewares": "생활용품",
    "telephony": "휴대폰", "watches_gifts": "시계·선물", "food_drink": "식음료", "baby": "유아용품", "stationery": "문구",
    "tablets_printing_image": "태블릿·인쇄", "toys": "장난감", "fixed_telephony": "유선 전화기", "garden_tools": "정원용품",
    "fashion_bags_accessories": "가방·패션 잡화", "small_appliances": "소형 가전", "consoles_games": "게임기·게임", "audio": "오디오",
    "fashion_shoes": "신발", "cool_stuff": "아이디어 상품", "luggage_accessories": "여행 가방", "air_conditioning": "냉난방 기기",
    "construction_tools_construction": "건설 공구", "kitchen_dining_laundry_garden_furniture": "주방·세탁·정원 가구",
    "costruction_tools_garden": "정원 공구", "fashion_male_clothing": "남성 의류", "pet_shop": "반려동물 용품",
    "office_furniture": "사무용 가구", "market_place": "마켓플레이스", "electronics": "전자제품", "home_appliances": "가전제품",
    "party_supplies": "파티 용품", "home_confort": "홈 컴포트", "costruction_tools_tools": "공구", "agro_industry_and_commerce": "농업·상업",
    "furniture_mattress_and_upholstery": "매트리스·소파", "books_technical": "전문 서적", "home_construction": "주택 자재",
    "musical_instruments": "악기", "furniture_living_room": "거실 가구", "construction_tools_lights": "조명 공구",
    "industry_commerce_and_business": "산업·비즈니스", "food": "식품", "art": "미술품", "furniture_bedroom": "침실 가구",
    "books_general_interest": "일반 도서", "construction_tools_safety": "안전 공구", "fashion_underwear_beach": "속옷·수영복",
    "fashion_sport": "스포츠 의류", "signaling_and_security": "신호·보안 장비", "computers": "컴퓨터", "christmas_supplies": "크리스마스 용품",
    "fashio_female_clothing": "여성 의류", "home_appliances_2": "가전제품 2", "books_imported": "수입 도서", "drinks": "음료",
    "cine_photo": "영화·사진", "la_cuisine": "주방용품", "music": "음악", "home_comfort_2": "홈 컴포트 2",
    "small_appliances_home_oven_and_coffee": "오븐·커피 가전", "cds_dvds_musicals": "CD·DVD·뮤지컬", "dvds_blu_ray": "DVD·블루레이",
    "flowers": "꽃", "arts_and_craftmanship": "공예", "diapers_and_hygiene": "기저귀·위생용품", "fashion_childrens_clothes": "아동복",
    "security_and_services": "보안·서비스", "pc_gamer": "게이밍 PC", "portateis_cozinha_e_preparadores_de_alimentos": "휴대용 주방 기기",
}
STATE_KO = {
    "SP": "상파울루", "RJ": "리우데자네이루", "MG": "미나스제라이스", "RS": "히우그란지두술", "PR": "파라나",
    "SC": "산타카타리나", "BA": "바이아", "ES": "이스피리투산투", "GO": "고이아스", "DF": "연방구(브라질리아)",
    "PE": "페르남부쿠", "CE": "세아라", "PA": "파라", "MT": "마투그로수", "MA": "마라냥", "MS": "마투그로수두술",
    "PB": "파라이바", "PI": "피아우이", "RN": "히우그란지두노르치", "AL": "알라고아스", "SE": "세르지피",
    "TO": "토칸칭스", "RO": "혼도니아", "AM": "아마조나스", "AC": "아크리", "AP": "아마파", "RR": "호라이마",
}
ST = ["delivered", "shipped", "invoiced", "processing", "approved", "created", "canceled", "unavailable"]
ST_KO = ["배송 완료", "배송 중", "송장 발행", "처리 중", "결제 승인", "주문 생성", "취소", "상품 없음"]
PT = ["credit_card", "boleto", "voucher", "debit_card", "not_defined"]
PT_KO = ["신용카드", "볼레토(현금 결제)", "바우처", "체크카드", "미정", "결제 정보 없음"]
OT_KO = ["상품 1개", "같은 상품 여러 개", "한 판매자 · 다른 상품", "여러 판매자", "상품 정보 없음"]
SZ_KO = ["1~5건", "6~20건", "21~100건", "101~500건", "501건 이상", "판매자 정보 없음"]
DL_KO = ["정시 도착", "1~3일 늦게 도착", "4~7일 늦게 도착", "8일 이상 늦게 도착", "도착 기록 없음"]
# 공개 사이트에 실명이 나가지 않도록 리뷰 속 사람 이름을 가림 (번역 중 발견된 것)
NAMES = ["Lourdes Marli Barichello Conceição", "Diego Machado Lopes", "Cyro Dias Braga", "Suzana Costa", "Hélder", "mateus"]

o = rd("olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
c = rd("olist_customers_dataset.csv")
it = rd("olist_order_items_dataset.csv")
p = rd("olist_products_dataset.csv")
pay = rd("olist_order_payments_dataset.csv")
rv = rd("olist_order_reviews_dataset.csv")
en = dict(zip(*rd("product_category_name_translation.csv").values.T))

o = o.merge(c[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id", how="left")
o = o.sort_values(["order_purchase_timestamp", "order_id"]).reset_index(drop=True)

# 고객: 첫 주문 순서대로 0부터 번호 / 고객별 몇 번째 주문인지
cu = o.customer_unique_id.map({u: i for i, u in enumerate(o.customer_unique_id.drop_duplicates())})
sq = (o.groupby("customer_unique_id").cumcount() > 0).astype(int)   # 0 첫 주문, 1 재주문

# 주문 월
months = pd.period_range(o.order_purchase_timestamp.min().to_period("M"), o.order_purchase_timestamp.max().to_period("M"), freq="M")
m = o.order_purchase_timestamp.dt.to_period("M").map({mm: i for i, mm in enumerate(months)})

# 상태
st = o.order_status.map({k: i for i, k in enumerate(ST)})

# 카테고리·판매자: 주문의 첫 번째 상품 기준
fi = it[it.order_item_id == 1].merge(p[["product_id", "product_category_name"]], on="product_id", how="left").set_index("order_id")
cat_name = o.order_id.map(fi.product_category_name).map(lambda x: x if pd.isna(x) else CAT_KO.get(en.get(x, x), CAT_KO.get(x, x)))
cat_name = np.where(~o.order_id.isin(fi.index), "상품 정보 없음", cat_name.fillna("카테고리 없음"))
CATS = pd.Series(cat_name).value_counts().index.tolist()
cat = pd.Series(cat_name).map({k: i for i, k in enumerate(CATS)})
seller = o.order_id.map(fi.seller_id)
se = seller.map({k: i for i, k in enumerate(seller.value_counts().index)}).fillna(-1).astype(int)
items = o.order_id.map(it.groupby("order_id").size()).fillna(0).astype(int)

# 주문 구성: 한 주문에 담긴 상품 (여러 판매자 > 상품 1개 > 같은 상품 여러 개 > 한 판매자의 다른 상품 여러 개)
ag = it.groupby("order_id").agg(n_items=("order_item_id", "size"), n_products=("product_id", "nunique"), n_sellers=("seller_id", "nunique"))
a_items, a_prod, a_sell = (o.order_id.map(ag[k]).fillna(0).astype(int) for k in ("n_items", "n_products", "n_sellers"))
ot = np.select([a_items == 0, a_sell > 1, a_items == 1, a_prod == 1], [4, 3, 0, 1], 2)
# 판매자 규모: 주문의 첫 상품 판매자가 전체 기간에 받은 주문 수 구간
seller_orders = it.groupby("seller_id").order_id.nunique()
so = seller.map(seller_orders)
sz = np.select([so.isna(), so <= 5, so <= 20, so <= 100, so <= 500], [5, 0, 1, 2, 3], 4)

# 지역
STATES = o.customer_state.value_counts().index.tolist()
s = o.customer_state.map({k: i for i, k in enumerate(STATES)})

# 결제: 합계 금액(헤알, 반올림) / 가장 큰 결제의 수단
pv = o.order_id.map(pay.groupby("order_id").payment_value.sum()).fillna(0).round().astype(int)
main = pay.sort_values("payment_value").groupby("order_id").last().payment_type
pt = o.order_id.map(main).map({k: i for i, k in enumerate(PT)}).fillna(len(PT)).astype(int)

# 배송: 예정일 대비 도착 (날짜 기준) / 주문부터 도착까지 일수
dlv, est = o.order_delivered_customer_date.dt.normalize(), o.order_estimated_delivery_date.dt.normalize()
late = (dlv - est).dt.days
dl = np.select([dlv.isna(), late <= 0, late <= 3, late <= 7], [4, 0, 1, 2], 3)
dd = (o.order_delivered_customer_date - o.order_purchase_timestamp).dt.days.fillna(-1).astype(int)

# 리뷰: 주문마다 답변 시각이 가장 늦은 리뷰 하나
r = rv.sort_values("review_answer_timestamp").groupby("order_id").last()
rs = o.order_id.map(r.review_score).fillna(0).astype(int)   # 0 리뷰 없음
txt = o.order_id.map(r.review_comment_message).fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

KO = os.path.join(BASE, "results", "review_ko.json")
ko = json.load(open(KO, encoding="utf-8")) if os.path.exists(KO) else {}
name_re = re.compile(r"(?<![A-Za-zÀ-ÿ])(" + "|".join(map(re.escape, NAMES)) + r")(?![A-Za-zÀ-ÿ])", re.IGNORECASE)
mask = lambda x: name_re.sub("○○○", x)  # noqa: E731
has = txt.ne("")
reviews = [[int(i), int(rs[i]), mask(txt[i]), mask(ko.get(txt[i], ""))] for i in np.flatnonzero(has.values)]

cols = {"cu": cu, "m": m, "st": st, "c": cat, "s": s, "se": se, "n": items, "pt": pt, "pv": pv, "dl": dl, "dd": dd, "rs": rs, "sq": sq, "ot": ot, "sz": sz}
for k, v in cols.items():
    assert not pd.isna(pd.Series(v)).any(), k
data = {
    "dims": {
        "m": [str(x) for x in months], "st": ST_KO, "c": CATS, "s": STATES, "sKo": [STATE_KO.get(x, x) for x in STATES],
        "pt": PT_KO, "dl": DL_KO, "ot": OT_KO, "sz": SZ_KO,
    },
    "cols": {k: [int(x) for x in v] for k, v in cols.items()},
}
with open(os.path.join(OUT, "orders.js"), "w", encoding="utf-8") as fp:
    fp.write("window.OLIST=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
with open(os.path.join(OUT, "reviews.js"), "w", encoding="utf-8") as fp:
    fp.write("window.OLIST_REVIEWS=" + json.dumps(reviews, ensure_ascii=False, separators=(",", ":")) + ";\n")

# 확인용 요약
print(f"주문 {len(o):,}건 · 고객 {cu.nunique():,}명 · 판매자 {(se >= 0).groupby(se).size().shape[0] - (1 if (se < 0).any() else 0):,}명")
print(f"기간 {months[0]} ~ {months[-1]} ({len(months)}개월) · 카테고리 {len(CATS)}개 · 결제 합계 R$ {pv.sum():,}")
print(f"재주문 {sq.sum():,}건 · 리뷰 있는 주문 {(rs > 0).sum():,} · 글 있는 리뷰 {len(reviews):,} (번역 {sum(1 for x in reviews if x[3]):,})")
for f in ("orders.js", "reviews.js"):
    print(f, f"{os.path.getsize(os.path.join(OUT, f)) / 1024:,.0f} KB")
