#!/usr/bin/env python3
"""부정 리뷰 원문(포르투갈어)의 한국어 번역을 results/review_ko.json 에 모은다.

build_explorer.py 가 이 파일을 읽어 사이트에 번역문을 넣는다.

실제 번역은 Claude 가 했다 (2026-11-20):
    1) 리뷰를 results/translate_work/in_XX.json 묶음으로 나눔
    2) 묶음별로 번역해 out_XX_pN.json ({번호: 번역}) 으로 저장
    3) .venv/bin/python translate_reviews.py --merge  → review_ko.json
Google 번역 무료 주소(main)는 요청 제한(429)으로 쓰지 못했고 참고로만 남겨 둔다.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import pandas as pd

from analyze import RAW

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "results", "review_ko.json")
URL = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=pt&tl=ko&dt=t"
CHUNK = 1500  # 한 번에 보내는 글자 수


def texts_to_translate():
    f = pd.read_csv(os.path.join(BASE, "results", "first_purchase_customers.csv"))
    rv = pd.read_csv(os.path.join(RAW, "olist_order_reviews_dataset.csv"))
    r = rv.sort_values("review_answer_timestamp").groupby("order_id").last()
    t = f.order_id.map(r.review_comment_message).fillna("").astype(str).str.strip()
    t = t[(f.review_group == "부정") & t.ne("")].str.replace(r"\s+", " ", regex=True)
    return sorted(set(t))


def call(text):
    data = urllib.parse.urlencode({"q": text}).encode()
    for wait in (1, 3, 10, 30):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, data=data), timeout=30) as res:
                out = json.load(res)
            return "".join(seg[0] for seg in out[0] if seg and seg[0])
        except Exception as e:  # noqa: BLE001
            print("  재시도:", e)
            time.sleep(wait)
    raise RuntimeError("번역 실패")


def main():
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    todo = [t for t in texts_to_translate() if t not in cache]
    print(f"번역할 리뷰 {len(todo)}건 (저장된 번역 {len(cache)}건)")
    i = 0
    while i < len(todo):
        batch, size = [], 0
        while i < len(todo) and (not batch or size + len(todo[i]) < CHUNK):
            batch.append(todo[i]); size += len(todo[i]) + 1; i += 1
        out = call("\n".join(batch)).split("\n")
        if len(out) != len(batch):  # 줄 수가 어긋나면 한 건씩
            out = [call(t) for t in batch]
        cache.update({t: k.strip() for t, k in zip(batch, out)})
        if i % 200 < len(batch) or i == len(todo):
            print(f"  {i}/{len(todo)}")
            json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
        time.sleep(0.3)
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    print("완료:", CACHE)


def merge():
    """translate_work/out_*.json 을 원문 기준 사전으로 합쳐 review_ko.json 에 저장."""
    import glob
    work = os.path.join(BASE, "results", "translate_work")
    texts = json.load(open(os.path.join(work, "all_texts.json"), encoding="utf-8"))
    out = {}
    for f in sorted(glob.glob(os.path.join(work, "out_*_p*.json"))):
        out.update(json.load(open(f, encoding="utf-8")))
    cache = {texts[int(i)]: k.strip() for i, k in out.items() if k.strip()}
    missing = [i for i in range(len(texts)) if str(i) not in out]
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"번역 {len(cache)}/{len(texts)}건 저장, 빠진 번호 {len(missing)}개 {missing[:10]}")


if __name__ == "__main__":
    import sys
    merge() if "--merge" in sys.argv else main()
