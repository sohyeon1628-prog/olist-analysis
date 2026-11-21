#!/usr/bin/env python3
"""평가용 정답 세트 표본 추출 (results/sentiment/guideline.md 참고).

별점마다 80건, seed 42. 판정용 파일에는 원문만, 별점은 따로 저장한다.
"""
import json, os, random, re

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results", "sentiment")
s = open(os.path.join(BASE, "docs", "data", "reviews.js"), encoding="utf-8").read()
rows = json.loads(s[s.index("["):s.rindex("]") + 1])   # [주문 번호, 별점, 원문, 번역]

rng = random.Random(42)
pick = []
for star in range(1, 6):
    pool = [r for r in rows if r[1] == star and r[2].strip()]
    pick += rng.sample(pool, 80)
rng.shuffle(pick)

texts = [{"id": i + 1, "text": re.sub(r"\s+", " ", r[2]).strip()} for i, r in enumerate(pick)]
key = [{"id": i + 1, "order": r[0], "star": r[1]} for i, r in enumerate(pick)]
json.dump(texts, open(os.path.join(OUT, "gold_texts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
json.dump(key, open(os.path.join(OUT, "gold_key.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("저장:", len(texts), "건")
