#!/usr/bin/env python3
"""번역이 없는 리뷰를 Azure 번역기(Translator, 무료 F0 등급)로 번역한다. 보여 주기용이며 분석에는 쓰지 않는다.

- 대상: docs/data/reviews.js 에서 한국어 번역(Claude)이 없는 리뷰의 원문. 이름 가림(○○○)이 적용된 글을 보낸다.
- 결과: results/review_ko_machine.json  {원문: 한국어}  — build_explorer.py 가 읽어 '기계 번역'으로 표시한다.
- 알려진 오류(검사에서 발견)는 사전 규칙(fix)으로 고친다. 규칙은 results/review_translation_rules.md 참고.
- 키는 .env 에서만 읽고 화면에 출력하지 않는다:
    AZURE_TRANSLATOR_KEY=...
    AZURE_TRANSLATOR_REGION=koreacentral

    .venv/bin/python translate_azure.py --self-test      # 키 없이 사전 규칙만 확인
    .venv/bin/python translate_azure.py --test           # 3건만 시험 번역
    .venv/bin/python translate_azure.py                  # 전부 (중간에 멈춰도 다시 실행하면 이어서)
    .venv/bin/python translate_azure.py --audit 100      # 검사용 무작위 100건 → results/machine_translation_audit.md
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "results", "review_ko_machine.json")
URL = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&from=pt&to=ko"
MAX_ITEMS, MAX_CHARS = 100, 9000      # 요청 한 번에 보내는 양 (Azure 한도 1,000개 · 5만 자보다 작게)
FREE_MONTHLY = 2_000_000              # F0 등급: 월 200만 자


# ---------- 사전 규칙 (검사에서 발견된 반복 오류) ----------
def fix(src, ko):
    s = src.lower()
    if re.search(r"\bprazo", s):                       # prazo = 배송 예정일 (마감일·기한으로 번역되는 오류)
        ko = re.sub(r"마감\s*일|마감\s*기한|마감\s*시간|기한", "예정일", ko)
    if re.search(r"nota fiscal|\bnf-?e?\b", s):         # nota fiscal = 세금계산서 (청구서로 번역되는 오류)
        ko = ko.replace("청구서", "세금계산서")
    for en, pat in (("Lannister", r"라니스터|라니스타"), ("Stark", r"스타크"), ("Targaryen", r"타르가르옌|타르가리엔|타르가르|타르가리안"),
                    ("Baratheon", r"바라테온|바라시온|바라티온")):   # 마켓플레이스 이름은 번역하지 않음
        if en.lower() in s:
            ko = re.sub(rf"(?:{pat})(?:\s*(?:가문|가|집안|하우스))?", en, ko)
    return ko


def self_test():
    cases = [("Chegou antes do prazo", "마감일 전에 도착했어요", "예정일 전에 도착했어요"),
             ("Não veio a nota fiscal", "청구서가 도착하지 않았습니다", "세금계산서가 도착하지 않았습니다"),
             ("Comprei na lannister", "라니스터 가문에서 샀어요", "Lannister에서 샀어요"),
             ("Produto bom", "좋은 기한 제품", "좋은 기한 제품")]      # 키워드 없으면 건드리지 않음
    ok = True
    for src, bad, want in cases:
        got = fix(src, bad)
        print(("OK  " if got == want else "FAIL"), src, "→", got)
        ok &= got == want
    sys.exit(0 if ok else 1)


# ---------- 입출력 ----------
def load_todo():
    s = open(os.path.join(BASE, "docs", "data", "reviews.js"), encoding="utf-8").read()
    rows = json.loads(s[s.index("["):s.rindex("]") + 1])   # [주문 번호, 별점, 원문(가림), 한국어]
    texts = sorted({r[2] for r in rows if r[2].strip() and not r[3].strip() and re.search(r"[A-Za-zÀ-ÿ]", r[2])})
    return texts


def load_done():
    return json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}


def save(done):
    tmp = OUT + ".tmp"
    json.dump(done, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    os.replace(tmp, OUT)


def env():
    vals = {}
    with open(os.path.join(BASE, ".env"), encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                vals[k.strip()] = v.strip().strip('"').strip("'")
    if not vals.get("AZURE_TRANSLATOR_KEY"):
        sys.exit(".env 에 AZURE_TRANSLATOR_KEY 가 비어 있습니다.")
    return vals["AZURE_TRANSLATOR_KEY"], vals.get("AZURE_TRANSLATOR_REGION", "koreacentral")


def call(texts, key, region):
    body = json.dumps([{"Text": t} for t in texts]).encode()
    req = urllib.request.Request(URL, data=body, headers={"Ocp-Apim-Subscription-Key": key, "Ocp-Apim-Subscription-Region": region,
                                                           "Content-Type": "application/json"})
    for wait in (5, 15, 60, 120, 300):
        try:
            with urllib.request.urlopen(req, timeout=60) as res:
                return [r["translations"][0]["text"] for r in json.load(res)]
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:200]
            if e.code in (401, 403):
                sys.exit(f"키 또는 지역이 맞지 않습니다 ({e.code}): {msg}")
            if e.code == 429 or e.code >= 500:
                w = int(e.headers.get("Retry-After") or wait)
                print(f"  잠시 대기 {w}초 ({e.code})", flush=True); time.sleep(w); continue
            sys.exit(f"요청 실패 ({e.code}): {msg}")
        except urllib.error.URLError as e:
            print("  연결 재시도:", e.reason, flush=True); time.sleep(wait)
    sys.exit("계속 실패해서 멈춥니다. 다시 실행하면 이어서 합니다.")


def run(test=False):
    todo_all = load_todo(); done = load_done()
    todo = [t for t in todo_all if t not in done]
    chars = sum(len(t) for t in todo)
    print(f"번역 대상 {len(todo_all):,}개 · 남은 {len(todo):,}개 · {chars:,}자 (무료 한도 {FREE_MONTHLY:,}자)", flush=True)
    if chars > FREE_MONTHLY * 0.95:
        sys.exit("남은 글자 수가 무료 한도에 가깝습니다. 나눠서 실행하세요.")
    key, region = env()
    if test:
        for t, k in zip(todo[:3], call(todo[:3], key, region)):
            print(f"{t}\n  → {fix(t, k)}")
        return
    pos, n = 0, 0
    while pos < len(todo):
        batch, size = [], 0
        while pos < len(todo) and len(batch) < MAX_ITEMS and (not batch or size + len(todo[pos]) <= MAX_CHARS):
            batch.append(todo[pos]); size += len(todo[pos]); pos += 1
        for t, k in zip(batch, call(batch, key, region)):
            k = fix(t, k).strip()
            if k and k != t:
                done[t] = k
        n += 1
        if n % 10 == 0 or pos == len(todo):
            save(done); print(f"  {pos:,}/{len(todo):,}", flush=True)
    save(done)
    print(f"완료: {len(done):,}개 저장 → {os.path.relpath(OUT, BASE)}")


def audit(n):
    done = load_done()
    if not done:
        sys.exit("검사할 번역이 아직 없습니다.")
    keys = sorted(done)
    pick = random.Random(7).sample(keys, min(n, len(keys)))
    lines = [f"# 기계 번역 검사 표본 {len(pick)}건 (무작위, seed 7)", "", "원문과 번역을 대조해 뜻이 틀린 것을 표시한다. 기준은 `review_translation_rules.md`.", ""]
    for i, t in enumerate(pick, 1):
        lines += [f"{i}. {t}", f"   → {done[t]}", ""]
    open(os.path.join(BASE, "results", "machine_translation_audit.md"), "w", encoding="utf-8").write("\n".join(lines))
    print(f"검사 표본 {len(pick)}건 → results/machine_translation_audit.md")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--self-test" in a:
        self_test()
    elif "--audit" in a:
        audit(int(a[a.index("--audit") + 1]) if len(a) > a.index("--audit") + 1 else 100)
    else:
        run(test="--test" in a)
