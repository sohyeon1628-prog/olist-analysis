#!/usr/bin/env python3
"""BERTimbau 감성 모델: B2W 리뷰로 학습 → 정답 세트로 평가 → Olist 리뷰에 적용.

설정은 results/sentiment/model_setup.md 에 미리 정해 두었다.
    .venv-ml/bin/python sentiment_model.py train    # 학습 + B2W 검증
    .venv-ml/bin/python sentiment_model.py gold     # 정답 세트 400건 평가
    .venv-ml/bin/python sentiment_model.py olist    # Olist 글 리뷰 전체 판정
"""
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE = os.path.dirname(os.path.abspath(__file__))
SD = os.path.join(BASE, "results", "sentiment")
MODEL_DIR = os.path.join(SD, "model")
NAME = "neuralmind/bert-base-portuguese-cased"
SEED, N_TRAIN, N_VAL, MAX_LEN, BATCH, LR = 42, 4000, 500, 128, 16, 2e-5
torch.set_num_threads(4)
norm = lambda s: re.sub(r"\s+", " ", str(s)).strip()


def length_batches(texts, tok, batch, rng):
    """길이가 비슷한 글끼리 배치를 묶는다 (패딩 줄이기). 배치 순서는 섞는다."""
    lens = [len(tok.tokenize(t)[:MAX_LEN]) for t in texts]
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    out = []
    for c in range(0, len(idx), batch * 50):
        chunk = sorted(idx[c:c + batch * 50], key=lambda i: lens[i])
        out += [chunk[k:k + batch] for k in range(0, len(chunk), batch)]
    rng.shuffle(out)
    return out


def predict(texts, tok, model):
    """긍정 확률 목록. 길이순으로 묶어 빠르게 계산하고 원래 순서로 돌려준다."""
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    p = np.zeros(len(texts))
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(order), 64):
            ids = order[k:k + 64]
            x = tok([texts[i] for i in ids], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            p[ids] = torch.softmax(model(**x).logits, -1)[:, 1].numpy()
            if (k // 64) % 50 == 0:
                print(f"  판정 {k + len(ids):,}/{len(texts):,} · {time.time() - t0:.0f}초", flush=True)
    return p


def train():
    d = pd.read_csv(os.path.join(BASE, "data", "b2w", "B2W-Reviews01.csv"), usecols=["overall_rating", "review_text"], low_memory=False)
    d["text"] = d.review_text.map(norm)
    d = d[d.text.ne("") & d.overall_rating.ne(3)].drop_duplicates("text")
    d["y"] = (d.overall_rating >= 4).astype(int)
    neg = d[d.y == 0].sample(N_TRAIN + N_VAL, random_state=SEED)
    pos = d[d.y == 1].sample(N_TRAIN + N_VAL, random_state=SEED)
    tr = pd.concat([neg[:N_TRAIN], pos[:N_TRAIN]])
    va = pd.concat([neg[N_TRAIN:], pos[N_TRAIN:]])
    print(f"학습 {len(tr):,}건 · 검증 {len(va):,}건", flush=True)

    random.seed(SEED); torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2)
    texts, ys = tr.text.tolist(), tr.y.tolist()
    batches = length_batches(texts, tok, BATCH, random.Random(SEED))
    no_decay = ["bias", "LayerNorm.weight"]
    params = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": 0.01},
              {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
    opt = torch.optim.AdamW(params, lr=LR)
    sch = get_linear_schedule_with_warmup(opt, int(len(batches) * .1), len(batches))
    model.train(); t0 = time.time(); run = 0.0
    for s, b in enumerate(batches, 1):
        x = tok([texts[i] for i in b], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        loss = model(**x, labels=torch.tensor([ys[i] for i in b])).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad()
        run = .98 * run + .02 * loss.item() if s > 1 else loss.item()
        if s % 25 == 0 or s == len(batches):
            el = time.time() - t0
            print(f"  학습 {s}/{len(batches)} · 손실 {run:.3f} · {el / 60:.0f}분 경과, 약 {el / s * (len(batches) - s) / 60:.0f}분 남음", flush=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR); tok.save_pretrained(MODEL_DIR)

    p = predict(va.text.tolist(), tok, model)
    acc = float(((p >= .5).astype(int) == va.y.values).mean())
    json.dump({"b2w_val_accuracy": acc, "n_train": len(tr), "n_val": len(va)}, open(os.path.join(SD, "b2w_val.json"), "w"), indent=1)
    print(f"B2W 검증 정확도 (별점 기준, 참고용): {acc:.3f}", flush=True)


def load():
    return AutoTokenizer.from_pretrained(MODEL_DIR), AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)


def gold():
    tok, model = load()
    t = json.load(open(os.path.join(SD, "gold_texts.json"), encoding="utf-8"))
    lab = {x["id"]: x["label"] for x in json.load(open(os.path.join(SD, "gold_labels_claude.json"), encoding="utf-8"))}
    star = {x["id"]: x["star"] for x in json.load(open(os.path.join(SD, "gold_key.json"), encoding="utf-8"))}
    p = predict([x["text"] for x in t], tok, model)
    df = pd.DataFrame({"id": [x["id"] for x in t], "p_pos": p})
    df["model"] = np.where(df.p_pos >= .5, "긍정", "부정")
    df["gold"] = df.id.map(lab); df["star"] = df.id.map(star)
    df.to_csv(os.path.join(SD, "gold_model.csv"), index=False, encoding="utf-8-sig")
    e = df[df.gold.isin(["부정", "긍정"])]
    print(f"정답 세트 정확도 (부정·긍정 {len(e)}건): {(e.model == e.gold).mean():.3f}")
    for g in ("부정", "긍정"):
        s = e[e.gold == g]; print(f"  {g} 재현율: {(s.model == g).mean():.3f} ({len(s)}건)")
    for g in ("혼합", "중립"):
        s = df[df.gold == g]; print(f"  {g} {len(s)}건 → 모델 부정 {(s.model == '부정').mean():.0%}")
    b = df[df.gold.isin(["부정", "긍정"])]
    print(f"  (비교) 별점 1~2=부정·4~5=긍정으로 맞힌 비율: {((b.star >= 4) == (b.gold == '긍정'))[b.star != 3].mean():.3f}")


def olist():
    tok, model = load()
    s = open(os.path.join(BASE, "docs", "data", "reviews.js"), encoding="utf-8").read()
    rows = json.loads(s[s.index("["):s.rindex("]") + 1])
    texts = sorted({norm(r[2]) for r in rows if norm(r[2])})
    print(f"Olist 고유 글 {len(texts):,}개", flush=True)
    p = predict(texts, tok, model)
    pd.DataFrame({"text": texts, "p_pos": p}).to_csv(os.path.join(SD, "olist_text_sentiment.csv"), index=False, encoding="utf-8-sig")
    print("저장: results/sentiment/olist_text_sentiment.csv")


if __name__ == "__main__":
    {"train": train, "gold": gold, "olist": olist}[sys.argv[1]]()
