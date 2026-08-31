#!/usr/bin/env python3
"""T1 파이프라인 내 실검증 — 모의가 아니라 진짜 크롭으로. (RTX 6000 · thor7 / H100 · thor4)

    MODEL=Qwen/Qwen3.5-9B THOR_ROOT=data/thor7_t7view A3_PREFIX=/tmp/t7_a_ \\
      QC_PREFIX=/tmp/t7_q_ OUT_JSONL=t1_scores_t7.jsonl \\
      python scripts/exp_t1_verify_pipeline.py

각 타입단일 이동 타겟: 후보(점수 q0.80+) 를 최신부터 최대 MAXWALK 장 걸으며
프레임 크롭을 2AFC 로짓으로 채점해 전부 기록한다. 문턱 판정·정지 규칙은
로컬 스윕에서 재현한다(§89 — 문턱은 도메인·모델별로 밀린다).
크롭 상자는 프레임의 1/3 (make_sim_pairs CROP=SZ/3 과 동일 기하 — 768 캘리브레이션
AUC 0.944 가 이 조건의 측정치다). 프레임 해상도는 이미지에서 읽는다(384 고정 아님).

⚠️ 모의와 다른 점: 실제 오검출은 같은 혼동물의 반복이라 FA 가 상관될 수 있다.
이 실행이 그 위험을 판정한다. 대조 라벨은 그 패치에서 두 번째로 강한 타겟 타입.
"""
import glob, json, os
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B")
ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.environ.get("A3_PREFIX", "/tmp/a3_")
QCP = os.environ.get("QC_PREFIX", "/tmp/qc_")
OUTJ = os.environ.get("OUT_JSONL", "t1_verified.jsonl")
MAXWALK = int(os.environ.get("MAXWALK", "20"))
FLOOR = float(os.environ.get("FLOOR", "0.80"))   # 후보 점수 하한 분위 — rtx7_walksim 으로 고를 것
pr = AutoProcessor.from_pretrained(MODEL)
md = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map="auto").eval()
tok = pr.tokenizer
IDA = tok.encode("A", add_special_tokens=False)[0]
IDB = tok.encode("B", add_special_tokens=False)[0]
IDC = tok.encode("C", add_special_tokens=False)[0]
def words(t): return "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
print("모델 %s" % MODEL, flush=True)


def _logits(img, q):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    try:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = pr.apply_chat_template(msgs, add_generation_prompt=True)
    inp = pr(images=[img], text=text, return_tensors="pt").to(md.device)
    with torch.no_grad():
        return md(**inp).logits[0, -1]

def sab(img, a, b):
    lg = _logits(img, "Which object is in this image: (A) %s or (B) %s? Answer only A or B." % (a, b))
    return float(lg[IDA] - lg[IDB])

def sac(img, a, b):
    # "둘 다 아님" 선택지 — 2AFC 는 혼동물(정체가 B 도 아닌 것)을 못 거른다 (§FA 상관)
    lg = _logits(img, "Which is in this image: (A) %s, (B) %s, or (C) neither? Answer only A, B, or C." % (a, b))
    return float(lg[IDA] - max(float(lg[IDB]), float(lg[IDC])))


from collections import Counter
out = open(OUTJ, "w")
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    BXa = za["bx"] if "bx" in za.files else None
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json"))
    lv = {int(os.path.basename(p)[:-4]): p
          for p in glob.glob(os.path.join(hd, "live", "*.jpg"))}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    moves = {m["oid"]: m for m in g["moves"]}
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        if oid not in moves: continue                    # T1 후보(이동)만 — 비용 절약
        ti = vocab.index(v0["type"])
        TS = QS[:, j] + STx[:, j]
        th80 = np.quantile(TS, FLOOR)
        cands = sorted(np.where(TS >= th80)[0], key=lambda i: -ts[i])[:MAXWALK]
        ver = []; walked = 0
        for i in cands:
            t = int(ts[i])
            if t not in lv: continue
            walked += 1
            im = Image.open(lv[t]).convert("RGB")
            W, H = im.size
            if BXa is not None:
                # 박스 크롭 — 진짜 수용 19% 의 크롭 빗나감 몫 회수 (§112 레버 b).
                # bx = (cx,cy,w,h) 패딩 정방 정규화, 1.3배 여유, 최소 96px
                bcx, bcy, bw, bh = [float(x) * max(W, H) for x in BXa[i, ti]]
                h2 = max(48, int(max(bw, bh) * 0.65))
                cx, cy = bcx, bcy
            else:
                cx = (P[i, ti] % pw + .5) / pw * W
                cy = (P[i, ti] // pw + .5) / ph * H
                h2 = max(64, W // 6)     # 상자 = W/3, make_sim_pairs 와 동일 기하
            c = im.crop((max(0, int(cx)-h2), max(0, int(cy)-h2),
                         min(W, int(cx)+h2), min(H, int(cy)+h2))).resize((336, 336))
            alt_c = int(np.argsort(-S[i, :nT])[1] if np.argsort(-S[i, :nT])[0] == ti
                        else np.argsort(-S[i, :nT])[0])
            _a, _b = words(v0["type"]), words(vocab[alt_c])
            s = sab(c, _a, _b); s2 = sac(c, _a, _b)
            # ⚠️ 조기중단·문턱판정 제거 — **점수만 기록**하고 문턱은 로컬에서 스윕한다.
            # (1차 실행에서 실사 문턱 0.875 가 시뮬 로짓 분포에 안 맞아 전부 통과 —
            # 검증 프레임 진짜 비율 0.32, T1 0.365 로 통계 이하. 도메인별 캘리브레이션 필수)
            ver.append([int(i), round(s, 3), round(s2, 3)])
        out.write(json.dumps(dict(house=hn, oid=oid, scored=ver, walked=walked)) + "\n")
        out.flush()
    print(hn, "완료", flush=True)
out.close()
print("→", OUTJ)
