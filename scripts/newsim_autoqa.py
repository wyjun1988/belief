#!/usr/bin/env python3
"""새 시뮬 에피소드 GT → 자동 QA 생성 + 우리 스택 채점. (M1 Max)

    python scripts/newsim_autoqa.py --ep data/newsim/ep1

문항 생성 (GT 스트림에서):
  now    "X 지금 어디?"  — 질의 시각 t 의 GT 방 (10초 간격 × 타입단일 물체)
  before "X 전에 어디?"  — 이동 물체에 한해, 이동 이전의 GT 방
4지선다 = 정답 + 무작위 3방. 채점 = 우리 결합 국소화 + 기록 규칙 (ep1demo 레시피).

⚠️ EgoLifeQA 교훈(§95): 벤치는 과제 정합이 먼저 — 이건 우리 과제 정의 그대로다.
"""
import argparse, glob, json, os
import numpy as np
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--ep", default="data/newsim/ep1")
a = ap.parse_args()
g = json.load(open(os.path.join(a.ep, "gt.json")))
sm = g["scene_meta"]; rt = g["room_types"]; rids = sorted(rt)
live = {m["t"]: m for m in g["live"]}
cnt = Counter(v["type"] for v in g["gt0"].values())
moves = {m["oid"]: m for m in g["moves"]}
rng = np.random.default_rng(0)

# GT 방 이력
def room_at(oid, t):
    m = moves.get(oid)
    if m and t >= m["t"]: return m["to"]
    return g["gt0"][oid]["room"]

qa = []
T = max(live)
for oid, v in g["gt0"].items():
    if not v["room"] or cnt[v["type"]] > 1: continue
    for tq in range(20, T, 10):
        gtr = room_at(oid, tq)
        qa.append(dict(oid=oid, typ=v["type"], tq=tq, kind="now", gt=gtr))
    m = moves.get(oid)
    if m and m["t"] + 10 < T:
        qa.append(dict(oid=oid, typ=v["type"], tq=T, kind="before", gt=m["frm"]))
print("자동 QA %d문항 (now %d · before %d) · 물체 %d종"
      % (len(qa), sum(q["kind"] == "now" for q in qa),
         sum(q["kind"] == "before" for q in qa),
         len({q["oid"] for q in qa})), flush=True)

# ── 우리 스택: OWL 점수 (ep1demo 와 동일 인라인 캐시) ──
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tgt_types = sorted({q["typ"] for q in qa})
st_types = sorted({v["type"] for v in sm["static"].values()})
vocab = tgt_types + [t for t in st_types if t not in tgt_types]
nT = len(tgt_types)
pr = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
md = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
def w2(t): return "a photo of a " + t.replace("_", " ")
ti = pr(text=[[w2(v) for v in vocab]], images=[Image.new("RGB", (256, 256), (128,)*3)],
        return_tensors="pt").to(DEV)
with torch.no_grad():
    o = md.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                 pixel_values=ti["pixel_values"], return_dict=True)
TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)
CJ = os.path.join(a.ep, "autoqa_owl.npz")
fr = sorted(glob.glob(os.path.join(a.ep, "live", "*.jpg")))
if os.path.exists(CJ):
    z = np.load(CJ, allow_pickle=True); S, P, ph, pw = z["s"], z["p"], int(z["ph"]), int(z["pw"])
else:
    S_, P_ = [], []
    for i in range(0, len(fr), 4):
        ims = [Image.open(p).convert("RGB") for p in fr[i:i+4]]
        pv = pr(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
        with torch.no_grad():
            fm = md.image_embedder(pixel_values=pv)[0]
            b, ph, pw, hd = fm.shape
            lg, _ = md.class_predictor(fm.reshape(b, ph*pw, hd),
                                       TX.unsqueeze(0).expand(b, -1, -1),
                                       MK.unsqueeze(0).expand(b, -1))
        p_ = torch.sigmoid(lg)
        S_.append(p_.amax(1).float().cpu().numpy()); P_.append(p_.argmax(1).int().cpu().numpy())
    S, P = np.concatenate(S_), np.concatenate(P_)
    np.savez_compressed(CJ, s=S, p=P, ph=ph, pw=pw)
ts = np.array([int(os.path.basename(p)[:-4]) - 1 for p in fr])
arm = np.array([live.get(t, {}).get("room", "") for t in ts], object)
py, px = P // pw, P % pw
rtypes = {}
for v in sm["static"].values():
    rtypes.setdefault(v["room"], {}).setdefault(v["type"], 0)
idf = {t: 1.0/max(sum(t in rtypes.get(r, {}) for r in rids), 1) for t in st_types}

def loc(idx, ti_):
    acc = {r: 0.0 for r in rids}
    base = 0.0
    for i in idx:
        cy, cx = py[i, ti_], px[i, ti_]
        sc = {r: 0.0 for r in rids}
        for c in range(nT, len(vocab)):
            t = vocab[c]
            if S[i, c] < .05: continue
            d = np.hypot(py[i, c]-cy, px[i, c]-cx)
            w = float(S[i, c]) / (1 + d/6) * idf.get(t, .2)
            for r in rids:
                if t in rtypes.get(r, {}): sc[r] += w
        # §92 융합 가중: 같은방 1 · 이웃 0.2 · 기타 0.05 (문 정보 없으니 같은방/기타)
        for r in rids:
            sc[r] *= 1.0 if r == arm[i] else 0.1
        t2 = sum(sc.values()) + 1e-9
        for r in rids: acc[r] += sc[r]/t2
    return max(acc, key=acc.get) if sum(acc.values()) > 0 else None

ok = Counter(); n = Counter()
for q in qa:
    ti_ = vocab.index(q["typ"])
    TS = S[:, ti_]
    mask = ts < q["tq"]
    idx = np.where(mask)[0]
    if len(idx) < 5: continue
    th = np.quantile(TS[idx], 0.9)
    hits = sorted([i for i in idx if TS[i] >= th], key=lambda i: ts[i])
    evs = []
    for i in hits:
        if evs and ts[i] - ts[evs[-1][-1]] <= 20: evs[-1].append(i)
        else: evs.append([i])
    evs = [e for e in evs if len(e) >= 2] or [hits[-3:]] if hits else []
    if not evs: continue
    use = evs[-1] if q["kind"] == "now" else (evs[-2] if len(evs) >= 2 else evs[-1])
    pred = loc(sorted(use, key=lambda i: -TS[i])[:3], ti_)
    n[q["kind"]] += 1
    ok[q["kind"]] += (pred == q["gt"])
for k in ("now", "before"):
    if n[k]:
        print("%-7s %d/%d = %.3f  (우연 %.2f)" % (k, ok[k], n[k], ok[k]/n[k], 1/len(rids)))
print("전체    %d/%d = %.3f" % (sum(ok.values()), sum(n.values()),
                                sum(ok.values())/max(sum(n.values()), 1)))
