#!/usr/bin/env python3
"""개체 재식별 검증기 후보 비교 — 같은 후보 크롭에서 DINOv2 크롭 코사인 vs VLM s_ab.

    THOR_ROOT=data/hssd20S2 A3_PREFIX=... SCORES=t1.jsonl python scripts/reid_compare.py

exemplar = 매핑워크에서 그 물체 bbox 가 가장 큰 프레임의 크롭. 후보 = 채점 파이프라인과
같은 OWL 박스 크롭. DINOv2-base CLS/평균패치 코사인의 AUC 를 s_ab 와 나란히 놓는다.
로그: OWL 임베딩 재식별(exnew)은 +0.03 뿐이었다(§112 부근). DINOv2 크롭 재식별은 미시도.
"""
import glob, json, os
import numpy as np, torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = os.environ.get("THOR_ROOT", "data/hssd20S2")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "/tmp/hs2_a_"))
SC = os.environ.get("SCORES", "/tmp/t1_scores.jsonl")
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEV).eval()

def feats(im):
    x = proc(images=im, return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad(): h = dino(pixel_values=x).last_hidden_state[0]
    cls = h[0]; pm = h[1:].mean(0)
    return (cls / cls.norm()).cpu().numpy(), (pm / pm.norm()).cpu().numpy()

def auc(y, x):
    y = np.asarray(y, bool); x = np.asarray(x, float)
    if y.all() or (~y).all(): return float("nan")
    order = np.argsort(x); r = np.empty(len(x)); r[order] = np.arange(len(x))
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    for u in np.where(cnt > 1)[0]:
        m = inv == u; r[m] = r[m].mean()
    return float((r[y].sum() - y.sum() * (y.sum() - 1) / 2) / (y.sum() * (~y).sum()))

def crop_box(im, cx, cy, w, h):
    W, H = im.size; h2 = max(48, int(max(w, h) * 0.65))
    return im.crop((max(0, int(cx)-h2), max(0, int(cy)-h2), min(W, int(cx)+h2), min(H, int(cy)+h2)))

rows = []  # truth, hit, s_ab, s_ac, dino_cls, dino_pm
for rc in [json.loads(l) for l in open(SC)]:
    hn, oid = rc["house"], rc["oid"]
    hd = [d for d in glob.glob(ROOT + "/house_*") if os.path.basename(os.path.realpath(d)) == hn]
    if not hd: continue
    hd = hd[0]; g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    mvt = {m["oid"]: m["t"] for m in g["moves"]}
    if oid not in mvt: continue
    za = np.load(A3P + hn + ".npz", allow_pickle=True); ts, vocab, bx = za["ts"], list(za["vocab"]), za["bx"]
    ti = vocab.index(g["gt0"][oid]["type"])
    # exemplar: 매핑워크에서 bbox 최대 프레임
    best = None
    for k, mp in enumerate(g.get("map", [])):
        b = (mp.get("box") or {}).get(oid)
        if b and (best is None or (b[2]-b[0])*(b[3]-b[1]) > best[0]): best = ((b[2]-b[0])*(b[3]-b[1]), k, b)
    if best is None: continue
    mfs = sorted(glob.glob(os.path.join(hd, "map", "*.jpg")))
    if best[1] >= len(mfs): continue
    b = best[2]; ex = Image.open(mfs[best[1]]).convert("RGB").crop((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
    ec, ep = feats(ex)
    for i, s_ab, s_ac in rc["scored"]:
        t = int(ts[i]); m = live.get(t, {})
        truth = t > mvt[oid] and oid in (m.get("vis") or [])
        bcx, bcy, bw, bh = [float(v) * 768 for v in bx[i, ti]]
        c = (m.get("ctr") or {}).get(oid); hit = bool(truth and c and np.hypot(bcx - c[0], bcy - c[1]) <= 60)
        im = Image.open(os.path.join(hd, "live", "%06d.jpg" % t)).convert("RGB")
        cc, cp = feats(crop_box(im, bcx, bcy, bw, bh))
        rows.append((truth, hit, s_ab, s_ac, float(cc @ ec), float(cp @ ep)))
R = np.array(rows, float); y = R[:, 0] > 0; hit = R[:, 1] > 0; neg = ~y
print("크롭 %d · 진짜 %d (박스적중 %d)" % (len(R), y.sum(), hit.sum()))
print("%-18s %-14s %-20s %-12s" % ("점수", "풀AUC(진짜/전체)", "AUC(박스적중 vs 오검출)", "수용@기각.95"))
for c, nm in ((2, "s_ab(VLM)"), (3, "s_ac(VLM)"), (4, "DINOv2 CLS 코사인"), (5, "DINOv2 패치평균 코사인")):
    x = R[:, c]; th = np.quantile(x[neg], 0.95)
    yy = np.concatenate([np.ones(hit.sum(), bool), np.zeros(neg.sum(), bool)]); xx = np.concatenate([x[hit], x[neg]])
    print("%-18s %-14.3f %-20.3f %-12.3f" % (nm, auc(y, x), auc(yy, xx), float((x[hit] >= th).mean())))
# 결합: z(s_ab)+z(DINO cls)
z = (R[:, 2:] - R[:, 2:].mean(0)) / (R[:, 2:].std(0) + 1e-9)
for nm, xz in (("z(s_ab)+z(DINO cls)", z[:, 0] + z[:, 2]), ("z(s_ac)+z(DINO cls)", z[:, 1] + z[:, 2])):
    th = np.quantile(xz[neg], 0.95)
    print("%-18s %-14.3f %-20s %-12.3f" % (nm, auc(y, xz), "—", float((xz[hit] >= th).mean())))
