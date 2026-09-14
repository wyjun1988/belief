#!/usr/bin/env python3
"""채택 판정 추론 묶음 — 검증기 통과 프레임 전부를 (참조, 후보) 이미지 쌍으로 굽는다 (2026-09-14).
학습셋과 같은 박스 규칙·같은 크기. GPU 서버에서 한 번에 돌리려고 만든다(M2 1.6 s/장 → 수 시간).
  VERIFY_JSONL=... A3_PREFIX=... python scripts/adopt_pack.py data/hssd_v2 ~/khcache/adopt_infer_v2
→ <out>/images/*.jpg + <out>/items.jsonl [{house,oid,type,t,i,ref,cand}]"""
import os, json, sys, numpy as np
from PIL import Image, ImageDraw
ROOT, OUT = sys.argv[1], os.path.expanduser(sys.argv[2])
VJ = os.path.expanduser(os.environ["VERIFY_JSONL"]); A3P = os.path.expanduser(os.environ["A3_PREFIX"])
VTH, VTH2 = float(os.environ.get("VERIFY_TH", "2.069")), float(os.environ.get("VERIFY_TH2", "0.887"))
W_OUT = int(os.environ.get("IMG_W", "448"))
os.makedirs(OUT + "/images", exist_ok=True)
def save(src, dst, box):
    if os.path.exists(dst): return True
    if not os.path.exists(src): return False
    im = Image.open(src).convert("RGB"); W, H = im.size
    if box:
        x0, y0, x1, y1 = [max(0, min(W - 1 if i % 2 == 0 else H - 1, int(v))) for i, v in enumerate(box)]
        if x1 > x0 and y1 > y0: ImageDraw.Draw(im).rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=max(3, W // 150))
    s = W_OUT / float(W); im.resize((W_OUT, max(8, int(H * s)))).save(dst, quality=90); return True
G = {}; Z = {}; REF = {}; n = 0
fo = open(OUT + "/items.jsonl", "w")
for ln in open(VJ):
    d = json.loads(ln); h, oid = d["house"], d["oid"]
    if h not in G:
        try: G[h] = json.load(open(f"{ROOT}/{h}/gt.json")); Z[h] = np.load(A3P + h + ".npz", allow_pickle=True)
        except Exception: G[h] = None
    if G[h] is None: continue
    g = G[h]; z = Z[h]; ts = z["ts"]; vocab = [str(x) for x in z["vocab"]]; typ = oid.split("|")[0]
    if typ not in vocab: continue
    ti = vocab.index(typ)
    if (h, oid) not in REF:
        c0 = [(m["dist"][oid], k) for k, m in enumerate(g["map"]) if oid in (m.get("ctr") or {}) and oid in (m.get("dist") or {})]
        if not c0: REF[(h, oid)] = None
        else:
            _, k0 = min(c0); mk = g["map"][k0]; bx = (mk.get("box") or {}).get(oid); ctr = mk["ctr"][oid]
            rb = bx if bx else [ctr[0] - 60, ctr[1] - 60, ctr[0] + 60, ctr[1] + 60]
            f = "%s_ref_%s.jpg" % (h, oid.replace("|", "_").replace(" ", "_"))
            REF[(h, oid)] = f if save(f"{ROOT}/{h}/map/%04d.jpg" % k0, OUT + "/images/" + f, rb) else None
    if REF[(h, oid)] is None: continue
    for e in d.get("scored") or []:
        if not (e[1] >= VTH and (len(e) < 3 or e[2] >= VTH2)): continue
        i = int(e[0])
        if i >= len(ts): continue
        t = int(ts[i]); src = f"{ROOT}/{h}/live/%06d.jpg" % t
        if not os.path.exists(src): continue
        W = float(Image.open(src).size[0]); bcx, bcy, bw, bh = [float(x) * W for x in z["bx"][i, ti]]
        box = [bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2] if (bw > 1 and bh > 1) else None
        f = "%s_c%06d_%s.jpg" % (h, t, oid.replace("|", "_").replace(" ", "_"))
        if not save(src, OUT + "/images/" + f, box): continue
        fo.write(json.dumps(dict(house=h, oid=oid, type=typ.replace("_", " ").lower(), t=t, i=i, ref=REF[(h, oid)], cand=f)) + "\n"); n += 1
fo.close(); print("ADOPT_PACK_DONE %d건 → %s" % (n, OUT))
