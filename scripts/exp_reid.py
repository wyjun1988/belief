# re-ID: 같은 타입이 **여럿인** 타겟에 대해, 저장해둔 exemplar 이미지가
# "그 머그컵" 을 다른 머그컵과 구별하나. 글자 질의는 원리적으로 못 하는 일이다.
import json, glob, os, numpy as np, torch
from PIL import Image
from collections import Counter
from transformers import Owlv2Processor, Owlv2ForObjectDetection
DEV = "mps"; CK = "google/owlv2-base-patch16-ensemble"; STRIDE = 8
pr = Owlv2Processor.from_pretrained(CK)
md = Owlv2ForObjectDetection.from_pretrained(CK).to(DEV).eval()
def words(t): return "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
def feats(ims):
    pv = pr(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad():
        fm = md.image_embedder(pixel_values=pv)[0]
        b, ph, pw, hd = fm.shape
        f = fm.reshape(b, ph*pw, hd)
        ce = md.class_head.dense0(f)
        ce = ce / (torch.linalg.norm(ce, dim=-1, keepdim=True) + 1e-6)
    return f, ce, ph, pw

out = {}
for hd in sorted(glob.glob("data/thor2z/house_*")):
    hn = os.path.basename(hd)
    g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    typ = {o: v["type"] for o, v in g["gt0"].items()}
    cnt = Counter(typ.values())
    # ⚠️ 여기서는 **중복 타입만** 본다 — 단일 타입은 imgq.py 가 이미 쟀다
    dupo = [o for o, v in g["gt0"].items() if v["room"] and cnt[v["type"]] > 1]
    mps = sorted(glob.glob(hd + "/map/*.jpg"))
    best = {}
    for k, m in enumerate(g["map"]):
        for oid, b in m.get("box", {}).items():
            if oid not in dupo: continue
            a = (b[2]-b[0]) * (b[3]-b[1])
            if a > best.get(oid, (0,))[0]: best[oid] = (a, k, b)
    tgts = [o for o in dupo if o in best]
    if not tgts: continue
    QE = []
    for oid in tgts:
        _, k, b = best[oid]
        im = Image.open(mps[k]).convert("RGB"); W, H = im.size
        f, ce, ph, pw = feats([im])
        cy = min(max(int((b[1]+b[3])/2 / H * ph), 0), ph-1)
        cx = min(max(int((b[0]+b[2])/2 / W * pw), 0), pw-1)
        QE.append(ce[0, cy*pw + cx])
    QE = torch.stack(QE)
    ti = pr(text=[["a photo of a " + words(typ[o]) for o in tgts]],
            images=[Image.new("RGB", (256,256), (128,)*3)], return_tensors="pt").to(DEV)
    with torch.no_grad():
        o_ = md.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                      pixel_values=ti["pixel_values"], return_dict=True)
    TX = o_.text_embeds[0]; MK = torch.ones(len(tgts), dtype=torch.bool, device=DEV)
    lv = sorted(glob.glob(hd + "/live/*.jpg"))[::STRIDE]
    ts = [int(os.path.basename(p)[:-4]) for p in lv]
    ST, SI = [], []
    for i in range(0, len(lv), 4):
        ims = [Image.open(p).convert("RGB") for p in lv[i:i+4]]
        f, ce, ph, pw = feats(ims)
        for Q, acc in ((TX, ST), (QE, SI)):
            with torch.no_grad():
                lg, _ = md.class_head(f, Q.unsqueeze(0).expand(len(ims), -1, -1),
                                      MK.unsqueeze(0).expand(len(ims), -1))
            acc.append(torch.sigmoid(lg).amax(1).float().cpu().numpy())
        if i % 160 == 0: print("  %s %d/%d" % (hn, i, len(lv)), flush=True)
    ST = np.concatenate(ST); SI = np.concatenate(SI)
    moves = sorted(g["moves"], key=lambda m: m["t"])
    for j, oid in enumerate(tgts):
        mv = [x for x in moves if x["oid"] == oid]; t0 = mv[-1]["t"] if mv else 0
        ok = [i for i, t in enumerate(ts) if t > t0]
        if len(ok) < 30: continue
        vis = np.array([oid in live[ts[i]].get("vis", []) for i in ok])
        # **같은 타입의 다른 인스턴스**가 보이는 프레임 (re-ID 가 가려야 할 대상)
        oth = np.array([any(typ.get(o) == typ[oid] and o != oid
                            for o in live[ts[i]].get("vis", [])) for i in ok])
        if vis.sum() < 3 or not (oth & ~vis).any(): continue
        for nm, S in (("글자", ST), ("이미지", SI), ("합", ST + SI)):
            sc = S[ok, j]
            for K in (5, 10, 25):
                top = np.argsort(-sc)[:K]
                out.setdefault((nm, K), []).append(vis[top].mean())
                out.setdefault((nm, "혼동%d" % K), []).append((oth[top] & ~vis[top]).mean())
            # 정답 인스턴스 vs **같은 타입 다른 인스턴스** 만 놓고 가르나
            p, n = sc[vis], sc[oth & ~vis]
            if len(n):
                out.setdefault((nm, "auc"), []).append(
                    (p[:, None] > n[None, :]).mean() + .5*(p[:, None] == n[None, :]).mean())
print("\n=== re-ID · 같은 타입이 여럿인 타겟 (n=%d) ===" % len(out[("글자", 5)]))
for nm in ("글자", "이미지", "합"):
    print("  %-4s 같은타입 구별 AUC **%.3f**" % (nm, np.median(out[(nm, "auc")])))
    print("       정밀도  " + " · ".join("@%d %.3f" % (K, np.mean(out[(nm, K)])) for K in (5,10,25)))
    print("       상위K 중 **다른 인스턴스** 비율  "
          + " · ".join("@%d %.3f" % (K, np.mean(out[(nm, "혼동%d" % K)])) for K in (5,10,25)))
