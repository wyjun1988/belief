# **이미지 질의**: 씬그래프를 만들 때 저장해둔 그 물체의 crop 을 exemplar 로 써서
# 프레임을 고른다. 글자 "머그컵" 이 아니라 **그 머그컵**. 타겟은 사용자 지시대로
# 집에 같은 타입이 하나뿐인 것만 — 똑같이 생긴 여럿은 우리 타겟이 아니다.
import json, glob, os, sys, numpy as np, torch
from PIL import Image
from collections import Counter
from transformers import Owlv2Processor, Owlv2ForObjectDetection
ROOT = os.environ.get("THOR_ROOT", "data/thor3")
OUT = os.environ.get("QCACHE_PREFIX", "/tmp/qc_")
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
CK = "google/owlv2-base-patch16-ensemble"
STRIDE = int(os.environ.get("STRIDE", "8"))
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
# 점수는 코사인 유사도(-1..1). 시그모이드 포화 문제로 교체했다.
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    g = json.load(open(hd + "/gt.json")); live = {m["t"]: m for m in g["live"]}
    cnt = Counter(v["type"] for v in g["gt0"].values())
    uniq = [o for o, v in g["gt0"].items() if v["room"] and cnt[v["type"]] == 1]
    mps = sorted(glob.glob(hd + "/map/*.jpg"))
    # ── 씬그래프 구축: 물체마다 가장 큰 bbox 를 가진 맵 프레임에서 exemplar 를 뽑는다 ──
    best = {}
    for k, m in enumerate(g["map"]):
        for oid, b in m.get("box", {}).items():
            if oid not in uniq: continue
            a = (b[2]-b[0]) * (b[3]-b[1])
            if a > best.get(oid, (0,))[0]: best[oid] = (a, k, b)
    tgts = [o for o in uniq if o in best]
    QE = []
    for oid in tgts:
        _, k, b = best[oid]
        im = Image.open(mps[k]).convert("RGB")
        W, H = im.size
        f, ce, ph, pw = feats([im])
        # OWLv2 는 정사각 패딩 후 리사이즈 — 원본이 정사각이라 비율 그대로
        cy = int((b[1]+b[3])/2 / H * ph); cx = int((b[0]+b[2])/2 / W * pw)
        cy = min(max(cy, 0), ph-1); cx = min(max(cx, 0), pw-1)
        QE.append(ce[0, cy*pw + cx])
    QE = torch.stack(QE)                       # (n_tgt, dim) 이미지 질의
    ti = pr(text=[[ "a photo of a " + words(g["gt0"][o]["type"]) for o in tgts]],
            images=[Image.new("RGB", (256,256), (128,)*3)], return_tensors="pt").to(DEV)
    with torch.no_grad():
        o_ = md.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                      pixel_values=ti["pixel_values"], return_dict=True)
    # ⚠️ **`[0]` 을 붙이면 안 된다.** `text_embeds` 는 이미 (질의수, dim) 이라
    # `[0]` 은 **첫 물체의 질의 하나**를 뽑아 전체에 복사한다. 실제로 물렸다 —
    # 캐시 19열의 열간 상관이 1.000 이었고, 그 탓에 "글자 @5 0.133" 이 나왔다.
    # 이미지 쪽(열간 상관 0.808)은 정상이었으므로 이미지 수치만 유효했다.
    TX = o_.text_embeds
    assert TX.shape[0] == len(tgts), "글자 질의 %d != 타겟 %d" % (TX.shape[0], len(tgts))
    MK = torch.ones(len(tgts), dtype=torch.bool, device=DEV)
    lv = sorted(glob.glob(hd + "/live/*.jpg"))[::STRIDE]
    ts = [int(os.path.basename(p)[:-4]) for p in lv]
    ST, SI = [], []
    for i in range(0, len(lv), 4):
        ims = [Image.open(p).convert("RGB") for p in lv[i:i+4]]
        f, ce, ph, pw = feats(ims)
        # ⚠️ **시그모이드를 쓰면 안 된다.** class_head 의 logit_shift/logit_scale 은
        # 텍스트 임베딩 스케일에 맞춰 학습됐다. exemplar 임베딩을 넣으면 스케일이 달라
        # 시그모이드가 포화한다(실측 중앙 0.994, 90분위 하락이 전부 0.0000).
        # 순위는 살아남아 정밀도@5·AUC 는 유효했지만, **절댓값 비교가 필요한 부재
        # 검출에서 신호가 통째로 죽는다.** 정규화 내적(코사인)을 그대로 쓴다.
        ce_n = ce                                    # 이미 L2 정규화됨
        for Q, acc in ((TX, ST), (QE, SI)):
            Qn = Q / (torch.linalg.norm(Q, dim=-1, keepdim=True) + 1e-6)
            with torch.no_grad():
                sim = ce_n @ Qn.transpose(0, 1)      # (batch, 패치, 질의)
            acc.append(sim.amax(1).float().cpu().numpy())
        if i % 160 == 0: print("  %s %d/%d" % (hn, i, len(lv)), flush=True)
    ST = np.concatenate(ST); SI = np.concatenate(SI)
    np.savez_compressed(OUT + "%s.npz" % hn, st=ST, si=SI, ts=np.array(ts),
                        tg=np.array(tgts, object))
    moves = sorted(g["moves"], key=lambda m: m["t"])
    for j, oid in enumerate(tgts):
        mv = [x for x in moves if x["oid"] == oid]; t0 = mv[-1]["t"] if mv else 0
        ok = [i for i, t in enumerate(ts) if t > t0]
        if len(ok) < 30: continue
        vis = np.array([oid in live[ts[i]].get("vis", []) for i in ok])
        if vis.sum() < 3 or vis.all(): continue
        for nm, S in (("글자", ST), ("이미지", SI), ("합", ST + SI)):
            sc = S[ok, j]
            for K in (5, 10, 25):
                out.setdefault((nm, K), []).append(vis[np.argsort(-sc)[:K]].mean())
            p, n = sc[vis], sc[~vis]
            out.setdefault((nm, "auc"), []).append(
                (p[:, None] > n[None, :]).mean() + .5*(p[:, None] == n[None, :]).mean())
        out.setdefault(("기저", 0), []).append(vis.mean())
print("\n=== 이미지 질의 (타입 단일 타겟만 · n=%d) ===" % len(out[("글자", 5)]))
print("  기저(보이는 비율) %.3f" % np.mean(out[("기저", 0)]))
for nm in ("글자", "이미지", "합"):
    print("  %-4s AUC %.3f · " % (nm, np.median(out[(nm, "auc")]))
          + " · ".join("**@%d %.3f**" % (K, np.mean(out[(nm, K)])) for K in (5, 10, 25)))
