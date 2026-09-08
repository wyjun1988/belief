#!/usr/bin/env python3
"""앵커 문맥 프레임(무GT): 타겟마다 (1) 스캔 프레임에서 타겟 박스 옆의 가구 검출을 등록부(anchor_registry)와 CLIP 으로 대조해
**기록 앵커 인스턴스**를 정하고, (2) 1fps 앵커 프레임 중 임베딩 카메라방 == 앵커 방이고 그 타입이 검출된 프레임의 크롭을 등록부와 대조해
**같은 인스턴스일 때만** 문맥 프레임으로 모은다(§166-29). 검증기(abs_verify_mlx.py RETR=anchor)가 이 프레임의 앵커 박스 주변에서 타겟 유무를 묻는다.
  THOR_ROOT=... A3_PREFIX=... ROOM_JSONL=... OUT_JSONL=... [PHANTOM_JSON=...] python scripts/anchor_ctx.py
행: {house, oid, type, anchor_id, anchor_type, anchor_room, reg_sim, n_room, n_type, frames:[[t, x0,y0,x1,y1, sim], ...]}"""
import os, json, glob, math, time, collections, numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, CLIPModel, CLIPProcessor
ROOT = os.environ.get("THOR_ROOT", "data/hssd150_all"); A3P = os.path.expanduser(os.environ.get("A3_PREFIX")); RJ = os.environ.get("ROOM_JSONL"); OUTJ = os.environ.get("OUT_JSONL", "/tmp/anchor_ctx.jsonl")
HOUSES = os.environ.get("HOUSES", "").split(); PHJ = os.environ.get("PHANTOM_JSON", os.path.join(ROOT, "phantom_ids.json"))
S_TH = float(os.environ.get("ANCH_STH", "0.15")); ADJ = float(os.environ.get("ANCH_R", "0.25")); TAU = float(os.environ.get("ANCH_TAU", "0.8")); MARGIN = float(os.environ.get("ANCH_MARGIN", "0.03")); KSCAN = int(os.environ.get("K_SCAN", "3"))
EDGE = float(os.environ.get("ANCH_EDGE", "0.01")); BIG = float(os.environ.get("ANCH_BIG", "0.5"))   # 1판 진단(§166-29): 큰 앵커(카운터·테이블)의 일부만 보인 프레임에서 거짓 부재 → 경계에 잘린 박스는 제외(아주 크면 허용)
def same_words(a, b): return bool(set(a.split()) & set(b.split()))                                  # "table lamp" vs "floor lamp": 앵커가 타겟과 닮아 거짓 존재(③ 7건 중 4건)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"); on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
cm = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEV).eval(); cpp = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
PH = json.load(open(PHJ)) if os.path.exists(PHJ) else {}
rooms = collections.defaultdict(dict)
for l in open(RJ):
    r = json.loads(l); rooms[r["house"]][int(r["t"])] = r["room"]
def owl_multi(im, types):
    inp = op(text=[["a photo of a %s" % t for t in types]], images=[im], return_tensors="pt").to(DEV)
    with torch.no_grad(): out = on(**inp)
    W, H = im.size; res = op.post_process_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[H, W]]))[0]
    by = collections.defaultdict(list)
    for b, s, l in zip(res["boxes"], res["scores"], res["labels"]): by[int(l)].append(([float(v) for v in b], float(s)))
    return {types[l]: sorted(v, key=lambda x: -x[1])[:3] for l, v in by.items()}
def crop(im, b, m=0.15):
    x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0; W, H = im.size
    return im.crop((int(max(0, x0 - m * w)), int(max(0, y0 - m * h)), int(min(W, max(x1 + m * w, x0 + 8))), int(min(H, max(y1 + m * h, y0 + 8)))))
def clip_emb(ims):
    out = []
    for i in range(0, len(ims), 32):
        with torch.no_grad(): e = cm.get_image_features(**cpp(images=ims[i:i+32], return_tensors="pt").to(DEV))
        out.append((e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)
def adj(tb, ab):
    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2; acx, acy = (ab[0] + ab[2]) / 2, (ab[1] + ab[3]) / 2
    if ab[0] <= tcx <= ab[2] and ab[1] <= tcy <= ab[3]: return 0.0
    return math.hypot(tcx - acx, tcy - acy)
done = set()
if os.path.exists(OUTJ):
    for l in open(OUTJ): r = json.loads(l); done.add((r["house"], r["oid"]))
out = open(OUTJ, "a")
hds = [os.path.join(ROOT, h) for h in HOUSES] if HOUSES else sorted(glob.glob(os.path.join(ROOT, "house_*")))
for hd in hds:
    hdr = os.path.realpath(hd); hn = os.path.basename(hd); fa = A3P + hn + ".npz"; rp = os.path.join(hdr, "anchor_registry.json")
    if not (os.path.exists(fa) and os.path.exists(rp) and os.path.exists(os.path.join(hdr, "initmap_raw.json"))): print(hn, "재료 없음", flush=True); continue
    za = np.load(fa, allow_pickle=True); S, ts, BX = za["s"], za["ts"], za["bx"]; vocab, nT = list(za["vocab"]), int(za["nT"])
    reg = json.load(open(rp)); rz = np.load(os.path.join(hdr, "anchor_registry.npz")); REMB, RIDS = rz["emb"], list(rz["ids"])
    if not reg: print(hn, "등록부 비어 있음", flush=True); continue
    g = json.load(open(hdr + "/gt.json")); live = {m["t"]: m for m in g["live"]}; raw = json.load(open(hdr + "/initmap_raw.json")); mfs = sorted(glob.glob(os.path.join(hdr, "map", "*.jpg")))
    gf = os.path.join(hdr, "room_groups.json"); gm = json.load(open(gf))["groups"] if os.path.exists(gf) else {}; grp = lambda r: gm.get(r, r) if r else r
    ph = set((PH.get(hn) or {}).get("phantom") or []) | set((PH.get(hn) or {}).get("hosts") or [])
    cnt = collections.Counter(v["type"] for o, v in g["gt0"].items() if o not in ph)
    arm = [grp(rooms[hn].get(int(t), live[int(t)].get("room"))) for t in ts]
    lv = {int(os.path.basename(p)[:-4]): p for p in glob.glob(os.path.join(hdr, "live", "*.jpg"))}
    rtypes = sorted({r["type"] for r in reg}); by_type = {t: [r for r in reg if r["type"] == t] for t in rtypes}
    def match(e, typ):   # → [(id, sim)] 그 타입 인스턴스별 최대 뷰 유사도, 내림차순
        sc = {}
        for j, aid in enumerate(RIDS):
            if aid.split("#")[0] != typ: continue
            sc[aid] = max(sc.get(aid, -1), float(REMB[j] @ e))
        return sorted(sc.items(), key=lambda x: -x[1])
    T0 = time.time(); n_obj = 0; n_anch = 0; n_fr = 0
    # 라이브 크롭 캐시: (프레임 i, 타입) → 임베딩
    live_emb = {}
    for oid, v0 in g["gt0"].items():
        if oid in ph or cnt[v0["type"]] != 1 or not v0["room"] or v0["type"] not in vocab[:nT] or (hn, oid) in done: continue
        ti = vocab.index(v0["type"]); n_obj += 1
        rec = dict(house=hn, oid=oid, type=v0["type"], anchor_id=None, anchor_type=None, anchor_room=None, reg_sim=None, n_room=0, n_type=0, frames=[])
        # (1) 기록 앵커: 타겟 타입 raw 투영점 상위 프레임에서 인접 가구 검출 → 등록부 대조
        pts = sorted(raw.get(v0["type"], []), key=lambda p: -p[2]); ks = []
        for p in pts:
            if int(p[5]) not in ks and int(p[5]) < len(mfs): ks.append(int(p[5]))
            if len(ks) >= KSCAN: break
        votes = collections.defaultdict(float); scan_fr = []          # scan_fr: 타겟이 앵커 옆에 있던 스캔 프레임(전반 증거 = 기록 자체)
        for k in ks:
            im = Image.open(mfs[k]).convert("RGB"); W, H = im.size; det = owl_multi(im, [v0["type"]] + rtypes)
            tb = det.get(v0["type"])
            if not tb or tb[0][1] < 0.1: continue
            tb = tb[0][0]; best = None
            for at in rtypes:
                if at == v0["type"] or same_words(at, v0["type"]): continue          # 타겟 자신·닮은 타입은 앵커가 아니다
                for ab, sc in det.get(at, []):
                    if sc < S_TH: continue
                    d = adj(tb, ab) / max(W, H)
                    if d <= ADJ and (best is None or d < best[0]): best = (d, at, ab)
            if best is None: continue
            e = clip_emb([crop(im, best[2])])[0]; m = match(e, best[1])
            if m and m[0][1] >= TAU:
                ab = best[2]; rx = ((tb[0] + tb[2]) / 2 - ab[0]) / max(1e-6, ab[2] - ab[0]); ry = ((tb[1] + tb[3]) / 2 - ab[1]) / max(1e-6, ab[3] - ab[1])   # 앵커 박스 안 타겟 상대 위치
                votes[m[0][0]] += m[0][1]; scan_fr.append([int(k)] + [round(v, 1) for v in tb] + [m[0][0], round(float(np.clip(rx, -0.5, 1.5)), 3), round(float(np.clip(ry, -0.5, 1.5)), 3)])
        if not votes:
            out.write(json.dumps(rec) + "\n"); continue
        aid = max(votes, key=votes.get); ainfo = next(r for r in reg if r["id"] == aid); at = ainfo["type"]; aroom = grp(ainfo.get("room")); aidx = vocab.index(at) if at in vocab[:nT] else None
        _sf = [f for f in scan_fr if f[5] == aid]
        rec.update(anchor_id=aid, anchor_type=at, anchor_room=aroom, reg_sim=round(votes[aid] / max(1, len(ks)), 3), scan_frames=[f[:5] for f in _sf],
                   rel=[round(float(np.mean([f[6] for f in _sf])), 3), round(float(np.mean([f[7] for f in _sf])), 3)] if _sf else None); n_anch += 1
        if aidx is None or aroom is None:
            out.write(json.dumps(rec) + "\n"); continue
        # (2) 라이브 문맥 프레임: 카메라방 == 앵커 방 · 타입 검출 · 같은 인스턴스
        cand = [i for i in range(len(ts)) if arm[i] == aroom and S[i, aidx] >= S_TH and int(ts[i]) in lv]; rec["n_room"] = int(sum(1 for i in range(len(ts)) if arm[i] == aroom)); rec["n_type"] = len(cand)
        need = [i for i in cand if (i, at) not in live_emb]
        if need:
            ims = []
            for i in need:
                img = Image.open(lv[int(ts[i])]).convert("RGB"); W, H = img.size; cx, cy, bw, bh = [float(x) * max(W, H) for x in BX[i, aidx]]
                ims.append(crop(img, [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
            E = clip_emb(ims)
            for i, e in zip(need, E): live_emb[(i, at)] = e
        single = len(by_type.get(at, [])) == 1
        for i in cand:
            m = match(live_emb[(i, at)], at)
            if not m or m[0][0] != aid or m[0][1] < TAU: continue
            if not single and len(m) > 1 and m[0][1] - m[1][1] < MARGIN: continue
            img_sz = 768; cx, cy, bw, bh = [float(x) * img_sz for x in BX[i, aidx]]
            _x0, _y0, _x1, _y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            if max(bw, bh) < BIG * img_sz and (_x0 <= EDGE * img_sz or _y0 <= EDGE * img_sz or _x1 >= (1 - EDGE) * img_sz or _y1 >= (1 - EDGE) * img_sz): continue   # 잘린 앵커: 타겟 자리가 화면 밖일 수 있다
            rec["frames"].append([int(ts[i]), round(cx - bw / 2, 1), round(cy - bh / 2, 1), round(cx + bw / 2, 1), round(cy + bh / 2, 1), round(m[0][1], 3)])
        n_fr += len(rec["frames"]); out.write(json.dumps(rec) + "\n"); out.flush()
    print("%s: 타겟 %d · 기록 앵커 있음 %d · 문맥 프레임 %d (%.0fs)" % (hn, n_obj, n_anch, n_fr, time.time() - T0), flush=True)
out.close(); print("ANCHOR_CTX_DONE →", OUTJ)
