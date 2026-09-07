#!/usr/bin/env python3
"""카메라방을 **프레임 임베딩**으로 정한다 — 매핑워크 방 노드 최대 유사도 + Viterbi (§82 THOR 0.964 의 HSSD 이식).

    THOR_ROOT=data/hssd40_c3 HOUSES=4 MODEL=clip OUT_JSONL=~/khcache/room_embed_clip.jsonl python scripts/room_embed.py

왜: ③ 부재 게이팅은 앵커 프레임 300장 전부의 **방**을 요구하지만 좌표는 필요 없다(§165-5). SfM/PnP 를 300장에 돌리는
대신 프레임당 임베딩 하나(수십 ms)로 방을 정한다. 어제의 room_retrieval.py(OWL 검출 프로파일, 0.217)는 §82 가
계통 오류로 판정한 "검출 점수 방출" 계열이었다 — 임베딩 방식과 다르다.
  방출  em[t, r] = max_{n∈방 r 의 매핑워크 프레임} cos(live_t, node_n)       (§57/§82 그대로)
  전이  Viterbi(stay=STAY, temp=TEMP) — 사람은 순간이동하지 않는다. 값은 §82 의 것을 **미리 고정**(GT 로 고르지 않는다).
입력 GT 0: 매핑워크 방 라벨은 사용자 입력(스캔 때 "여기는 부엌"), live GT 방은 채점에만 쓴다.
산출: jsonl {house, t, room, sim, room_argmax} — eval_online ROOM_JSONL (사다리 '카메라방:검색').
"""
import glob, json, os, sys, time
import numpy as np, torch
from PIL import Image

ROOT = os.environ.get("THOR_ROOT", "data/hssd40_c3"); HOUSES = int(os.environ.get("HOUSES", "4"))
MODEL = os.environ.get("MODEL", "clip"); STRIDE = int(os.environ.get("STRIDE", "1")); MWSTRIDE = int(os.environ.get("MWSTRIDE", "1"))
TEMP = float(os.environ.get("TEMP", "0.01")); STAY = float(os.environ.get("STAY", "0.9"))
EMIT = os.environ.get("EMIT", "max"); KNN = int(os.environ.get("KNN", "10")); CENTER = os.environ.get("CENTER", "0") == "1"
# EMIT: max=방 노드 최대 유사도(§57) · knn=상위 K 노드의 방 투표(유사도 가중) — max 는 "빈 벽" 같은 특징 없는 노드 한 장에 끌린다
# CENTER: 노드 평균 임베딩을 빼고 코사인 — 도메인 공통 성분(렌더 질감)을 걷어내 장소 특이 성분만 남긴다
OUTJ = os.path.expanduser(os.environ.get("OUT_JSONL", "/tmp/room_embed_%s.jsonl" % MODEL))
CACHE = os.path.expanduser(os.environ.get("EMB_CACHE", "~/khcache/room_embed"))   # 임베딩 캐시(모델별·채별)
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs(CACHE, exist_ok=True)

if MODEL == "clip":
    from transformers import CLIPModel, CLIPProcessor
    CK = "openai/clip-vit-base-patch16"; pr = CLIPProcessor.from_pretrained(CK); md = CLIPModel.from_pretrained(CK).to(DEV).eval()
    def _emb(ims):
        return md.get_image_features(**pr(images=ims, return_tensors="pt").to(DEV))
elif MODEL == "siglip":
    from transformers import AutoModel, AutoProcessor
    CK = os.environ.get("SIGLIP_CK", "google/siglip-base-patch16-224"); pr = AutoProcessor.from_pretrained(CK); md = AutoModel.from_pretrained(CK).to(DEV).eval()
    def _emb(ims):
        return md.get_image_features(**pr(images=ims, return_tensors="pt").to(DEV))
elif MODEL == "dinov2":
    from transformers import AutoModel, AutoImageProcessor
    CK = "facebook/dinov2-base"; pr = AutoImageProcessor.from_pretrained(CK); md = AutoModel.from_pretrained(CK).to(DEV).eval()
    def _emb(ims):
        return md(**pr(images=ims, return_tensors="pt").to(DEV)).pooler_output
else:
    sys.exit("MODEL ∈ clip|siglip|dinov2")
print("방 임베딩 · %s (%s) · %s · temp %.3f stay %.2f · emit %s(k%d) center %d" % (MODEL, CK, DEV, TEMP, STAY, EMIT, KNN, int(CENTER)), flush=True)

def emb(paths, tag):
    cf = os.path.join(CACHE, "%s_%s.npy" % (MODEL, tag))
    if os.path.exists(cf):
        e = np.load(cf)
        if len(e) == len(paths): return e
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 16]]
        with torch.no_grad(): e = _emb(ims)
        e = e / e.norm(dim=-1, keepdim=True); out.append(e.float().cpu().numpy())
    E = np.concatenate(out) if out else np.zeros((0, 1), np.float32); np.save(cf, E); return E

def viterbi(S, stay, temp):
    Z = (S - S.max(1, keepdims=True)) / temp
    logem = Z - np.log(np.exp(Z).sum(1, keepdims=True) + 1e-12)
    T, K = logem.shape
    tr = np.full((K, K), np.log((1 - stay) / max(K - 1, 1))); np.fill_diagonal(tr, np.log(stay))
    dp = logem[0].copy(); bp = np.zeros((T, K), int)
    for t in range(1, T):
        m = dp[:, None] + tr; bp[t] = np.argmax(m, 0); dp = m.max(0) + logem[t]
    path = np.zeros(T, int); path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1): path[t - 1] = bp[t, path[t]]
    return path

fo = open(OUTJ, "w"); TOT = {"argmax": [0, 0], "viterbi": [0, 0]}; T0 = time.time()
sweep = {}
for hd in sorted(glob.glob(ROOT + "/house_*"))[:HOUSES]:
    rd = os.path.realpath(hd); hn = os.path.basename(rd)
    g = json.load(open(os.path.join(rd, "gt.json")))
    mp = sorted(glob.glob(os.path.join(rd, "map", "*.jpg")))
    if len(g["map"]) != len(mp): print("  %s gt.map %d ≠ map %d — 건너뜀" % (hn, len(g["map"]), len(mp))); continue
    mwp = mp[::MWSTRIDE]; mwr = [m["room"] for m in g["map"]][::MWSTRIDE]
    lv = sorted(glob.glob(os.path.join(rd, "live", "*.jpg")))[::STRIDE]; ts = [int(os.path.basename(p)[:-4]) for p in lv]
    t1 = time.time(); ME = emb(mwp, hn + "_map"); LE = emb(lv, hn + "_live"); dt = time.time() - t1
    rids = sorted(set(mwr))
    ME2, LE2 = ME, LE
    if CENTER:
        mu = ME.mean(0, keepdims=True); ME2 = ME - mu; LE2 = LE - mu
        ME2 = ME2 / (np.linalg.norm(ME2, axis=1, keepdims=True) + 1e-9); LE2 = LE2 / (np.linalg.norm(LE2, axis=1, keepdims=True) + 1e-9)
    sim = LE2 @ ME2.T                                                          # (L, M)
    em = np.zeros((len(LE), len(rids)), np.float32); ridx = {r: k for k, r in enumerate(rids)}
    if EMIT == "max":
        for k, r in enumerate(rids):
            idx = [i for i, x in enumerate(mwr) if x == r]; em[:, k] = sim[:, idx].max(1)
    elif EMIT == "pos":
        # 상위 K 노드 **프레임의 위치**(스캔 포즈 — 초기 스캔 단계라 허용) 중앙값 = 카메라 위치 추정 → 평면도 폴리곤으로 방.
        # 라벨 투표와 다른 점: 거실에서 부엌을 바라본 live 는 "부엌처럼 보이는, 거실에서 찍은" 노드와 맞고 그 노드의 위치는 거실이다.
        polys = (g.get("scene_meta") or {}).get("polys") or {}
        def _pip(pt, pl):
            x, z = pt; ins = False; n = len(pl)
            for i_ in range(n):
                x1, z1 = pl[i_][0], pl[i_][-1]; x2, z2 = pl[(i_ + 1) % n][0], pl[(i_ + 1) % n][-1]
                if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
            return ins
        def _room_of(pt):
            for r_, pl in polys.items():
                if _pip(pt, pl): return r_
            return min(polys, key=lambda r_: min((pt[0] - v[0]) ** 2 + (pt[1] - v[-1]) ** 2 for v in polys[r_])) if polys else None
        mpos = np.array([m["apos"] for m in g["map"]])[::MWSTRIDE]
        top = np.argsort(-sim, axis=1)[:, :KNN]
        rids = sorted(set(list(polys.keys()) + rids)); ridx = {r: k for k, r in enumerate(rids)}; em = np.zeros((len(LE), len(rids)), np.float32)
        for i in range(len(LE)):
            pt = np.median(mpos[top[i]], 0); r_ = _room_of(pt)
            if r_ is not None: em[i, ridx[r_]] = 1.0
            # 소프트 방출: 상위 K 각 노드의 위치가 든 방에 유사도 가중 (Viterbi 재료)
            for j in top[i]:
                rj = _room_of(mpos[j])
                if rj is not None: em[i, ridx[rj]] += max(sim[i, j], 0) * 0.5
        em = em / (em.sum(1, keepdims=True) + 1e-9)
    else:                                                                       # knn: 상위 K 노드의 방에 유사도 가중 투표
        top = np.argsort(-sim, axis=1)[:, :KNN]
        for i in range(len(LE)):
            for j in top[i]: em[i, ridx[mwr[j]]] += max(sim[i, j], 0)
        em = em / (em.sum(1, keepdims=True) + 1e-9)                              # 확률 비슷하게
    pa = em.argmax(1); pv = viterbi(em, STAY, TEMP)
    live = {m["t"]: m for m in g["live"]} if isinstance(g["live"], list) else g["live"]
    gt = np.array([live[t]["room"] if t in live else None for t in ts], object)
    ok = gt != None
    # 열린 공간은 한 방(사용자 규칙): <house>/room_groups.json 이 있으면 예측·GT 를 그룹으로 사상해 채점(GROUPS=1 기본)
    _gf = os.path.join(rd, "room_groups.json"); _gm = json.load(open(_gf))["groups"] if (os.environ.get("GROUPS", "1") == "1" and os.path.exists(_gf)) else {}
    _grp = lambda r_: _gm.get(r_, r_)
    if _gm: gt = np.array([_grp(x) if x is not None else None for x in gt], object)
    PA = np.array([_grp(rids[k]) for k in pa], object); PV = np.array([_grp(rids[k]) for k in pv], object)
    acc_a = float(np.mean(PA[ok] == gt[ok])); acc_v = float(np.mean(PV[ok] == gt[ok]))
    TOT["argmax"][0] += int((PA[ok] == gt[ok]).sum()); TOT["argmax"][1] += int(ok.sum())
    TOT["viterbi"][0] += int((PV[ok] == gt[ok]).sum()); TOT["viterbi"][1] += int(ok.sum())
    for te in (0.005, 0.01, 0.02, 0.05):
        for st in (0.8, 0.9, 0.95, 0.98):
            p_ = viterbi(em, st, te); P_ = np.array([_grp(rids[k]) for k in p_], object); sweep.setdefault((te, st), [0, 0]); sweep[(te, st)][0] += int((P_[ok] == gt[ok]).sum()); sweep[(te, st)][1] += int(ok.sum())
    for i, t in enumerate(ts):
        fo.write(json.dumps({"house": hn, "t": int(t), "room": rids[pv[i]], "sim": float(em[i, pv[i]]), "room_argmax": rids[pa[i]]}) + "\n")
    # 방별 혼동 요약(GT 방 → 예측 최빈)
    conf = {}
    for r_ in sorted(set(gt[ok])):
        sel = (gt == r_) & ok; pr_ = PV[sel]
        u, c = np.unique(pr_, return_counts=True); conf[r_] = "%s %.2f" % (u[c.argmax()], c.max() / len(pr_))
    print("  %s 노드 %d(방 %d%s) · live %d · 임베딩 %.0fs(%.0fms/장) · GT 일치 argmax %.3f · **Viterbi %.3f** · %s" % (
        hn, len(ME), len(rids), ("→그룹 %d" % len(set(_gm.values()))) if _gm else "", len(LE), dt, 1000 * dt / max(len(ME) + len(LE), 1), acc_a, acc_v, conf), flush=True)
fo.close()
print("전체 GT 일치: argmax %.3f · **Viterbi %.3f** (n=%d) → %s · %.0fs" % (TOT["argmax"][0] / max(TOT["argmax"][1], 1), TOT["viterbi"][0] / max(TOT["viterbi"][1], 1), TOT["viterbi"][1], OUTJ, time.time() - T0), flush=True)
print("(진단) temp×stay 스윕:", " ".join("%.3f/%.2f=%.3f" % (te, st, v[0] / max(v[1], 1)) for (te, st), v in sorted(sweep.items())), flush=True)
