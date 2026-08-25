#!/usr/bin/env python3
"""**앵커의 배치를 증거로 쓴다.**

사용자 제안: 선별한 프레임에 앵커가 많이 들어 있으니 그 **배치**도 증거로 쓰자.

지금 방식은 가방(bag)이다 — 어떤 타입이 보이나만 세고 기하를 버린다.
그런데 한 프레임에 정적 인스턴스가 14개나 보이고(오픈플랜) 그중 여러 방 것이
섞이므로, 가방만으로는 방이 안 갈린다.

배치를 쓰는 두 가지:
  ② **쌍 거리 정합** — 화면상 가까운 앵커 쌍은 3D 로도 가까워야 한다.
     옆방 문틀이 화면 구석에 걸린 경우, 3D 로는 멀어서 걸러진다.
  ③ **좌우 순서 정합** — 한 시점에서 본 각도 순서는 화면 x 순서와 같다.
     방마다 앵커의 각도 배열이 다르므로 순서가 맞는 방이 정답일 가능성이 크다.
"""
import json, glob, os, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor3")
K = int(os.environ.get("TOPK", "10"))


def ci(a):
    a = np.asarray(a, float); n = len(a)
    if n == 0: return 0., 0., 0.
    b = [a[np.random.randint(0, n, n)].mean() for _ in range(2000)]
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


res = {}
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = "/tmp/a3_%s.npz" % hn, "/tmp/qc_%s.npz" % hn
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT = list(zq["tg"])
    g = json.load(open(hd + "/gt.json")); sm = g.get("scene_meta")
    if not sm or "pos" not in next(iter(sm["static"].values())): continue
    live = {m["t"]: m for m in g["live"]}
    rids = sorted({v["room"] for v in sm["static"].values()})
    if len(rids) < 2: continue
    rtypes = {}; rpos = {}
    for v in sm["static"].values():
        rtypes.setdefault(v["room"], Counter())[v["type"]] += 1
        rpos.setdefault(v["room"], {}).setdefault(v["type"], []).append(v["pos"])
    allt = set().union(*[set(rtypes[r]) for r in rids])
    idf = {t: 1.0/max(sum(t in rtypes[r] for r in rids), 1) for t in allt}
    adj = {r: set() for r in rids}
    for a, b in sm["doors"]:
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    py, px = P // pw, P % pw
    arm = np.array([live[t]["room"] for t in ts])
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]; t0 = mv[-1]["t"] if mv else 0
        gtf = [i for i, t in enumerate(ts) if t > t0 and oid in live[t].get("vis", [])][:K]
        if len(gtf) < 3: continue

        def vote(mode):
            out = []
            for i in gtf:
                cy, cx = py[i, ti], px[i, ti]
                obs = [(vocab[c], px[i, c], py[i, c], float(S[i, c]))
                       for c in range(nT, len(vocab))
                       if S[i, c] >= .05 and vocab[c] in idf]
                if not obs: continue
                sc = {r: 0.0 for r in rids}
                for r in rids:
                    here = [o for o in obs if o[0] in rtypes[r]]
                    if not here: continue
                    if mode == "bag":
                        for t, ax, ay, s in here:
                            d = np.hypot(ay-cy, ax-cx)
                            sc[r] += s * idf[t] / (1.0 + d/6.0)
                    elif mode == "pair":
                        # 화면 거리 ↔ 3D 거리 정합. 화면에서 가까운 쌍이 3D 로도 가까운가
                        acc = 0.0; np_ = 0
                        for a in range(len(here)):
                            for b in range(a+1, len(here)):
                                ta, xa, ya, sa = here[a]; tb, xb, yb, sb = here[b]
                                ds = np.hypot(xa-xb, ya-yb) / pw           # 0~1
                                pa = np.array(rpos[r][ta]); pb = np.array(rpos[r][tb])
                                d3 = np.min(np.linalg.norm(pa[:, None] - pb[None], axis=-1))
                                d3n = min(d3 / 8.0, 1.0)                    # 8m 로 정규화
                                acc += sa*sb*idf[ta]*idf[tb]*(1.0 - abs(ds - d3n))
                                np_ += 1
                        sc[r] = acc / max(np_, 1) * len(here)
                    else:   # order — 화면 x 순서가 3D 각도 순서로 설명되나
                        here2 = sorted(here, key=lambda o: o[1])
                        pts = [np.array(rpos[r][t][0]) for t, *_ in here2]
                        if len(pts) < 3:
                            sc[r] = 0.0; continue
                        best = 0.0
                        for ang in np.linspace(0, np.pi, 12, endpoint=False):
                            u = np.array([np.cos(ang), np.sin(ang)])
                            proj = [float(p @ u) for p in pts]
                            inc = sum(proj[a] < proj[a+1] for a in range(len(proj)-1))
                            best = max(best, max(inc, len(proj)-1-inc) / (len(proj)-1))
                        sc[r] = best * sum(o[3]*idf[o[0]] for o in here)
                nb = {arm[i]} | adj.get(arm[i], set())
                for r in rids:
                    if r not in nb: sc[r] *= .25
                out.append(max(rids, key=lambda r: sc[r]))
            return max(set(out), key=out.count) if out else None

        for mode, nm in (("bag", "① 가방 (지금)"), ("pair", "② +쌍 거리 정합"),
                         ("order", "③ +좌우 순서 정합")):
            res.setdefault(nm, []).append(vote(mode) == tgt)

print("=== 앵커 **배치**를 증거로 (프레임 GT · 상위 %d) ===" % K)
for k in ("① 가방 (지금)", "② +쌍 거리 정합", "③ +좌우 순서 정합"):
    if k not in res: continue
    m, lo, hi = ci(res[k])
    print("  %-20s %.3f [%.3f %.3f]  n=%d" % (k, m, lo, hi, len(res[k])))
