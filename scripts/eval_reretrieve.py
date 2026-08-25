#!/usr/bin/env python3
"""**2차 검색으로 증거 프레임을 늘린다.**

사용자 제안: 방을 특정할 때, 1차로 찾은 프레임에서 앵커를 뽑고 **그 앵커로 한 번 더
검색해** 같은 자리를 보여주는 프레임을 모은다. 그걸로 방 판정 증거를 늘린다.

⚠️ 왜 될 법한가 — 타겟은 37장에만 찍히지만 **자리(정적 물체)는 안 움직이므로 훨씬
자주 찍힌다.** 그리고 여러 프레임에서 **일관되게** 함께 나오는 앵커만 남기면, 우연히
화면 구석에 걸린 옆방 문틀이 걸러진다(프레임당 정적 인스턴스가 14개나 보인다).

  1차   타겟 점수 상위 K 프레임
  자리 서명  그 프레임들에서 타겟에 **화면상 가까운** 앵커 타입을 가중 집계
  2차   전 프레임을 자리 서명과의 일치도로 재정렬 → 상위 M
  판정   2차 프레임들의 앵커 증거로 방 투표
"""
import json, glob, os, sys, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor3")
K = int(os.environ.get("TOPK", "10"))      # 1차
M = int(os.environ.get("TOPM", "40"))      # 2차
GTFRAME = os.environ.get("GTFRAME", "1") == "1"


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
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json")); sm = g.get("scene_meta")
    if not sm: continue
    live = {m["t"]: m for m in g["live"]}
    rids = sorted({v["room"] for v in sm["static"].values()})
    if len(rids) < 2: continue
    rtypes = {}
    for v in sm["static"].values():
        rtypes.setdefault(v["room"], Counter())[v["type"]] += 1
    allt = set().union(*[set(rtypes[r]) for r in rids])
    idf = {t: 1.0/max(sum(t in rtypes[r] for r in rids), 1) for t in allt}
    adj = {r: set() for r in rids}
    for a, b in sm["doors"]:
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    AS = S[:, nT:]                                   # 정적 타입 점수
    ANorm = AS / (np.linalg.norm(AS, axis=1, keepdims=True) + 1e-9)
    py, px = P // pw, P % pw
    arm = np.array([live[t]["room"] for t in ts])
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1: continue
        if v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]; t0 = mv[-1]["t"] if mv else 0
        ok = np.array([i for i, t in enumerate(ts) if t > t0])
        if len(ok) < 50: continue
        if GTFRAME:
            first = np.array([i for i in ok if oid in live[ts[i]].get("vis", [])])[:K]
        else:
            first = ok[np.argsort(-(QS[ok, j] + STx[ok, j]))[:K]]
        if len(first) < 3: continue

        def roomvote(idx, prox):
            """앵커 증거로 방 투표. prox=True 면 타겟 화면위치 근접 가중."""
            out = []
            for i in idx:
                sc = {r: 0.0 for r in rids}
                cy, cx = py[i, ti], px[i, ti]
                for c in range(nT, len(vocab)):
                    t = vocab[c]
                    if t not in idf or S[i, c] < .05: continue
                    w = float(S[i, c]) * idf[t]
                    if prox:
                        d = np.hypot(py[i, c]-cy, px[i, c]-cx); w /= (1.0 + d/6.0)
                    for r in rids:
                        if t in rtypes[r]: sc[r] += w
                nb = {arm[i]} | adj.get(arm[i], set())
                for r in rids:
                    if r not in nb: sc[r] *= .25
                out.append(max(rids, key=lambda r: sc[r]))
            return max(set(out), key=out.count) if out else None

        # ① 기준선 — 1차 프레임만
        res.setdefault("① 1차 프레임만 (지금)", []).append(roomvote(first, True) == tgt)

        # ② 자리 서명 → 2차 검색 → 늘어난 프레임으로 투표
        sig = np.zeros(len(vocab) - nT)
        for i in first:
            cy, cx = py[i, ti], px[i, ti]
            for c in range(nT, len(vocab)):
                if S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                sig[c-nT] += float(S[i, c]) / (1.0 + d/6.0)
        if sig.sum() <= 0:
            res.setdefault("② +2차 검색", []).append(False)
            res.setdefault("③ +일관성 필터", []).append(False); continue
        sig /= np.linalg.norm(sig)
        sim = ANorm[ok] @ sig
        second = ok[np.argsort(-sim)[:M]]
        res.setdefault("② +2차 검색", []).append(roomvote(second, False) == tgt)

        # ③ 2차 프레임에서 **일관되게** 나오는 앵커만 남겨 서명을 정제 → 재투표
        keep = (AS[second] >= .05).mean(0) >= 0.5          # 절반 이상에서 검출된 타입
        sig2 = sig * keep
        if sig2.sum() > 0:
            sig2 /= np.linalg.norm(sig2)
            third = ok[np.argsort(-(ANorm[ok] @ sig2))[:M]]
            sc = {r: 0.0 for r in rids}
            for c in np.where(keep)[0]:
                t = vocab[nT + c]
                if t not in idf: continue
                for r in rids:
                    if t in rtypes[r]: sc[r] += float(sig[c]) * idf[t]
            res.setdefault("③ +일관성 필터", []).append(max(rids, key=lambda r: sc[r]) == tgt)
        else:
            res.setdefault("③ +일관성 필터", []).append(False)

print("=== 2차 검색으로 증거 늘리기 (프레임 %s · 1차 %d → 2차 %d) ==="
      % ("GT" if GTFRAME else "실전", K, M))
for k in ("① 1차 프레임만 (지금)", "② +2차 검색", "③ +일관성 필터"):
    if k not in res: continue
    m, lo, hi = ci(res[k])
    print("  %-22s %.3f [%.3f %.3f]  n=%d" % (k, m, lo, hi, len(res[k])))
