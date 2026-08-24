#!/usr/bin/env python3
"""**갱신이 무엇을 더하나.** 정지된 지도 대비 동적 갱신의 기여를 분리해 잰다.

    $P scripts/eval_update.py [글자|이미지]

⚠️ **왜 이렇게 재야 하나.** 전체 타겟에 대고 재면 "씬그래프 방 그대로" 가 0.859 로
나온다. 1시간에 8건만 움직이니 정지된 지도가 236/276 을 맞히기 때문이다. 그 위에서
어떤 갱신을 해도 이득이 안 보인다(실측 0.855).

그런데 **사람은 안 움직인 물건을 안 묻는다.** "안경 어디 뒀지?" 는 늘 있던 자리에
없어서 묻는 것이다. 갱신의 가치는 물건이 움직였을 때만 드러나므로 그 구간을 따로 본다.

이동 물체에서는 "씬그래프 방 그대로" 가 **정의상 0** 이다. 거기서 나오는 점수는
전부 갱신이 번 것이다.
"""
import json, glob, os, sys, numpy as np
from collections import Counter

MODE = sys.argv[1] if len(sys.argv) > 1 else "이미지"
ROOT = os.environ.get("THOR_ROOT", "data/thor3")
PR = json.load(open("data/thor_prior.json"))
MV = json.load(open("data/thor_move.json"))


def load():
    rows = []
    for hd in sorted(glob.glob(ROOT + "/house_*")):
        hn = os.path.basename(os.path.realpath(hd))
        fa, fq = "/tmp/a3_%s.npz" % hn, "/tmp/q3_%s.npz" % hn
        if not (os.path.exists(fa) and os.path.exists(fq)):
            continue
        za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
        S, ts, vocab = za["s"], za["ts"], list(za["vocab"])
        QT, QS = list(zq["tg"]), zq["si"]
        g = json.load(open(hd + "/gt.json")); rt = g["room_types"]; rids = sorted(rt)
        live = {m["t"]: m for m in g["live"]}
        nrt = Counter(rt[r] for r in rids)
        moves = sorted(g["moves"], key=lambda m: m["t"])
        cnt = Counter(v["type"] for v in g["gt0"].values())
        for oid, v in g["gt0"].items():
            if not v["room"] or cnt[v["type"]] > 1 or oid not in QT:
                continue
            if v["type"] not in vocab:
                continue
            ti = vocab.index(v["type"]); qi = QT.index(oid)
            sg = v["room"]
            mv = [x for x in moves if x["oid"] == oid]
            tgt = mv[-1]["to"] if mv else sg
            t0 = mv[-1]["t"] if mv else 0
            TS = S[:, ti] if MODE == "글자" else QS[:, qi]
            idx = [i for i, t in enumerate(ts) if live[t]["room"] == sg]
            early = [i for i in idx if ts[i] <= t0] if mv else idx[:len(idx) // 2]
            late = [i for i in idx if ts[i] > t0] if mv else idx[len(idx) // 2:]
            drop = (float(np.quantile(TS[early], .9) - np.quantile(TS[late], .9))
                    if len(early) >= 3 and len(late) >= 3 else None)
            rows.append(dict(sg=sg, tgt=tgt, moved=bool(mv), drop=drop, typ=v["type"],
                             rids=rids, nrt=nrt, rt=rt,
                             revis=bool(mv) and len(late) >= 3))
    return rows


def prior_rank(r, exclude=None):
    """인구 사전확률로 방 순위. exclude 는 부재로 배제된 방."""
    p = PR.get(r["typ"], {})
    sc = {q: p.get(r["rt"][q], .25) / max(r["nrt"][r["rt"][q]], 1) for q in r["rids"]}
    if exclude:
        sc.pop(exclude, None)
    return max(sc, key=sc.get) if sc else r["sg"]


def dest_rank(r, exclude=None):
    """**이동 목적지** 사전확률. 물건이 옮겨졌다는 걸 알 때 쓰는 분포 —
    배치 사전확률과 다르다(머그컵은 부엌에 놓이지만 거실에서 발견된다)."""
    d = MV["dest"].get(r["typ"], {})
    sc = {q: d.get(r["rt"][q], .25) / max(r["nrt"][r["rt"][q]], 1) for q in r["rids"]}
    if exclude:
        sc.pop(exclude, None)
    return max(sc, key=sc.get) if sc else r["sg"]


def ci(a):
    a = np.asarray(a, float); n = len(a)
    if n == 0:
        return 0., 0., 0.
    b = [a[np.random.randint(0, n, n)].mean() for _ in range(2000)]
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


rows = load()
mvd = [r for r in rows if r["moved"]]
sta = [r for r in rows if not r["moved"]]
dr = [r["drop"] for r in rows if r["drop"] is not None]
print("=== 갱신의 기여 [%s 질의] ===" % MODE)
print("  타겟 %d (이동 %d · 제자리 %d) · 이동 후 원래 방 재방문 %d"
      % (len(rows), len(mvd), len(sta), sum(r["revis"] for r in mvd)))

print("\n── 전체 타겟 (지금까지 재던 방식) ──")
for nm, f in (("씬그래프 방 그대로 (갱신 없음)", lambda r: r["sg"]),
              ("인구 사전확률만 (관측 없음)", lambda r: prior_rank(r))):
    m, lo, hi = ci([f(r) == r["tgt"] for r in rows])
    print("  %-28s %.3f [%.3f %.3f]" % (nm, m, lo, hi))

print("\n── **이동한 물체만** — 갱신의 가치가 드러나는 구간 ──")
print("  %-28s %.3f   ← 정의상 0" % ("씬그래프 방 그대로",
      np.mean([r["sg"] == r["tgt"] for r in mvd])))
for nm, f in (("인구 사전확률", lambda r: prior_rank(r, r["sg"])),
              ("**이동목적지 사전확률**", lambda r: dest_rank(r, r["sg"]))):
    m, lo, hi = ci([f(r) == r["tgt"] for r in mvd])
    print("  %-28s %.3f [%.3f %.3f]" % (nm, m, lo, hi))

print("\n── 부재 문턱 스윕: 이동 구간 이득 vs 제자리 구간 손실 ──")
print("  %-7s %-9s %-9s %-9s %-9s" % ("문턱", "이동 정답", "제자리 정답", "전체", "부재발동"))
for q in (1.0, .95, .9, .85, .8, .7, .5):
    th = np.quantile(dr, q) if dr else 1e9
    def ans(r):
        ab = r["drop"] is not None and r["drop"] >= th
        return (dest_rank(r, r["sg"]) if ab else r["sg"]), ab
    am = [ans(r) for r in mvd]; asx = [ans(r) for r in sta]
    fire = sum(a[1] for a in am + asx)
    print("  %-7.2f %-9.3f %-9.3f %-9.3f %d"
          % (q, np.mean([a[0] == r["tgt"] for a, r in zip(am, mvd)]) if mvd else 0,
             np.mean([a[0] == r["tgt"] for a, r in zip(asx, sta)]) if sta else 0,
             np.mean([a[0] == r["tgt"] for a, r in zip(am + asx, mvd + sta)]), fire))
