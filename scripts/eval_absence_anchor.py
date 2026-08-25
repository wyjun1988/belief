#!/usr/bin/env python3
"""**앵커로 게이팅한 부재 검출.**

    $P scripts/eval_absence_anchor.py [글자|이미지]

⚠️ **왜 확장질의−키워드를 따로 안 만드나.** "책상 위 노트북" 에서 노트북을 뺀
"책상 위" 가 곧 정적 문맥이고, 그게 앵커다. 앵커 점수를 이미 프레임마다 계산하므로
키워드 제거 기제는 **중복**이다.

⚠️ **앵커가 더하는 것은 조건③ 해소다.** 부재 증거는 씬 단위(SceneDiff 0.714)·
방 단위(SuperMemory 2.4~2.5)에서는 잡혔지만 **자리 단위(~1m)에서는 우연**이었다.
물건이 자리를 떠나는 것과 시야에서 벗어나는 것이 겹쳐, "안 보인다" 가 "없다" 를
뜻하지 않았기 때문이다. 앵커가 보이는 프레임으로 한정하면 **"그 자리가 화면에
있는데 물건이 없다"** 가 되어 둘이 분리된다.

기제:
  ① 물체가 있던 시절, 그 물체가 잘 보인 프레임들의 **정적 타입 점수 분포** = 자리 서명
  ② 이후 프레임마다 자리 서명과의 일치도 = "그 자리가 화면에 있나"
  ③ 자리가 보이는 프레임에서만 물체 점수를 본다 → 낮으면 부재
"""
import json, glob, os, sys, numpy as np
from collections import Counter

MODE = sys.argv[1] if len(sys.argv) > 1 else "이미지"
ROOT = os.environ.get("THOR_ROOT", "data/thor3")


def ci(a):
    a = np.asarray(a, float); n = len(a)
    if n == 0: return 0., 0., 0.
    b = [a[np.random.randint(0, n, n)].mean() for _ in range(2000)]
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


rows = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = "/tmp/a3_%s.npz" % hn, "/tmp/qc_%s.npz" % hn
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, ts, vocab, nT = za["s"], za["ts"], list(za["vocab"]), int(za["nT"])
    QT, QS = list(zq["tg"]), zq["si"]
    ST = S[:, nT:]                                     # 정적 타입 점수 (앵커)
    g = json.load(open(hd + "/gt.json"))
    live = {m["t"]: m for m in g["live"]}
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for oid, v in g["gt0"].items():
        if not v["room"] or cnt[v["type"]] > 1 or oid not in QT: continue
        if v["type"] not in vocab: continue
        ti = vocab.index(v["type"]); qi = QT.index(oid)
        TS = S[:, ti] if MODE == "글자" else QS[:, qi]
        mv = [x for x in moves if x["oid"] == oid]
        t0 = mv[-1]["t"] if mv else 0
        inr = [i for i, t in enumerate(ts) if live[t]["room"] == v["room"]]
        early = [i for i in inr if ts[i] <= t0] if mv else inr[:len(inr)//2]
        late = [i for i in inr if ts[i] > t0] if mv else inr[len(inr)//2:]
        if len(early) < 5 or len(late) < 5: continue
        # ① 자리 서명 — 물체가 잘 보인 프레임 상위 30% 의 정적 점수 평균
        k = max(3, len(early)//3)
        sig = ST[[early[j] for j in np.argsort(-TS[early])[:k]]].mean(0)
        sig = sig / (np.linalg.norm(sig) + 1e-9)
        # ② 자리 일치도
        def place(idx):
            A = ST[idx]; A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            return A @ sig
        pe, pl = place(early), place(late)
        # ③ **자리가 보이는 프레임에서만** 물체 점수를 비교
        thp = np.quantile(pe, .7)                      # 자리가 확실히 보이는 수준
        ge = [i for i, p in zip(early, pe) if p >= thp]
        gl = [i for i, p in zip(late, pl) if p >= thp]
        if len(ge) < 3 or len(gl) < 3: continue
        rows.append(dict(
            moved=bool(mv), typ=v["type"],
            gated=float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9)),   # 앵커 게이팅
            plain=float(np.quantile(TS[early], .9) - np.quantile(TS[late], .9)),  # 게이팅 없음
            nvis=len(gl), frac=len(gl)/len(late)))

mvd = [r for r in rows if r["moved"]]; sta = [r for r in rows if not r["moved"]]
print("=== 앵커 게이팅 부재 검출 [%s 질의] ===" % MODE)
print("  타겟 %d (이동 %d · 제자리 %d)" % (len(rows), len(mvd), len(sta)))
print("  자리가 보이는 프레임 비율 중앙 %.3f" % np.median([r["frac"] for r in rows]))
print("\n  %-14s %-22s %-22s %s" % ("신호", "이동 하락", "제자리 하락", "부재 AUC"))
for key, nm in (("plain", "게이팅 없음"), ("gated", "**앵커 게이팅**")):
    a = np.array([r[key] for r in mvd]); b = np.array([r[key] for r in sta])
    if not len(a) or not len(b): continue
    auc = (a[:, None] > b[None, :]).mean() + .5*(a[:, None] == b[None, :]).mean()
    bs = [(lambda x, y: (x[:, None] > y[None, :]).mean())(
        a[np.random.randint(0, len(a), len(a))], b[np.random.randint(0, len(b), len(b))])
        for _ in range(2000)]
    m1, l1, h1 = ci(a); m2, l2, h2 = ci(b)
    print("  %-14s %+.4f [%+.4f %+.4f]  %+.4f [%+.4f %+.4f]  **%.3f** [%.3f %.3f]"
          % (nm, m1, l1, h1, m2, l2, h2, auc, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
