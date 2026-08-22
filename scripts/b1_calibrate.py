#!/usr/bin/env python3
"""**B1 — 판정 여유를 확률로 보정해 내보낸다.**

    $P scripts/b1_calibrate.py --in e2e.json

⚠️ **왜 여유(margin)인가.** A5(증거 누적)와 A5'(시점 짝짓기)를 다 해봤지만 둘 다
확률 보정이 안 섰다 — 어떤 확률을 내놓든 실제 부재율이 기저율(0.4) 근처로 평평했다.
원인은 우도비가 `log(p_fire/(1−q_miss))` 인데 목격 창이 오염되면 `q_miss ≈ 1−p_fire`
가 되어 **증거가 0** 이 되는 것이었다.

반면 **판정 여유**(결정 문턱까지의 거리)는 기권 없이 전수 판정했을 때 오류율을
단조롭게 정렬했다 — 커버리지 20% 에서 0.279, 100% 에서 0.445.
쓸 만한 불확실성은 새로 만들 필요가 없었고 **이미 판정 통계 안에 있었다.**

여기서는 그 여유를 확률로 바꾼다. ⚠️ **보정은 반드시 held-out 으로 잰다** —
같은 자료에 맞추고 그 자료에서 보정도를 보고하면 항상 좋아 보인다.
질의 시점 T 를 하나씩 빼는 leave-one-out 으로 한다(시점이 5개).
"""
import argparse, json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()
    rows = [x for x in json.load(open(args.inp))
            if x.get("margin") == x.get("margin") and x["state"] in ("a", "c")]
    X = np.array([x["margin"] for x in rows])
    Y = np.array([not x["truly_here"] for x in rows]).astype(float)
    T = np.array([x["T"] for x in rows])
    ts = sorted(set(T.tolist()))
    print("질의 %d · 시점 %d · 실제 부재율 %.3f" % (len(rows), len(ts), Y.mean()))

    def fit(x, y):
        """로지스틱 1변수 — 뉴턴법 몇 번이면 충분하다."""
        w = np.zeros(2)
        Xd = np.stack([np.ones_like(x), x], 1)
        for _ in range(50):
            p = 1 / (1 + np.exp(-Xd @ w))
            g = Xd.T @ (y - p)
            W = p * (1 - p) + 1e-9
            H = Xd.T @ (Xd * W[:, None]) + 1e-6 * np.eye(2)
            w += np.linalg.solve(H, g)
        return w

    P = np.zeros(len(rows))
    for t in ts:                                  # leave-one-T-out
        tr, te = T != t, T == t
        if tr.sum() < 10 or te.sum() == 0:
            P[te] = Y[tr].mean() if tr.sum() else 0.5
            continue
        w = fit(X[tr], Y[tr])
        P[te] = 1 / (1 + np.exp(-(w[0] + w[1] * X[te])))

    print("\n보정 (held-out)  %-12s %6s %10s" % ("확률 구간", "건수", "실제 부재율"))
    ece = 0.0
    for i in range(args.bins):
        lo, hi = i / args.bins, (i + 1) / args.bins
        m = (P >= lo) & (P < hi if i < args.bins - 1 else P <= 1.0)
        if m.sum():
            print("                 %.1f~%.1f %8d %10.3f" % (lo, hi, m.sum(), Y[m].mean()))
            ece += m.mean() * abs(Y[m].mean() - P[m].mean())
    print("  **ECE %.3f** (0에 가까울수록 확률이 정직하다)" % ece)
    base = np.full(len(P), Y.mean())
    print("  Brier  보정판 %.3f · 기저율만 %.3f" % (np.mean((P - Y) ** 2), np.mean((base - Y) ** 2)))

    conf = np.abs(P - 0.5); o = np.argsort(-conf); pred = P >= 0.5
    print("\n위험-커버리지 (확률로 확신 정렬)")
    print("   %-10s %8s %8s %8s" % ("커버리지", "오류율", "재현", "오경보"))
    for c in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = max(2, int(c * len(P))); idx = o[:k]
        g = Y[idx] > 0.5; p = pred[idx]
        print("   %9.0f%% %8.3f %8.3f %8.3f"
              % (100 * c, float((p != g).mean()),
                 float(p[g].mean()) if g.sum() else float("nan"),
                 float(p[~g].mean()) if (~g).sum() else float("nan")))


if __name__ == "__main__":
    main()
