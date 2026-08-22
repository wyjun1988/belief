#!/usr/bin/env python3
"""**2차 엔드투엔드** — 초기 맵 1회 + 이후 1fps RGB 만. pose·depth 없음.

    $P scripts/thor2_e2e.py --root data/thor2 --cache … --stride 3

### 실사용 조건

    t=0    **초기 맵** — 전 방을 돌며 방 키(CLIP latent 평균)를 만든다. 설치 시 1회.
    t>0    1fps 배회. **RGB 만.** 방은 키와의 유사도로 재식별한다.
    질의   여러 시각에 "X 어디 있어?"

### 1차와 달라진 점

  · 방 키를 **초기 맵에서** 만든다 (1차는 전체를 한꺼번에 군집했다 — 정확도 0.366)
  · **시간 연속성**을 쓴다 — 연속 프레임은 같은 방일 확률이 높다(중앙값 평활)
  · **저장 간격**을 실험한다 — 얼마나 성글게 저장해도 답이 유지되는가

### GT

물체 o, 질의 시각 T:
  · T 이전 마지막 이동이 있으면 그 시각 t_m 이후 **원래 방**을 다시 봤는가로 (b)/(c) 판정
  · 이동이 없으면 그 방을 봤는가로 (a)/(b)
"""
import argparse, glob, json, os
from collections import Counter, defaultdict

import numpy as np


def smooth(lab, w):
    """시간 연속성 — 창 안 최빈값으로 평활. 실제로 순간이동하지 않으므로 정당하다."""
    if w <= 1:
        return lab
    out = lab.copy()
    for i in range(len(lab)):
        s = max(0, i - w // 2); e = min(len(lab), i + w // 2 + 1)
        out[i] = Counter(lab[s:e]).most_common(1)[0][0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stride", type=int, default=1, help="캐시 위에 더 성글게(1=캐시 그대로)")
    ap.add_argument("--smooth", type=int, default=5, help="시간 평활 창(프레임)")
    ap.add_argument("--wq", type=float, default=0.90)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--queries", type=int, default=4, help="주택당 질의 시각 수")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    houses = sorted(glob.glob(os.path.join(args.root, "house_*")))
    prior = defaultdict(Counter); seen_in = defaultdict(set)
    gts = {}
    for hd in houses:
        g = json.load(open(os.path.join(hd, "gt.json")))
        gts[hd] = g
        rt = g["room_types"]
        for oid, v in g["gt0"].items():
            if v["room"]:
                prior[v["type"]][rt[v["room"]]] += 1
                seen_in[(v["type"], rt[v["room"]])].add(hd)

    def belief(ot, ex):
        c = Counter()
        for a, n in prior.get(ot, {}).items():
            m = n - (1 if ex in seen_in.get((ot, a), ()) else 0)
            if m > 0:
                c[a] = m
        return [a for a, _ in c.most_common(args.topk)]

    rows = []; racc = []
    for hd in houses:
        cf = os.path.join(args.cache, os.path.basename(hd) + ".npz")
        if not os.path.exists(cf):
            continue
        z = np.load(cf, allow_pickle=True)
        em, el, ol, ts = z["em"], z["el"], z["ol"], z["ts"]
        vocab = list(z["vocab"]); vi = {w: i for i, w in enumerate(vocab)}
        g = gts[hd]; rt = g["room_types"]
        mrooms = [m["room"] for m in g["map"]]
        if len(mrooms) != len(em):
            continue
        # ── 방 키: 초기 맵에서 (설치 시 1회)
        keys, kn = [], []
        for r in sorted(set(mrooms)):
            ix = [i for i, x in enumerate(mrooms) if x == r]
            v = em[ix].mean(0); keys.append(v / (np.linalg.norm(v) + 1e-9)); kn.append(r)
        K = np.stack(keys)
        # ── 배회 프레임을 RGB 만으로 방에 배정 + 시간 평활
        sel = np.arange(0, len(ts), args.stride)
        lab = np.array([kn[int(np.argmax(K @ el[i]))] for i in sel])
        lab = smooth(lab, args.smooth)
        tt = ts[sel]; O = ol[sel]
        gtl = {m["t"]: m["room"] for m in g["live"]}
        gtr = np.array([gtl.get(int(t), None) for t in tt])
        ok = gtr != None
        if ok.sum():
            racc.append(float(np.mean(lab[ok] == gtr[ok])))
        # ── 질의 시각
        Q = [int(len(tt) * f) for f in np.linspace(0.4, 0.98, args.queries)]
        moves = sorted(g["moves"], key=lambda m: m["t"])
        for qi in Q:
            T = int(tt[qi])
            for oid, v0 in g["gt0"].items():
                ot = v0["type"]
                if ot not in vi or not v0["room"]:
                    continue
                j = vi[ot]
                mv = [m for m in moves if m["oid"] == oid and m["t"] <= T]
                r_true = mv[-1]["to"] if mv else v0["room"]
                r_old = mv[-1]["frm"] if mv else v0["room"]
                t_m = mv[-1]["t"] if mv else -1
                # GT 상태 — 이동 후 원래 방을 다시 봤는가
                after = [i for i in range(qi + 1) if tt[i] > t_m]
                revis = any(gtr[i] == r_old for i in after)
                gt_state = "b" if not revis else ("c" if mv else "a")
                # ── 시스템
                upto = np.arange(qi + 1)
                top = upto[np.argsort(-O[upto, j])[:3]]
                r_pred = Counter(lab[top]).most_common(1)[0][0]
                t_seen = int(tt[top].max())
                inr_b = [i for i in upto if lab[i] == r_pred and tt[i] <= t_seen]
                inr_a = [i for i in upto if lab[i] == r_pred and tt[i] > t_seen]
                sb = float(np.quantile(O[inr_b, j], args.wq)) if len(inr_b) else 0.0
                if len(inr_a) < 2:
                    state = "b"; sa = float("nan")
                else:
                    sa = float(np.quantile(O[inr_a, j], args.wq))
                    state = "c" if sa < args.ratio * sb else "a"
                rows.append(dict(house=os.path.basename(hd), T=T, oid=oid, otype=ot,
                                 r_gt=rt.get(r_old), r_pred=rt.get(r_pred, r_pred),
                                 r_now=rt.get(r_true), moved=bool(mv),
                                 state=state, gt_state=gt_state,
                                 belief=belief(ot, hd) if state == "c" else []))

    if not rows:
        print("표본 없음"); return
    n = len(rows)
    print("주택 %d · 질의 %d · stride %d · 평활 %d"
          % (len({r["house"] for r in rows}), n, args.stride, args.smooth))
    if racc:
        print("① 방 재식별 %.3f (초기 맵 키 + 시간 평활)" % float(np.mean(racc)))
    print("  예측 %s · GT %s" % (dict(Counter(r["state"] for r in rows)),
                                dict(Counter(r["gt_state"] for r in rows))))
    ok = sum(1 for r in rows if r["state"] == r["gt_state"])
    base = max(Counter(r["gt_state"] for r in rows).values()) / n
    print("② 상태 3분류 %.3f — 다수결 %.3f" % (ok / n, base))
    ab = [r for r in rows if r["state"] in ("a", "b")]
    if ab:
        h = sum(1 for r in ab if r["r_pred"] == r["r_gt"])
        mb = Counter(r["r_gt"] for r in ab).most_common(1)[0][1] / len(ab)
        print("③ 위치 답 %d건 · 방 정답 %.3f — 최빈 %.3f" % (len(ab), h / len(ab), mb))
    cc = [r for r in rows if r["state"] == "c" and r["r_now"]]
    if cc:
        t1 = sum(1 for r in cc if r["belief"][:1] == [r["r_now"]])
        tk = sum(1 for r in cc if r["r_now"] in r["belief"])
        mb = Counter(r["r_now"] for r in cc).most_common(1)[0][1] / len(cc)
        print("④ belief %d건 · top-1 %.3f · top-%d %.3f — 최빈 %.3f"
              % (len(cc), t1 / len(cc), args.topk, tk / len(cc), mb))
    full = sum(1 for r in rows
               if (r["gt_state"] in ("a", "b") and r["state"] == r["gt_state"]
                   and r["r_pred"] == r["r_gt"])
               or (r["gt_state"] == "c" and r["state"] == "c" and r["r_now"] in r["belief"]))
    print("⑤ **전체 정답 %.3f (%d/%d)**" % (full / n, full, n))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
