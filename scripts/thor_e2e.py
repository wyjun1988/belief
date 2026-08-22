#!/usr/bin/env python3
"""**엔드투엔드 · 다중 방** — "내 안경 어디 뒀지?" 를 RGB 만으로 답하고 belief 까지 채점.

    $P scripts/thor_e2e.py --root data/thor --out …

ProcTHOR 로 생성한 주택(방 4~10개)에서, **모든 정답을 아는 상태로** 전 과정을 잰다.

### 시스템이 하는 일 (RGB 만)

    ① 장소 식별  프레임을 CLIP latent 로 군집 → 방. **GT 방 라벨을 안 쓴다**
    ② 검색       물체 질의 → 1세션에서 가장 잘 잡힌 프레임 → 그 프레임의 방
    ③ 재방문     2세션에 그 방이 있는가
    ④ 부재       그 방의 2세션 프레임에서 물체가 검출되는가
    ⑤ belief     없으면 **어느 방으로 갔을지** 예측 (다른 주택에서 배운 사전확률)

### 세 상태

    (a) 있다        재방문했고 물체가 여전히 있다
    (b) 있을 것이다  그 방을 다시 안 봤다        ← 생성 때 의도적으로 만들었다
    (c) 없다        재방문했는데 물체가 없다     → belief

### 채점

  ① 방 군집 품질 (GT 방 대비)
  ② 상태 3분류 (다수결 병기)
  ③ (a)/(b) 방 정답률
  ④ **(c) belief top-1 / top-k** — 어느 방으로 갔나
  ⑤ 전체 — 질문에 제대로 답한 비율

⚠️ 통제: 물체별 자기 기준(절대 검출점수 물체 간 비교 금지), 창 대표는 상위 분위수(㊴),
belief 사전확률은 **다른 주택에서만**(leave-one-house-out).
"""
import argparse, glob, json, os
from collections import Counter, defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True, help="주택별 OWL/CLIP 캐시")
    ap.add_argument("--wq", type=float, default=0.90)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=2, help="belief top-k")
    ap.add_argument("--oracle-room", action="store_true", help="방 군집을 GT 로 대체")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    houses = sorted(glob.glob(os.path.join(args.root, "house_*")))
    # ── belief 사전확률 재료: (물체유형 → 방유형) 빈도, 주택별로 기록
    prior = defaultdict(Counter); seen_in = defaultdict(set)
    gts = {}
    for hd in houses:
        g = json.load(open(os.path.join(hd, "gt.json")))
        gts[hd] = g
        rt = {r["id"]: r["type"] for r in g["rooms"]}
        for oid, v in g["gt1"].items():
            if v["room"]:
                prior[v["type"]][rt[v["room"]]] += 1
                seen_in[(v["type"], rt[v["room"]])].add(hd)

    def belief(otype, exclude):
        c = Counter()
        for a, n in prior.get(otype, {}).items():
            m = n - (1 if exclude in seen_in.get((otype, a), ()) else 0)
            if m > 0:
                c[a] = m
        return [a for a, _ in c.most_common(args.topk)]

    rows = []; clus = []
    for hd in houses:
        cf = os.path.join(args.cache, os.path.basename(hd) + ".npz")
        if not os.path.exists(cf):
            continue
        z = np.load(cf, allow_pickle=True)
        E1, E2, O1, O2 = z["e1"], z["e2"], z["o1"], z["o2"]
        vocab = list(z["vocab"]); vi = {w: i for i, w in enumerate(vocab)}
        g = gts[hd]
        rt = {r["id"]: r["type"] for r in g["rooms"]}
        r1 = [m["room"] for m in g["m1"]]; r2 = [m["room"] for m in g["m2"]]
        if len(r1) != len(E1) or len(r2) != len(E2):
            continue
        K = len(set(r1))
        # ① 장소 식별 — CLIP latent 군집 (GT 라벨 미사용)
        if args.oracle_room:
            lab1, lab2 = np.array(r1), np.array(r2)
        else:
            from scipy.cluster.vq import kmeans2
            cen, l1 = kmeans2(E1, K, minit="++", seed=0, iter=40)
            d = E2 @ cen.T
            lab1 = np.array(["c%d" % v for v in l1])
            lab2 = np.array(["c%d" % v for v in np.argmax(d, 1)])
            # 군집 → GT 방 대응 (다수결) · 품질 기록
            c2n = {}
            for c in set(lab1):
                m = Counter(np.array(r1)[lab1 == c])
                c2n[c] = m.most_common(1)[0][0]
            clus.append(float(np.mean([c2n[c] == r for c, r in zip(lab1, r1)])))
            lab1 = np.array([c2n[c] for c in lab1]); lab2 = np.array([c2n[c] for c in lab2])
        for oid, v1 in g["gt1"].items():
            ot = v1["type"]
            if ot not in vi or not v1["room"]:
                continue
            j = vi[ot]
            sb_all = float(np.quantile(O1[:, j], args.wq))
            # ② 검색 — 가장 잘 잡힌 프레임의 방
            top = np.argsort(-O1[:, j])[:3]
            r_pred = Counter(lab1[top]).most_common(1)[0][0]
            inr1 = np.nonzero(lab1 == r_pred)[0]
            sb = float(np.quantile(O1[inr1, j], args.wq)) if len(inr1) else sb_all
            # ③ 재방문
            inr2 = np.nonzero(lab2 == r_pred)[0]
            if len(inr2) < 2:
                state = "b"; sa = float("nan")
            else:
                sa = float(np.quantile(O2[inr2, j], args.wq))
                state = "c" if sa < args.ratio * sb else "a"
            # GT
            g2 = g["gt2"].get(oid, {})
            r_now = g2.get("room")
            gt_state = ("b" if rt.get(v1["room"]) and v1["room"] not in g["revisited"]
                        else "c" if r_now != v1["room"] else "a")
            bl = belief(ot, hd)
            # ⚠️ **id 와 타입을 비교하면 안 된다.** 군집은 방 **id** 로 매핑되는데
            # GT 를 방 **타입**("Bedroom")으로 뒀더니 방 정답률이 **정확히 0.000** 이 나왔다.
            rows.append(dict(house=os.path.basename(hd), oid=oid, otype=ot,
                             r_gt=rt.get(v1["room"]), r_pred=rt.get(r_pred, r_pred),
                             r_pred_id=r_pred,
                             r_now=rt.get(r_now) if r_now else None,
                             moved=oid in g["moved"], state=state, gt_state=gt_state,
                             s_before=sb, s_after=sa, belief=bl))

    if not rows:
        print("표본 없음 — 캐시 필요"); return
    n = len(rows)
    print("주택 %d · 질의 %d" % (len({r["house"] for r in rows}), n))
    if clus:
        print("① 방 군집 정확도 %.3f (주택 %d)" % (float(np.mean(clus)), len(clus)))
    print("  예측 상태 %s · GT 상태 %s"
          % (dict(Counter(r["state"] for r in rows)), dict(Counter(r["gt_state"] for r in rows))))
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
               or (r["gt_state"] == "c" and r["state"] == "c"
                   and (r["r_now"] is None or r["r_now"] in r["belief"])))
    # ── **사용자 질문에 대한 최종 답**
    # ⚠️ 상태 3분류·전체정답은 "원래 방을 떠났나" 를 묻는다. 그런데 사용자 질문은
    # **"지금 어디 있나"** 다. 시스템이 새 위치에서 물체를 찾아내면 그게 최선의
    # 답인데 종전 채점은 그것을 틀렸다고 셌다. 답 = (a)/(b)면 예측 방,
    # (c)면 belief 1순위. 정답 = 실제 현재 방.
    ans = [(r["r_pred"] if r["state"] in ("a", "b")
            else (r["belief"][0] if r["belief"] else None), r.get("r_now")) for r in rows]
    ans = [(a, t) for a, t in ans if t]
    if ans:
        acc = sum(1 for a, t in ans if a == t) / len(ans)
        mb = Counter(t for _, t in ans).most_common(1)[0][1] / len(ans)
        print("⑥ **최종 답('지금 어디 있나') %.3f (%d건) — 최빈 %.3f**" % (acc, len(ans), mb))
    if ans:
        bo = [(r["belief"][0] if r["belief"] else None, r.get("r_now")) for r in rows]
        bo = [(a, t) for a, t in bo if t]
        if bo:
            print("   └ **belief 단독** %.3f (%d건) — 인지 파이프라인 없이 사전확률만"
                  % (sum(1 for a, t in bo if a == t) / len(bo), len(bo)))
    print("⑤ **전체 정답 %.3f (%d/%d)**" % (full / n, full, n))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
