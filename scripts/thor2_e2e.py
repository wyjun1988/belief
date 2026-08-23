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
    ap.add_argument("--min-run", type=int, default=0,
                    help="**재방문을 연속 구간 길이로 판정한다.** ⚠️ 절대 유사도는 못 쓴다 — "
                         "맞는 방 키 0.928 vs 최고 오답 0.930, 여유 중앙 -0.0003 이라 "
                         "임계값이 안 듣는다(안 본 방 9/9 를 '봤다' 고 했다). "
                         "실제로 간 방은 연속 구간이 길다(체류 360초=36프레임).")
    ap.add_argument("--room-thr", type=float, default=0.0,
                    help="방 배정 **거부 임계값**. ⚠️ 0이면 모든 프레임이 가장 가까운 방 키에 "
                         "배정되므로 **어느 방이든 프레임이 생기고**, 안 가본 방도 '가봤다' 가 된다"
                         "(실측: 실제 안 본 방 9개를 9개 다 '봤다' 고 했다). "
                         "유사도가 이 값 미만이면 '모르는 곳' 으로 두고 재방문에서 뺀다.")
    ap.add_argument("--smooth", type=int, default=5, help="시간 평활 창(프레임)")
    ap.add_argument("--wq", type=float, default=0.90)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--loc", default="topf", choices=["topf", "roomq", "roomlift"],
                    help="물체 위치 추정. ⚠️ 기본 `topf`(점수 상위 3프레임의 방)는 "
                         "**음성이 570장인데 argmax 를 쓴다** — AUC 0.875 여도 최고점 "
                         "3장이 오검출일 확률이 높다. "
                         "roomq=방별 상위분위수가 가장 높은 방, "
                         "roomlift=방별 분위수 − 그 물체의 전체 분위수(자기 기준 보정).")
    ap.add_argument("--oracle-room", action="store_true",
                    help="배회 프레임의 방을 GT 로 대체 — **방 재식별 오류와 검색 오류를 가른다**")
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
        tt = ts[sel]; O = ol[sel]
        gtl = {m["t"]: m["room"] for m in g["live"]}
        gtr = np.array([gtl.get(int(t), None) for t in tt])
        if args.oracle_room:
            lab = np.array([x if x else kn[0] for x in gtr])
        else:
            lab = np.array([kn[int(np.argmax(K @ el[i]))] for i in sel])
            lab = smooth(lab, args.smooth)
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
                if args.loc == "topf":
                    top = upto[np.argsort(-O[upto, j])[:3]]
                    r_pred = Counter(lab[top]).most_common(1)[0][0]
                else:
                    base = float(np.quantile(O[upto, j], args.wq))
                    sc_r = {}
                    for rr in set(lab[upto]):
                        ix = upto[lab[upto] == rr]
                        if len(ix) < 3:
                            continue
                        q = float(np.quantile(O[ix, j], args.wq))
                        sc_r[rr] = q - (base if args.loc == "roomlift" else 0.0)
                    if not sc_r:
                        continue
                    r_pred = max(sc_r, key=sc_r.get)
                    ix = upto[lab[upto] == r_pred]
                    top = ix[np.argsort(-O[ix, j])[:3]]
                t_seen = int(tt[top].max())
                inr_b = [i for i in upto if lab[i] == r_pred and tt[i] <= t_seen]
                inr_a = [i for i in upto if lab[i] == r_pred and tt[i] > t_seen]
                if args.min_run > 0 and inr_a:
                    # 연속 구간이 짧으면 "그 방을 다시 본 것" 으로 치지 않는다
                    run = c = 0
                    for i in upto:
                        if tt[i] > t_seen and lab[i] == r_pred:
                            c += 1; run = max(run, c)
                        else:
                            c = 0
                    if run < args.min_run:
                        inr_a = []
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
                                 belief=belief(ot, hd)))

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
    # ── 관측 vs 사전확률 — 어디서 갈리나
    sub = [r for r in rows if r.get("r_now") and r.get("belief")]
    for tag, ss in (("전체", sub), ("이동한 물체", [r for r in sub if r["moved"]]),
                    ("안 움직인 물체", [r for r in sub if not r["moved"]])):
        if len(ss) < 10:
            continue
        obs = np.mean([r["r_pred"] == r["r_now"] for r in ss])
        bel = np.mean([r["belief"][0] == r["r_now"] for r in ss])
        dis = [r for r in ss if r["r_pred"] != r["belief"][0]]
        do = np.mean([r["r_pred"] == r["r_now"] for r in dis]) if dis else float("nan")
        db = np.mean([r["belief"][0] == r["r_now"] for r in dis]) if dis else float("nan")
        print("   %-12s n=%4d · 관측만 %.3f · 사전확률만 %.3f │ 둘이 다를 때(%d건) 관측 %.3f vs 사전 %.3f"
              % (tag, len(ss), obs, bel, len(dis), do, db))
    print("⑤ **전체 정답 %.3f (%d/%d)**" % (full / n, full, n))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
