#!/usr/bin/env python3
"""**능력별 분해** — 지각 · 검색 · 재방문판정 · 부재를 따로 잰다.

    $P scripts/thor2_abilities.py --root data/thor2r --cache /tmp/thor2rcache

엔드투엔드 점수(0.403)가 어느 층에서 깎이는지 층별로 본다.

    ① 지각    물체가 **같은 방에 있을 때** 검출 점수가 높은가 (프레임 단위 AUC)
    ② 검색    그 신호로 **어느 방에 있나** 를 맞히는가
    ③ 재방문  그 방을 다시 봤는지 맞히는가  ← (b) 판정. **부재와 다른 능력이다**
    ④ 부재    물체가 방을 떠났을 때 그 방에서 점수가 떨어지는가

⚠️ (b) 는 부재 증거가 아니다. "그 방을 다시 안 봐서 확인 못 함" 이다.
   "없어진 것만 보고 어디 갔는지는 못 봄" 은 (c)+belief 다.
"""
import argparse, glob, json, os
from collections import Counter, defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--wq", type=float, default=0.90)
    ap.add_argument("--ctx-k", type=int, default=4, help="정적 동반 물체 수")
    ap.add_argument("--ctx-q", type=float, default=0.70,
                    help="그 정적 물체가 '보인다' 로 볼 분위수(초기 맵 기준)")
    ap.add_argument("--abs-room", default="gt", choices=["gt", "pred", "ctx"],
                    help="**부재를 잴 때 프레임을 무엇으로 고르나.** "
                         "`ctx` = **그 물체의 정적 동반 물체가 보이는 프레임**(키워드 제외). "
                         "방 라벨을 안 쓰므로 방 식별 오류의 영향을 받지 않는다 — "
                         "54 에서 드러난 '부재가 방 인지에 의존해 0.726→0.53' 문제를 우회한다. "
                         "정적 물체만 쓰는 이유: 움직이는 물체를 앵커로 쓰면 그것도 옮겨진다. "
                         "종전 측정(AUC 0.726)은 "
                         "`gt` 였다 — 오라클 방에서 잰 값이므로 '부재는 방 인지의 영향을 "
                         "덜 받는다' 는 주장의 근거가 되지 못한다. `pred` 로 다시 잰다.")
    ap.add_argument("--place-node", action="store_true",
                    help="**방 키를 평균 벡터 하나가 아니라 여러 노드로.** ⚠️ 방마다 맵 프레임 "
                         "12~120장을 한 벡터로 평균하면 여러 방향에서 본 모습이 뭉개진다 — "
                         "맞는 방 0.928 vs 최고 오답 0.930(여유 0.000). 노드별 최대유사도를 "
                         "방 점수로 쓰면 다봉 분포가 살아난다. 실측 방 식별 0.825 → 0.886.")
    ap.add_argument("--stay", type=float, default=0.0, help="전이 추적 머무를 확률")
    ap.add_argument("--min-run", type=int, default=0,
                    help="**연속 구간 길이로 재방문을 판정한다.** ⚠️ 절대 유사도는 못 쓴다 — "
                         "실측: 맞는 방 키 0.928 vs 최고 오답 0.930, 여유 중앙 -0.0003. "
                         "대신 실제로 간 방은 **연속 구간이 길고**(체류 360초=36프레임) "
                         "안 간 방은 흩어진 잡음이다. 이 길이 이상 연속이어야 '갔다' 로 본다.")
    ap.add_argument("--room-thr", type=float, default=0.0,
                    help="방 배정 **거부 임계값**. ⚠️ 0이면 모든 프레임이 가장 가까운 방 키에 "
                         "배정되므로 **어느 방이든 프레임이 생기고**, 안 가본 방도 '가봤다' 가 된다"
                         "(실측: 실제 안 본 방 9개를 9개 다 '봤다' 고 했다). "
                         "유사도가 이 값 미만이면 '모르는 곳' 으로 두고 재방문에서 뺀다.")
    ap.add_argument("--smooth", type=int, default=5)
    args = ap.parse_args()
    from scipy.stats import mannwhitneyu

    per, ret, rev, absn = [], [], [], []
    for hd in sorted(glob.glob(os.path.join(args.root, "house_*"))):
        cf = os.path.join(args.cache, os.path.basename(hd) + ".npz")
        if not os.path.exists(cf):
            continue
        z = np.load(cf, allow_pickle=True)
        em, el, ol, ts = z["em"], z["el"], z["ol"], z["ts"]
        vocab = list(z["vocab"]); vi = {w: i for i, w in enumerate(vocab)}
        g = json.load(open(os.path.join(hd, "gt.json")))
        rt = g["room_types"]
        mrooms = [m["room"] for m in g["map"]]
        if len(mrooms) != len(em):
            continue
        keys, kn = [], []
        for r in sorted(set(mrooms)):
            ix = [i for i, x in enumerate(mrooms) if x == r]
            v = em[ix].mean(0); keys.append(v / (np.linalg.norm(v) + 1e-9)); kn.append(r)
        K = np.stack(keys)
        gtl = {m["t"]: m["room"] for m in g["live"]}
        gtr = np.array([gtl.get(int(t), None) for t in ts])
        # 예측 방 (RGB 만 + 시간 평활)
        if args.place_node:
            mrr = np.array(mrooms, object) if "mrooms" in dir() else np.array(mr, object)
            Sn = el @ em.T
            sim = np.stack([Sn[:, mrr == r].max(1) for r in kn], 1)
        else:
            sim = el @ K.T
        if args.stay > 0:
            Z = (sim - sim.max(1, keepdims=True)) / 0.01
            logem = Z - np.log(np.exp(Z).sum(1, keepdims=True) + 1e-12)
            T_, K_ = logem.shape
            tr = np.full((K_, K_), np.log((1 - args.stay) / max(K_ - 1, 1)))
            np.fill_diagonal(tr, np.log(args.stay))
            dp = logem[0].copy(); bp = np.zeros((T_, K_), int)
            for t in range(1, T_):
                m = dp[:, None] + tr
                bp[t] = np.argmax(m, 0); dp = m.max(0) + logem[t]
            path = np.zeros(T_, int); path[-1] = int(np.argmax(dp))
            for t in range(T_ - 1, 0, -1):
                path[t - 1] = bp[t, path[t]]
            lab = np.array(kn, object)[path]
        else:
            best = np.argmax(sim, 1)
            lab = np.array([kn[b] if sim[i, b] >= args.room_thr else None
                            for i, b in enumerate(best)], object)
        if args.smooth > 1:
            sm = lab.copy()
            for i in range(len(lab)):
                s = max(0, i - args.smooth // 2); e = min(len(lab), i + args.smooth // 2 + 1)
                w = [x for x in lab[s:e] if x is not None]
                sm[i] = Counter(w).most_common(1)[0][0] if w else None
            lab = sm
        moves = sorted(g["moves"], key=lambda m: m["t"])

        def room_at(oid, t, home):
            mv = [m for m in moves if m["oid"] == oid and m["t"] <= t]
            return mv[-1]["to"] if mv else home

        # ── 정적 동반 물체를 초기 맵에서 배운다 (씬그래프 attribute)
        om = z["om"] if "om" in z.files else None
        STATIC = set(z["static"].tolist()) if "static" in z.files else set()
        okc = np.array([w in STATIC for w in vocab]) if STATIC else None
        CTX = {}
        if om is not None and okc is not None:
            mu = np.median(om, 0)
            for jj in range(om.shape[1]):
                hi = np.argsort(-om[:, jj])[:max(3, len(om) // 10)]
                lift = np.where(okc, om[hi].mean(0) - mu, -1e9)
                lift[jj] = -1e9
                CTX[jj] = np.argsort(-lift)[:args.ctx_k]

        for oid, v0 in g["gt0"].items():
            ot = v0["type"]
            if ot not in vi or not v0["room"]:
                continue
            j = vi[ot]
            truer = np.array([room_at(oid, int(t), v0["room"]) for t in ts])
            same = (gtr == truer)                     # 에이전트가 물체와 같은 방
            if same.sum() < 5 or (~same).sum() < 5:
                continue
            # ① 지각
            a = mannwhitneyu(ol[same, j], ol[~same, j], alternative="greater")[0] \
                / (same.sum() * (~same).sum())
            per.append(a)
            # ② 검색 — 방별 상위분위수가 가장 높은 방
            sc = {}
            for rr in set(lab):
                ix = np.nonzero(lab == rr)[0]
                if len(ix) >= 3:
                    sc[rr] = float(np.quantile(ol[ix, j], args.wq))
            if sc:
                ret.append(max(sc, key=sc.get) == truer[-1])
            # ④ 부재 — 물체가 떠난 방 vs 안 떠난 방에서, 떠난 뒤 점수 하락
            if args.abs_room == "ctx":
                # 그 물체의 정적 동반 물체가 **보이는** 프레임만 (키워드 제외)
                cx = CTX.get(j)
                if cx is None or not len(cx):
                    continue
                thr = np.quantile(ol[:, cx], args.ctx_q, axis=0)
                gate = (ol[:, cx] > thr).mean(1) >= 0.5
                RM = np.where(gate, "CTX", None).astype(object)
            else:
                RM = gtr if args.abs_room == "gt" else lab
            mv = [m for m in moves if m["oid"] == oid]
            if mv:
                m0 = mv[0]; R = "CTX" if args.abs_room == "ctx" else m0["frm"]
                b = np.nonzero((RM == R) & (ts <= m0["t"]))[0]
                a2 = np.nonzero((RM == R) & (ts > m0["t"]))[0]
                if len(b) >= 3 and len(a2) >= 3:
                    absn.append(("moved",
                                 float(np.quantile(ol[b, j], args.wq))
                                 - float(np.quantile(ol[a2, j], args.wq))))
            else:
                R = "CTX" if args.abs_room == "ctx" else v0["room"]
                ix = np.nonzero(RM == R)[0]
                if len(ix) >= 6:
                    h = len(ix) // 2
                    absn.append(("static",
                                 float(np.quantile(ol[ix[:h], j], args.wq))
                                 - float(np.quantile(ol[ix[h:], j], args.wq))))
        # ③ 재방문 판정 — 각 방을 실제로 다시 봤는가 vs 시스템 판단
        def maxrun(arr, v):
            best = c = 0
            for x in arr:
                c = c + 1 if x == v else 0
                best = max(best, c)
            return best
        for rr in set(kn):
            said = (maxrun(lab, rr) >= args.min_run) if args.min_run > 0 \
                else bool((lab == rr).sum() > 2)
            rev.append((bool((gtr == rr).sum() > 0), said))

    print("주택 %d" % len(glob.glob(os.path.join(args.root, "house_*"))))
    print("\n① **지각** — 물체와 같은 방에 있을 때 검출이 오르는가")
    A = np.array(per)
    print("   물체 %d · AUC 중앙 **%.3f** [%.3f %.3f] · 0.7 이상 %.0f%%"
          % (len(A), np.median(A), *np.quantile(A, [.25, .75]), 100 * (A >= .7).mean()))
    print("\n② **검색** — 어느 방에 있나")
    print("   질의 %d · 정답률 **%.3f**" % (len(ret), float(np.mean(ret))))
    print("\n③ **재방문 판정** — 그 방을 다시 봤는지")
    tp = sum(1 for a, b in rev if a and b); fn = sum(1 for a, b in rev if a and not b)
    fp = sum(1 for a, b in rev if not a and b); tn = sum(1 for a, b in rev if not a and not b)
    print("   방 %d · 정확도 %.3f · 실제 안 봤는데 봤다고 함 %d/%d"
          % (len(rev), (tp + tn) / max(len(rev), 1), fp, fp + tn))
    print("\n④ **부재** — 떠난 방에서 점수가 떨어지는가")
    mvv = [v for k, v in absn if k == "moved"]; stv = [v for k, v in absn if k == "static"]
    if len(mvv) >= 5 and len(stv) >= 5:
        u, p = mannwhitneyu(mvv, stv, alternative="greater")
        print("   이동 %d · 정적 %d · 하락 중앙 %+.4f vs %+.4f · **AUC %.3f** (p=%.3g)"
              % (len(mvv), len(stv), np.median(mvv), np.median(stv),
                 u / (len(mvv) * len(stv)), p))


if __name__ == "__main__":
    main()
