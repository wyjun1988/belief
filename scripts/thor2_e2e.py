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


def trmat(kn, stay, adj, leak):
    """방 전이 행렬. adj 가 있으면 **문으로 연결된 방**에만 질량을 준다."""
    K = len(kn); P = np.zeros((K, K))
    for i, ri in enumerate(kn):
        nb = [j for j, rj in enumerate(kn)
              if j != i and (rj in adj.get(ri, []) if adj else True)]
        non = [j for j in range(K) if j != i and j not in nb]
        rest = 1 - stay
        if nb:
            for j in nb:
                P[i, j] = rest * (1 - leak) / len(nb)
            for j in non:
                P[i, j] = rest * leak / max(len(non), 1)
        else:
            for j in non:
                P[i, j] = rest / max(len(non), 1)
        P[i, i] = stay
    P = np.maximum(P, 1e-9); P /= P.sum(1, keepdims=True)
    return np.log(P)


def viterbi(S, stay, temp=0.01, tr=None):
    """**방을 프레임마다 독립으로 맞히지 않는다 — 전이를 추적한다.**
    사람은 방을 순간이동하지 않는다; 문을 지나야 바뀐다. 중앙값 평활은 이것의
    조잡한 판본이다. 실측: 독립 argmax 0.469 · 평활 0.554 · **전이추적 0.825**.
    ⚠️ 방 키 유사도의 여유가 0.000 수준이라 **온도로 나눠** 로그확률로 만들어야 한다."""
    Z = (S - S.max(1, keepdims=True)) / temp
    logem = Z - np.log(np.exp(Z).sum(1, keepdims=True) + 1e-12)
    T, K = logem.shape
    if tr is None:
        tr = np.full((K, K), np.log((1 - stay) / max(K - 1, 1)))
        np.fill_diagonal(tr, np.log(stay))
    dp = logem[0].copy(); bp = np.zeros((T, K), int)
    for t in range(1, T):
        m = dp[:, None] + tr
        bp[t] = np.argmax(m, 0); dp = m.max(0) + logem[t]
    path = np.zeros(T, int); path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


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
    ap.add_argument("--topology", default=None,
                    help="방 인접 그래프(문 연결). ⚠️ 56 에서 무효였던 이유는 방 노드가 "
                         "**하나뿐**이라 전이 제약이 걸릴 자리가 없어서였다. "
                         "--place-node 와 함께 쓰면 이득이 난다(0.886 → 0.895).")
    ap.add_argument("--topo-leak", type=float, default=0.03)
    ap.add_argument("--place-node", action="store_true",
                    help="**방 키를 평균 벡터 하나가 아니라 여러 노드로.** ⚠️ 방마다 맵 프레임 "
                         "12~120장을 한 벡터로 평균하면 여러 방향에서 본 모습이 뭉개진다 — "
                         "맞는 방 0.928 vs 최고 오답 0.930(여유 0.000). 노드별 최대유사도를 "
                         "방 점수로 쓰면 다봉 분포가 살아난다. 실측 방 식별 0.825 → 0.886.")
    ap.add_argument("--stay", type=float, default=0.0,
                    help="**전이 추적(Viterbi)의 머무를 확률.** 0이면 끔(종전 방식). "
                         "0.99 권장 — 방 식별 0.469 → 0.825.")
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
    ap.add_argument("--loc", default="topf", choices=["topf", "roomq", "roomlift", "roomrate", "roomcount", "roomctx", "nodeq"],
                    help="물체 위치 추정. ⚠️ 기본 `topf`(점수 상위 3프레임의 방)는 "
                         "**음성이 570장인데 argmax 를 쓴다** — AUC 0.875 여도 최고점 "
                         "3장이 오검출일 확률이 높다. "
                         "roomq=방별 상위분위수가 가장 높은 방, "
                         "roomlift=방별 분위수 − 그 물체의 전체 분위수(자기 기준 보정). "
                         "**roomrate=방별 '문턱 넘는 프레임 비율'.** ⚠️ 분위수는 "
                         "**프레임 수에 민감하다** — 물체가 없는 방도 프레임이 144장이면 "
                         "오검출의 상위 분위수가 높아진다. 비율은 표본 수에 강건하다.")
    ap.add_argument("--targets", default=None,
                    help="**타겟 물체 목록**(thor_detectability.json) — 유형별 '보일 때 검출도'. "
                         "⚠️ 이 필터는 **다른 자료**(thor2v, 가시성 GT 보유)에서 정하고 "
                         "여기에 적용한다 — 같은 자료에서 정하면 누수다. "
                         "연필 0.01 · 시계 0.01 처럼 보일 때조차 안 잡히는 물체는 "
                         "관측으로 답할 수 없으므로 타겟에서 뺀다.")
    ap.add_argument("--target-thr", type=float, default=0.10)
    ap.add_argument("--cond2", type=float, default=0.0,
                    help="**조건② 를 검색에도 적용한다.** 그 물체가 기록 전체에서 이 값도 "
                         "못 넘으면 질의에서 뺀다. ⚠️ 부재에는 계속 걸었는데 검색에는 "
                         "안 걸고 있었다 — 실측: 보일 때 검출도 상위 25%% 물체는 검색 0.762, "
                         "하위 25%%(연필·시계·빵, 검출도 0.01)는 0.333.")
    ap.add_argument("--abs-thr", type=float, default=0.30,
                    help="`roomcount` 용 **절대 문턱**. ⚠️ `roomrate` 는 문턱을 "
                         "**그 물체 점수의 분위수**로 잡았다 — 같은 잡음 분포에서 뽑은 "
                         "문턱이라 누적의 이점이 사라졌다(0.393, 분위수와 동일). "
                         "검출기 자체의 동작점을 절대값으로 주면 '봤다' 를 셀 수 있다.")
    ap.add_argument("--node-smooth", type=float, default=0.0,
                    help="`nodeq` 에서 이웃 노드로 점수를 퍼뜨리는 정도(같은 방 노드끼리).")
    ap.add_argument("--ctx-k", type=int, default=5, help="동반 물체 수")
    ap.add_argument("--ctx-w", type=float, default=0.5, help="동반 물체 가중")
    ap.add_argument("--only-seen", action="store_true",
                    help="에이전트가 물체와 **한 번이라도 같은 방에 있었던** 질의만. "
                         "⚠️ `--oracle-find` 는 이 조건을 자동으로 걸어 표본이 줄므로, "
                         "공정 비교하려면 다른 설정에도 같이 걸어야 한다.")
    ap.add_argument("--oracle-find", action="store_true",
                    help="**검색을 GT 로** — 물체를 마지막으로 볼 수 있었던 시점의 실제 방")
    ap.add_argument("--oracle-revisit", action="store_true",
                    help="**재방문 판정을 GT 로** — 그 방을 실제로 다시 봤는가")
    ap.add_argument("--oracle-absent", action="store_true",
                    help="**부재 판정을 GT 로** — 물체가 그 방을 실제로 떠났는가")
    ap.add_argument("--oracle-room", action="store_true",
                    help="배회 프레임의 방을 GT 로 대체 — **방 재식별 오류와 검색 오류를 가른다**")
    ap.add_argument("--queries", type=int, default=4, help="주택당 질의 시각 수")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    TARGETS = None
    if args.targets:
        d = json.load(open(args.targets))
        TARGETS = {k for k, v in d.items() if v >= args.target_thr}
        print("타겟 물체 %d종 (문턱 %.2f)" % (len(TARGETS), args.target_thr))
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
        elif args.stay > 0:
            if args.place_node:
                mrr = np.array(mrooms, object)
                Snode = el[sel] @ em.T
                sc_room = np.stack([Snode[:, mrr == r].max(1) for r in kn], 1)
            else:
                sc_room = el[sel] @ K.T
            nodeassign = np.argmax(el[sel] @ em.T, 1)
            TR = None
            if args.topology:
                _tp = json.load(open(args.topology))
                TR = trmat(kn, args.stay, _tp.get(os.path.basename(hd), {}), args.topo_leak)
            lab = np.array(kn)[viterbi(sc_room, args.stay, tr=TR)]
        else:
            nodeassign = np.argmax(el[sel] @ em.T, 1)
            lab = np.array([kn[int(np.argmax(K @ el[i]))] for i in sel])
            lab = smooth(lab, args.smooth)
        ok = gtr != None
        if ok.sum():
            racc.append(float(np.mean(lab[ok] == gtr[ok])))
        # ── 질의 시각
        Q = [int(len(tt) * f) for f in np.linspace(0.4, 0.98, args.queries)]
        moves = sorted(g["moves"], key=lambda m: m["t"])
        # ── 초기 맵에서 **동반 물체**를 배운다 (설치 시 씬그래프 attribute)
        # ⚠️ 배회 기록에서 배우면 순환이다 — 물체를 못 찾으니 문맥도 못 배운다.
        # 초기 맵은 정지 촬영이라 조건이 좋고, 이것이 곧 씬그래프에 저장할 정보다.
        om = z["om"] if "om" in z.files else None
        # ⚠️ 동반 물체는 **정적인 것만** 쓴다. 움직이는 물체를 문맥으로 쓰면
        # 그것도 옮겨지므로 방해가 된다(사용자 지적). 침대·소파·조리대처럼
        # 안 움직이는 것이 방을 특정하는 안정적 앵커다.
        STATIC = set(z["static"].tolist()) if "static" in z.files else set()
        ok_ctx = np.array([w in STATIC for w in vocab]) if STATIC else None
        CTX = {}
        if om is not None and args.loc == "roomctx":
            mu = np.median(om, 0)
            for jj in range(om.shape[1]):
                hi = np.argsort(-om[:, jj])[:max(3, len(om) // 10)]
                lift = om[hi].mean(0) - mu
                lift[jj] = -1e9
                if ok_ctx is not None:
                    lift = np.where(ok_ctx, lift, -1e9)
                CTX[jj] = np.argsort(-lift)[:args.ctx_k]
        for qi in Q:
            T = int(tt[qi])
            for oid, v0 in g["gt0"].items():
                ot = v0["type"]
                if ot not in vi or not v0["room"]:
                    continue
                if TARGETS is not None and ot not in TARGETS:
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
                if args.cond2 > 0 and float(np.quantile(O[upto, j], 0.99)) < args.cond2:
                    continue          # 기록 어디서도 안 잡히는 물체는 관측으로 답할 수 없다
                if args.only_seen or args.oracle_find:
                    co = [i for i in upto
                          if gtr[i] is not None
                          and gtr[i] == (([m for m in moves if m["oid"] == oid and m["t"] <= tt[i]]
                                          or [dict(to=v0["room"])])[-1]["to"])]
                    if not co:
                        continue
                # ── ② 검색: 물체가 어느 방에 있(었)나
                if args.oracle_find:
                    # 물체와 같은 방에 있었던 마지막 시점의 실제 방
                    seen = [i for i in upto
                            if gtr[i] is not None
                            and gtr[i] == (mv[-1]["to"] if [m for m in moves
                                                           if m["oid"] == oid and m["t"] <= tt[i]]
                                           else v0["room"])]
                    if not seen:
                        continue
                    li = seen[-1]
                    r_pred = gtr[li]; t_seen = int(tt[li])
                    top = np.array([li])
                elif args.loc == "topf":
                    top = upto[np.argsort(-O[upto, j])[:3]]
                    r_pred = Counter(lab[top]).most_common(1)[0][0]
                    t_seen = int(tt[top].max())
                elif args.loc == "nodeq":
                    # ⚠️ **방 안 프레임을 분위수 하나로 뭉개면 안 된다.** 방에는 시점이
                    # 여럿인데 물체는 한두 곳에서만 보인다 — 방 키를 평균하던 것과 같은 실수다.
                    # 노드(=맵 프레임 시점)별로 집계하면 물체가 있는 시점에 봉우리가 선다.
                    na = nodeassign[upto]
                    nsc = {}
                    for nd in set(na.tolist()):
                        ix = upto[na == nd]
                        if len(ix) >= 2:
                            nsc[nd] = float(np.quantile(O[ix, j], args.wq))
                    if not nsc:
                        continue
                    if args.node_smooth > 0:
                        mrr = np.array(mrooms, object)
                        sm2 = {}
                        for nd, v in nsc.items():
                            sib = [u for u in nsc if mrr[u] == mrr[nd] and u != nd]
                            sm2[nd] = v + args.node_smooth * (
                                np.mean([nsc[u] for u in sib]) if sib else 0.0)
                        nsc = sm2
                    bn = max(nsc, key=nsc.get)
                    r_pred = np.array(mrooms, object)[bn]
                    ix = upto[na == bn]
                    top = ix[np.argsort(-O[ix, j])[:3]]
                    t_seen = int(tt[top].max())
                else:
                    base = float(np.quantile(O[upto, j], args.wq))
                    sc_r = {}
                    for rr in set(lab[upto]):
                        ix = upto[lab[upto] == rr]
                        if len(ix) < 3:
                            continue
                        if args.loc == "roomctx":
                            own = float(np.quantile(O[ix, j], args.wq))
                            cx = CTX.get(j)
                            ctx = float(np.mean([np.quantile(O[ix, c], args.wq)
                                                 for c in cx])) if cx is not None and len(cx) else 0.0
                            sc_r[rr] = own + args.ctx_w * ctx
                        elif args.loc == "roomcount":
                            sc_r[rr] = float(np.mean(O[ix, j] > args.abs_thr))
                        elif args.loc == "roomrate":
                            thr = float(np.quantile(O[upto, j], 0.90))
                            sc_r[rr] = float(np.mean(O[ix, j] > thr))
                        else:
                            q = float(np.quantile(O[ix, j], args.wq))
                            sc_r[rr] = q - (base if args.loc == "roomlift" else 0.0)
                    if not sc_r:
                        continue
                    r_pred = max(sc_r, key=sc_r.get)
                    ix = upto[lab[upto] == r_pred]
                    top = ix[np.argsort(-O[ix, j])[:3]]
                    t_seen = int(tt[top].max())
                # ── ③ 재방문 · ④ 부재
                if args.oracle_revisit:
                    revis_sys = any(gtr[i] == r_pred for i in upto if tt[i] > t_seen)
                else:
                    inr_a0 = [i for i in upto if lab[i] == r_pred and tt[i] > t_seen]
                    if args.min_run > 0 and inr_a0:
                        run = c = 0
                        for i in upto:
                            if tt[i] > t_seen and lab[i] == r_pred:
                                c += 1; run = max(run, c)
                            else:
                                c = 0
                        if run < args.min_run:
                            inr_a0 = []
                    revis_sys = len(inr_a0) >= 2
                inr_b = [i for i in upto if lab[i] == r_pred and tt[i] <= t_seen]
                inr_a = [i for i in upto if lab[i] == r_pred and tt[i] > t_seen]
                sb = float(np.quantile(O[inr_b, j], args.wq)) if len(inr_b) else 0.0
                sa = float("nan")
                if not revis_sys:
                    state = "b"
                elif args.oracle_absent:
                    state = "c" if r_true != r_pred else "a"
                else:
                    sa = float(np.quantile(O[inr_a, j], args.wq)) if len(inr_a) else 0.0
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
