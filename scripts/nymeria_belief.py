#!/usr/bin/env python3
"""세션 간 belief — "지난번에 저기서 봤는데, 지금은 어디 있나?"

    $P scripts/nymeria_belief.py                  # leave-one-session-out

SuperMemory 에서 막혔던 실험이다. 거기선 세션마다 좌표계가 따로라 "세션 A 의 방"과
"세션 B 의 방"을 같은 방이라 부를 근거가 없었다. Nymeria Loc_49 는 11세션이 처음부터
**하나의 graph_uid 좌표계**(±8m)라 정합 없이 바로 성립한다.

과제: 세션 s 를 가리고, s 에서 관측된 물체 클래스 c 가 **어느 방에 있을지** 맞힌다.
비교 대상 4종 —

    무작위      1/방수
    지속        c 를 마지막으로 본 방 (가장 순진한 belief: "그 자리에 있겠지")
    사전분포    다른 세션 전체에서 c 가 가장 자주 있던 방
    그래프      방의 **물체 서명**과 c 의 동시출현 이웃을 맞춰본다 —
                즉 다세션 씬그래프를 실제로 쓴다

'그래프'가 '지속'을 이기면 다세션 지도가 belief 에 기여한 것이고, 지면 지도 없이
마지막 관측만 기억해도 같다는 뜻이다. 이 대조가 이 실험의 전부다.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.nymeria_graph import D, house_frame, traj_of        # noqa: E402


def collapse(det, syn_path, max_surface=None):
    """표면형 검출을 개념 단위로 합친다(개념 점수 = 표면형 최댓값).

    검출을 재라벨링하는 것이 아니라 **같은 개념을 여러 말로 물어본 결과를 합치는**
    것이다. 표면형끼리는 make_synonyms 단계에서 충돌을 이미 걸렀으므로 한 검출이
    두 개념에 들어가지 않는다."""
    syn = json.load(open(syn_path))
    s2c = {}
    for c, d in syn.items():
        for w in (d["surface"][:max_surface] if max_surface else d["surface"]):
            s2c.setdefault(w, c)
    out = {}
    for name, rows in det.items():
        nr = []
        for r in rows:
            best = {}
            for w, sc in r["det"].items():
                c = s2c.get(w)
                if c is not None and sc > best.get(c, 0):
                    best[c] = sc
            nr.append({"sec": r["sec"], "det": best})
        out[name] = nr
    return out


def assign(det, seqs, cen, e1, e2, ctr, score_thr=0.35):
    """(세션, 클래스) → 방별 관측수. 검출 시각을 궤적 시각에 붙여 방을 정한다.

    ⚠️ score_thr 가 이 함수의 핵심이다. 방을 **착용자 위치**로 정하므로, 문 너머나
    먼 거리에서 잡힌 약한 검출은 물체를 엉뚱한 방에 넣는다. 실측: 움직일 수 없는
    가전(stove/oven/microwave/kitchen counter)의 세션 간 방 일치도가 문턱 0 에서
    0.53, 0.35 에서 **0.93** 이다(stove 0.50→1.00). 관측은 1/3 로 줄지만 지도가
    비로소 물체의 위치를 뜻하게 된다."""
    obs = defaultdict(Counter)
    frames = []                                   # (세션, 방, 클래스집합) — 공기 계산용
    for name, rows in det.items():
        if name not in seqs:
            continue
        sec, P = seqs[name]
        uv = np.stack([P @ e1, P @ e2], 1) - ctr
        for r in rows:
            k = int(np.argmin(np.abs(sec - r["sec"])))
            rm = int(np.argmin(np.linalg.norm(uv[k] - cen, axis=1)))
            ws = {w for w, sc in r["det"].items() if sc >= score_thr}
            if not ws:
                continue
            frames.append((name, rm, ws))
            for w in ws:
                obs[(name, w)][rm] += 1
    return obs, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--score-thr", type=float, default=0.35,
                    help="검출 점수 문턱 — 근접 관측만 남겨 방 배정 오염을 막는다")
    ap.add_argument("--min-obs", type=int, default=3,
                    help="세션 내 관측 수가 이보다 적은 클래스는 채점에서 뺀다(잡음)")
    ap.add_argument("--det", default="owl_det.json",
                    help="지각층 검출 파일. clip_det.json 을 주면 **같은 프레임의 CLIP**"
                         " 대조군이 된다 (점수 척도가 z 라 문턱 의미가 다르다)")
    ap.add_argument("--synonyms", default=None,
                    help="표면형 메타데이터 — 개념 점수를 표면형 최댓값으로 합친다")
    ap.add_argument("--max-surface", type=int, default=None,
                    help="개념당 표면형 상한(1 = 정식이름만 = 같은 검출 파일 내 대조군)")
    ap.add_argument("--out", default=os.path.join(D, "belief.json"))
    args = ap.parse_args()

    det = json.load(open(os.path.join(D, args.det)))
    if args.synonyms:
        det = collapse(det, os.path.join(D, args.synonyms), args.max_surface)
        print("표면형 집계: %s (개념당 최대 %s개)"
              % (args.synonyms, args.max_surface or "전체"))
    seqs = {}
    import glob
    for sd in sorted(glob.glob(os.path.join(D, "loc49", "*"))):
        if os.path.isdir(sd):
            try:
                seqs[os.path.basename(sd)] = traj_of(sd)
            except Exception:
                pass
    _, e1, e2, ctr = house_frame(seqs)
    P = np.concatenate([p for _, p in seqs.values()])
    U = np.stack([P @ e1, P @ e2], 1) - ctr
    from scipy.cluster.vq import kmeans2
    cen, _ = kmeans2(U, args.k, minit="++", seed=0, iter=60)

    obs, frames = assign(det, seqs, cen, e1, e2, ctr, args.score_thr)
    sess = sorted({s for s, _ in obs})
    print("세션 %d · 방 %d · (세션,클래스) 쌍 %d" % (len(sess), args.k, len(obs)))

    # 세션 순서 = 이름순(녹화 순). '지속'은 직전 세션까지의 마지막 관측을 쓴다.
    hit = Counter()
    n = 0
    per_room = Counter()
    for si, s in enumerate(sess):
        others = [t for t in sess if t != s]
        prior = [t for t in sess[:si]]                       # 인과: 과거 세션만
        # 홀드아웃 세션을 뺀 방 서명과 공기 이웃 (누수 방지)
        sig = defaultdict(Counter)
        nb_of = defaultdict(Counter)
        for name, rm, ws in frames:
            if name == s:
                continue
            for w in ws:
                sig[rm][w] += 1
            for w in ws:
                for u in ws:
                    if u != w:
                        nb_of[w][u] += 1
        for (ss, c), cnt in obs.items():
            if ss != s or sum(cnt.values()) < args.min_obs:
                continue
            truth = cnt.most_common(1)[0][0]
            n += 1
            per_room[truth] += 1

            # 지속 — 가장 최근 과거 세션에서 본 방
            last = None
            for t in reversed(prior):
                if (t, c) in obs:
                    last = obs[(t, c)].most_common(1)[0][0]
                    break
            hit["지속"] += (last == truth) if last is not None else 1.0 / args.k

            # 사전분포 — 다른 모든 세션 합산 최빈 방
            agg = Counter()
            for t in others:
                agg += obs.get((t, c), Counter())
            hit["사전분포"] += (agg.most_common(1)[0][0] == truth) if agg else 1.0 / args.k

            # 그래프 — 방 서명 vs c 의 **프레임 공기 이웃**. c 자신은 뺀다
            # (넣으면 사전분포와 같아진다). 이웃이 어느 방 서명과 닮았는지로 고른다.
            nb = nb_of.get(c, Counter())
            if nb:
                sc = {}
                for r in range(args.k):
                    tot = sum(sig[r].values()) or 1
                    sc[r] = sum(v * sig[r].get(w, 0) / tot for w, v in nb.items())
                hit["그래프"] += (max(sc, key=sc.get) == truth)
            else:
                hit["그래프"] += 1.0 / args.k

    print("\n채점 %d건 (leave-one-session-out · 방 분포 %s)"
          % (n, dict(sorted(per_room.items()))))
    base = 1.0 / args.k
    # 최빈방이 진짜 기준선이다 — 무작위(1/k)는 너무 관대해서 개선을 과장한다
    maj = max(per_room.values()) / n
    print("\n%-10s %-8s %s" % ("방식", "정확도", "최빈방 대비"))
    print("%-10s %.3f    —" % ("무작위", base))
    print("%-10s %.3f    —" % ("최빈방", maj))
    for kname in ("지속", "사전분포", "그래프"):
        a = hit[kname] / n
        print("%-10s **%.3f**   %+.0f%%" % (kname, a, 100 * (a / maj - 1)))
    json.dump({k: hit[k] / n for k in hit} | {"n": n, "random": base},
              open(args.out, "w"))
    print("\n→ %s" % args.out)


if __name__ == "__main__":
    main()
