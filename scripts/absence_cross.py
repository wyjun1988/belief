#!/usr/bin/env python3
"""부재 증거 — **시퀀스 간** 채점. 시퀀스 내부 반복 이동 문제를 우회한다.

    $P scripts/absence_cross.py

동기: 시퀀스 **내부**에서 채점하면 물건이 여러 번 옮겨져 "이동 전/후" 전제가
깨진다(구간별로 잘라도 AUC 0.49~0.58, 우연 수준). ADT 논문 3.2절에 따르면
같은 아파트 시퀀스들은 **동일한 Scene frame 을 공유**한다 — 실측으로 확인했다:
6시퀀스의 정적 물체 시작 위치 차이가 중앙 0.000~0.103 m.

그래서 **시퀀스 A 의 시작 상태 vs 시퀀스 B 의 시작 상태**를 비교한다:
  이동 물체 = 두 시퀀스 사이에 위치가 바뀐 물체 (깨끗한 단일 전이)
  정적 대조 = 위치가 그대로인 물체
  부재 신호 = A 에서 그 자리의 문맥으로 검색했을 때 B 에서 물체 존재도가 떨어지나

시퀀스 내부의 다중 이동이 개입하지 않으므로 **전제가 성립하는 유일한 설정**이다.
"""
import argparse, glob, itertools, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.absence_evidence import PLACES, absence_score, pmi_graph, presence  # noqa: E402

MOVE_M = 0.3


def load(sd):
    z = np.load(os.path.join(sd, "clip_frames.npz"))
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]
    return z["emb"].astype(np.float32), z["idx"], gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/seq", "data/seq_new"])
    ap.add_argument("--gates", default="0.5,0.7,1.0")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--topm", type=int, default=4)
    ap.add_argument("--topf", type=int, default=12)
    args = ap.parse_args()

    seqs = {}
    for r in args.roots:
        for d in sorted(glob.glob(os.path.join(ROOT, r, "*/"))):
            if os.path.exists(os.path.join(d, "gt", "objects.json")) and \
               os.path.exists(os.path.join(d, "clip_frames.npz")):
                seqs[os.path.basename(d.rstrip("/"))] = d
    print("시퀀스 %d개" % len(seqs))

    # 각 시퀀스의 지각층을 한 번씩만 만든다
    cache = {}
    for n, d in seqs.items():
        E, fidx, gt = load(d)
        cats = sorted({(r.get("category") or "").strip() for r in gt.values() if r.get("category")})
        vocab = sorted(set(c for c in cats if c and len(c) > 2) | set(PLACES))
        Z, P = presence(E, vocab, args.device)
        cache[n] = dict(Z=Z, P=P, G=pmi_graph(P, vocab), vocab=vocab,
                        vi={w: i for i, w in enumerate(vocab)}, fidx=fidx, gt=gt)
        print("  %-46s 어휘 %d · 프레임 %d" % (n[:46], len(vocab), len(fidx)), flush=True)

    def first_pos(gt):
        return {(r.get("name") or ""): (np.array(r["positions"][0]),
                                        (r.get("category") or "").strip())
                for r in gt.values() if r.get("name") and r.get("positions")}

    print("\n%-8s %-6s %-6s %-8s %s" % ("게이트", "이동", "정적", "AUC", "정적오탐10% 검출률"))
    for gate in [float(x) for x in args.gates.split(",")]:
        MV, ST = [], []
        for a, b in itertools.permutations(seqs, 2):
            A, B = cache[a], cache[b]
            pa, pb = first_pos(A["gt"]), first_pos(B["gt"])
            for nm in set(pa) & set(pb):
                (posA, cat), (posB, _) = pa[nm], pb[nm]
                if not cat or cat not in A["vi"] or cat not in B["vi"]:
                    continue
                moved = float(np.linalg.norm(posA - posB)) >= MOVE_M
                # A 전체 vs B 전체 — 각 시퀀스의 "그 자리" 문맥에서 물체 존재도
                fa = list(range(len(A["fidx"])))
                fb = list(range(len(B["fidx"])))
                # 두 시퀀스의 어휘가 다를 수 있으므로 A 기준으로 채점하되
                # B 의 Z 를 이어 붙여 하나의 시계열처럼 다룬다
                if A["vocab"] != B["vocab"]:
                    continue
                Z = np.concatenate([A["Z"], B["Z"]], axis=1)
                P = np.concatenate([A["P"], B["P"]], axis=1)
                G = A["G"]
                fb2 = [i + A["Z"].shape[1] for i in fb]
                r = absence_score(Z, P, G, A["vocab"], A["vi"][cat], fa, fb2,
                                  args.topm, args.topf, gate)
                if r:
                    (MV if moved else ST).append(r["drop"])
        if not MV or not ST:
            print("%-8.1f 판정 불가 (이동 %d · 정적 %d)" % (gate, len(MV), len(ST)))
            continue
        auc = float(np.mean([(x > y) + 0.5 * (x == y) for x in MV for y in ST]))
        thr = float(np.percentile(ST, 90))
        det = float(np.mean([x > thr for x in MV]))
        print("%-8.1f %-6d %-6d **%.3f**  %.2f" % (gate, len(MV), len(ST), auc, det))


if __name__ == "__main__":
    main()
