#!/usr/bin/env python3
"""부재 증거 — 두 ADT 시퀀스를 합쳐 채점한다.

    $P scripts/absence_pool.py --owl-dir data/adt_owl

시퀀스당 이동 물체가 4~5개뿐이라 단일 시퀀스 AUC 는 1~4개 표본에 얹힌다
(실측: 게이트를 올리면 판정 가능 이동물체가 1개가 되고 AUC 0.99 가 '나온다').
두 시퀀스의 하락값을 **한 풀에 모아** 채점하면 이동 9개 대 정적 170여개가 되어
그나마 읽을 수 있는 수가 된다. 그래도 작다 — 이 지표의 한계다.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.absence_evidence import (PLACES, absence_score, pmi_graph,  # noqa: E402
                                      presence)
from scripts.owl_presence import owl_z, report_src                       # noqa: E402

SEQS = ["Apartment_release_decoration_seq137_M1292",
        "Apartment_release_multiskeleton_party_seq102_M1292"]
# 신규 4시퀀스(multiuser_clean) — 이동 물체 75개. 기존 2시퀀스가 11개뿐이라
# AUC 0.761 의 신뢰구간이 넓었다. --root 를 data/seq_new 로 주면 이쪽을 쓴다.
SEQS_NEW = ["Apartment_release_multiuser_clean_seq114_M1292",
            "Apartment_release_multiuser_clean_seq117_M1292",
            "Apartment_release_multiuser_clean_seq118_M1292",
            "Apartment_release_multiuser_clean_seq119_M1292"]
OWLF = {"Apartment_release_decoration_seq137_M1292": "owl_adt_decoration.json",
        "Apartment_release_multiskeleton_party_seq102_M1292": "owl_adt_party.json"}


RAW = [False]


def one(seq, root, owl_dir, gate, thr, device="mps"):
    sd = os.path.join(root, seq)
    z = np.load(os.path.join(sd, "clip_frames.npz"))
    E, fidx = z["emb"].astype(np.float32), z["idx"]
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]
    cats = sorted({(r.get("category") or "").strip() for r in gt.values() if r.get("category")})
    vocab = sorted(set(c for c in cats if c and len(c) > 2) | set(PLACES))
    if owl_dir:
        raw = json.load(open(os.path.join(owl_dir, OWLF[seq])))
        owl = {"_": {int(os.path.splitext(k)[0]): v for k, v in raw.items()}}
        Z, src = owl_z(owl, [("_", int(f)) for f in fidx], vocab, E=E, device=device, thr=thr)
        P = Z > 1.5
    else:
        Z, P = presence(E, vocab, device)
        src = None
    G = pmi_graph(P, vocab)
    vi = {w: i for i, w in enumerate(vocab)}
    movers, statics = {}, []
    for k, r in gt.items():
        c = (r.get("category") or "").strip()
        if c not in vi:
            continue
        if r.get("moves"):
            movers[c] = (min(m["start_idx"] for m in r["moves"]),
                         max(m["end_idx"] for m in r["moves"]))
        elif r.get("motion_type") == "static":
            statics.append(c)
    statics = [c for c in set(statics) if c not in movers]
    mv, st = [], []
    for c, (s, e) in movers.items():
        fa = [i for i, f in enumerate(fidx) if f < s]
        fb = [i for i, f in enumerate(fidx) if f > e]
        r = absence_score(Z, P, G, vocab, vi[c], fa, fb, 4, 12, gate)
        if r:
            mv.append(r["drop"] if RAW[0] else r["ndrop"])
    mid = int(np.median(fidx))
    for c in statics:
        fa = [i for i, f in enumerate(fidx) if f < mid]
        fb = [i for i, f in enumerate(fidx) if f >= mid]
        r = absence_score(Z, P, G, vocab, vi[c], fa, fb, 4, 12, gate)
        if r:
            st.append(r["drop"] if RAW[0] else r["ndrop"])
    return mv, st, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--seqs", default=None,
                    help="쉼표 구분 시퀀스명. 'new' 면 신규 4시퀀스(이동 75개)")
    ap.add_argument("--owl-dir", default=None)
    ap.add_argument("--owl-thr", type=float, default=0.0)
    ap.add_argument("--gates", default="0.5,0.7,1.0")
    ap.add_argument("--norm", action="store_true",
                    help="문맥 하락으로 정규화한 ndrop 으로 채점. **기본은 원 drop 이다** —"
                         " 정규화 변형은 2026-08-17 에 이미 기각됐다(시퀀스 간 일관성 없음)."
                         " 실측: 풀링 AUC 가 원 drop 0.761 vs 정규화 0.543 으로 정규화가"
                         " 이 지표를 깎는다")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    RAW[0] = not args.norm

    seqs = SEQS
    if args.seqs == "new":
        seqs = SEQS_NEW
    elif args.seqs:
        seqs = [x.strip() for x in args.seqs.split(",") if x.strip()]
    print("시퀀스 %d개: %s" % (len(seqs), ", ".join(s[-12:] for s in seqs)))
    print("%-8s %-6s %-6s %-8s %s" % ("게이트", "이동", "정적", "AUC", "정적오탐10% 검출률"))
    for g in [float(x) for x in args.gates.split(",")]:
        MV, ST = [], []
        for seq in seqs:
            mv, st, _ = one(seq, args.root, args.owl_dir, g, args.owl_thr, args.device)
            MV += mv
            ST += st
        if not MV or not ST:
            print("%-8.1f 판정 불가" % g)
            continue
        auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in MV for b in ST]))
        thr = float(np.percentile(ST, 90))
        det = float(np.mean([a > thr for a in MV]))
        print("%-8.1f %-6d %-6d **%.3f**  %.2f" % (g, len(MV), len(ST), auc, det))


if __name__ == "__main__":
    main()
