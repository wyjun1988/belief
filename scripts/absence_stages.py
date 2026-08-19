#!/usr/bin/env python3
"""부재 증거를 **단계별로** 진단한다 — 장소 찾기인가, 키워드 채널인가.

    $P scripts/absence_stages.py

⑬ 은 "지각층이 존재/부재를 못 가른다(AUC 0.506)"로 결론냈는데, 그 테스트는
**전체 프레임 평균**이었다. 씬에 있는 물체도 918프레임 중 몇 장에만 보이므로
평균을 내면 없는 물체와 같아진다 — **설계와 다른 조건**이었다.

설계는 이렇다: 키워드를 뺀 **문맥(물체 조합)으로 장소 프레임을 먼저 찾고**, 그
안에서만 키워드 존재를 잰다. 그래서 진단도 그 단계를 따라간다:

  ⓪ 라벨   "이동" 이라도 변위가 작으면 **자리를 뜬 것이 아니다**. 부재 라벨로 부적격
  Ⓐ 장소   GT 포즈로 "그 자리가 보이는 프레임"을 만들고, 문맥 게이트가 그것을
            얼마나 맞히는지 잰다  → 낮으면 **장소 찾기가 문제**
  Ⓑ 키워드  장소를 GT 로 주고(오라클) 이동 전/후 키워드 z 를 비교한다
            → 안 갈리면 **키워드 채널이 한계**

Ⓑ 가 되는데 Ⓐ 가 나쁘면 부재는 **고칠 수 있는 문제**다.
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.absence_evidence import PLACES, context_of, pmi_graph, presence  # noqa
from scripts.owl_presence import owl_z                                       # noqa

SEQS = {"Apartment_release_decoration_seq137_M1292": "owl_adt_decoration.json",
        "Apartment_release_multiskeleton_party_seq102_M1292": "owl_adt_party.json"}


def visible(p, T, K, W, H, zmin=0.3, zmax=8.0, margin=0):
    """세계좌표 p 가 각 프레임에서 화면 안에 들어오는가. T: (N,4,4) cam→world."""
    R = T[:, :3, :3]; t = T[:, :3, 3]
    pc = np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), p[None] - t)
    z = pc[:, 2]
    ok = (z > zmin) & (z < zmax)
    u = K[0, 0] * pc[:, 0] / np.where(z == 0, 1e-9, z) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.where(z == 0, 1e-9, z) + K[1, 2]
    return ok & (u > margin) & (u < W - margin) & (v > margin) & (v < H - margin)


MODES = ["max", "mean", "cnt2", "cnt3", "min"]


def gate_of(Z, ctx, mode, thr):
    """문맥 프레임 선택 — 조합을 어떻게 요구하는가."""
    C = Z[ctx]
    if mode == "max":                       # 하나만 떠도 통과 (현행)
        return C.max(0) >= thr
    if mode == "mean":                      # 평균이 문턱 이상
        return C.mean(0) >= thr
    if mode == "min":                       # 전부 동시에
        return C.min(0) >= thr
    n = int(mode[3:])                       # cntN — N개 이상 동시에
    return (C >= thr).sum(0) >= min(n, len(ctx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--owl-dir", default=os.path.join(ROOT, "data", "adt_owl"))
    ap.add_argument("--min-disp", type=float, default=1.0,
                    help="이 변위 미만은 '자리를 뜬 것' 으로 안 본다")
    ap.add_argument("--ctx-gate", type=float, default=1.0)
    ap.add_argument("--topm", type=int, default=4)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    rows_A, rows_B = [], []
    ora_mv, ora_st = [], []      # 장소 오라클 AUC 용 (이동 / 정적)
    for seq, owlf in SEQS.items():
        sd = os.path.join(args.root, seq)
        z = np.load(os.path.join(sd, "clip_frames.npz"))
        E, fidx = z["emb"].astype(np.float32), z["idx"]
        ci = json.load(open(os.path.join(sd, "camera_info.json")))
        K = np.array(ci["intrinsics"], float); W, H = ci["width"], ci["height"]
        T = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
        gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]
        cats = sorted({(r.get("category") or "").strip() for r in gt.values()
                       if r.get("category")})
        vocab = sorted(set(c for c in cats if c and len(c) > 2) | set(PLACES))
        raw = json.load(open(os.path.join(args.owl_dir, owlf)))
        owl = {"_": {int(os.path.splitext(k)[0]): v for k, v in raw.items()}}
        Z, _ = owl_z(owl, [("_", int(f)) for f in fidx], vocab, E=E,
                     device=args.device, thr=0.0)
        G = pmi_graph(Z > 1.5, vocab)
        vi = {w: i for i, w in enumerate(vocab)}
        Tf = T[fidx]                                   # clip 프레임에 맞춘 포즈

        disp = []
        for k, r in gt.items():
            c = (r.get("category") or "").strip()
            if c not in vi:
                continue
            for mv in (r.get("moves") or []):
                disp.append(mv["displacement_m"])
                if mv["displacement_m"] < args.min_disp:
                    continue
                ki = vi[c]
                ctx = context_of(G, vocab, ki, args.topm)
                if not ctx:
                    continue
                pfrom = np.array(mv["from"], float)
                vis = visible(pfrom, Tf, K, W, H)      # 그 자리가 보이는 프레임
                s, e = mv["start_idx"], mv["end_idx"]
                pre = (fidx < s) & vis
                post = (fidx > e) & vis
                # Ⓐ 문맥 게이트가 고른 프레임 vs GT 가시 프레임
                # 현행은 max — **이웃 하나만 떠도 통과**하므로 조합이 아니라 OR 다.
                # 조합답게 여러 개를 동시에 요구하면 정밀도가 오르는지 본다.
                for mode in MODES:
                    gsel = gate_of(Z, ctx, mode, args.ctx_gate)
                    if not gsel.sum() or not vis.sum():
                        continue
                    inter = (gsel & vis).sum()
                    rows_A.append((mode, seq[:20], c, float(inter / gsel.sum()),
                                   float(inter / vis.sum()), int(gsel.sum())))
                # Ⓑ 장소 오라클 — GT 가시 프레임 안에서 이동 전/후 키워드 z
                if pre.sum() >= 3 and post.sum() >= 3:
                    ora_mv.append(float(np.median(Z[ki, pre]) - np.median(Z[ki, post])))
                    rows_B.append((seq[:20], c, mv["displacement_m"],
                                   float(np.median(Z[ki, pre])),
                                   float(np.median(Z[ki, post])),
                                   int(pre.sum()), int(post.sum())))
        # Ⓒ 장소 오라클 AUC — 정적 물체도 같은 방식으로 전/후를 잰다.
        #    장소 찾기를 GT 로 대체했을 때 부재가 갈리는지가 이 과제의 **상한**이다.
        moved = {(r.get("category") or "").strip() for r in gt.values() if r.get("moves")}
        mid = int(np.median(fidx))
        for k, r in gt.items():
            c = (r.get("category") or "").strip()
            if c not in vi or c in moved or not r.get("positions"):
                continue
            p0 = np.array(r["positions"][0], float)
            vis = visible(p0, Tf, K, W, H)
            a = (fidx < mid) & vis; b = (fidx >= mid) & vis
            if a.sum() >= 3 and b.sum() >= 3:
                ki = vi[c]
                ora_st.append(float(np.median(Z[ki, a]) - np.median(Z[ki, b])))
        d = np.array(disp)
        print("[%s] 이동 이벤트 %d · 변위 중앙 %.2f m · **%.0f%% 가 %.1f m 미만**"
              % (seq[:24], len(d), np.median(d),
                 100 * (d < args.min_disp).mean(), args.min_disp))

    print("\nⒶ 문맥 조합 방식별 — '그 자리가 보이는 프레임' 을 맞히는가")
    print("   %-6s %-8s %-8s %-8s %s" % ("방식", "정밀도", "재현율", "F1", "선택프레임"))
    for mode in MODES:
        rs = [r for r in rows_A if r[0] == mode]
        if not rs:
            print("   %-6s 선택 프레임 없음" % mode)
            continue
        pr = np.median([r[3] for r in rs]); rc = np.median([r[4] for r in rs])
        f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0
        print("   %-6s %-8.2f %-8.2f %-8.2f %d (n=%d)"
              % (mode, pr, rc, f1, int(np.median([r[5] for r in rs])), len(rs)))

    print("\nⒸ 장소 오라클 AUC — 장소 찾기를 GT 로 대체했을 때의 상한")
    if ora_mv and ora_st:
        a = np.array(ora_mv); b = np.array(ora_st)
        auc = float(np.mean([(x > y) + 0.5 * (x == y) for x in a for y in b]))
        print("   이동 %d · 정적 %d · **AUC %.3f** (우연 0.50)" % (len(a), len(b), auc))
        print("   하락 중앙: 이동 %+.3f · 정적 %+.3f" % (np.median(a), np.median(b)))
    else:
        print("   판정 가능 없음")

    print("\nⒷ 장소를 GT 로 준 뒤 이동 전/후 키워드 z")
    if rows_B:
        a = np.array([r[3] for r in rows_B]); b = np.array([r[4] for r in rows_B])
        print("   전 중앙 %.3f · 후 중앙 %.3f · **하락 중앙 %.3f** (n=%d)"
              % (np.median(a), np.median(b), np.median(a - b), len(rows_B)))
        print("   하락이 양수인 비율 **%.0f%%** (우연 50%%)"
              % (100 * (a > b).mean()))
        for r in rows_B:
            print("     %-20s %-16s 변위 %.2fm  전 %+.2f → 후 %+.2f  (%d/%d프레임)"
                  % (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    else:
        print("   판정 가능 없음 — 변위 문턱을 낮추거나 가시 프레임이 부족하다")


if __name__ == "__main__":
    main()
