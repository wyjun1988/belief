#!/usr/bin/env python3
"""방을 **연결성분**으로 정의한다 — k-means 볼록 셀의 대안.

    $P scripts/room_split_connect.py --sess s1 s8

⑰ 실측: 위치는 방을 담고 있다(1-NN 0.784, 최빈 0.547 대비 +43%). 그런데 k-means
군집으로 뭉치면 명명을 오라클로 줘도 0.664 다. k-means 가 만드는 것은 체류 중심
주변의 **볼록 셀**인데, 실제 방은 벽과 문으로 갈리고 문간은 **가는 목**이다.
볼록 셀은 목을 자를 수 없어 방을 가로질러 뭉갠다.

여기서는 방을 이렇게 정의한다:

    ① 궤적을 바닥면 점유격자에 굽는다(셀 크기 `--cell` m)
    ② 점유 셀을 **침식**해 가는 목(문간)을 끊는다(`--erode` 회)
    ③ 남은 덩어리의 **연결성분** = 방 씨앗
    ④ 끊긴 셀을 가장 가까운 씨앗에 되돌려 붙인다(팽창)

k 를 지정하지 않는다 — 방 개수는 **문간이 몇 개를 끊는가**로 정해진다.
평가는 `room_naming_probe.py` 와 같은 지표(군집→GT최빈방 상한)로 직접 비교한다.
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from scripts.supermem_rooms3d import load_traj, gravity, floor_basis   # noqa: E402
from scripts.room_naming_probe import norm_room, SESS5                 # noqa: E402


def segment_dwell(uv, cell, pct, sigma, min_cells):
    """**체류 시간**으로 방을 정의한다 — 방은 머무는 곳, 문간은 지나가는 곳.

    순수 형태학(팽창→침식)은 실패했다: 궤적 점유는 얇은 리본이라 침식하면 방이
    아니라 전부 끊긴다(실측: s8 이 통째로 지워지고 s1 은 1개 방으로 뭉갬).
    대신 셀별 **체류 프레임 수**를 세고 상위 분위만 남긴다. 문간은 빠르게 지나가
    체류가 낮으므로 자연히 끊긴다.
    """
    from scipy import ndimage
    lo = uv.min(0) - cell
    g = np.floor((uv - lo) / cell).astype(int)
    H, W = g[:, 1].max() + 2, g[:, 0].max() + 2
    dw = np.zeros((H, W), float)
    np.add.at(dw, (g[:, 1], g[:, 0]), 1.0)
    if sigma > 0:
        dw = ndimage.gaussian_filter(dw, sigma)
    occ = dw > 0
    thr = np.percentile(dw[occ], pct) if occ.any() else 0
    core = dw >= thr
    lab, n = ndimage.label(core)
    for i in range(1, n + 1):
        if (lab == i).sum() < min_cells:
            lab[lab == i] = 0
    ids = [i for i in np.unique(lab) if i]
    if not ids:
        return None, 0
    remap = {v: i for i, v in enumerate(ids)}
    _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    full = np.where(lab > 0, lab, lab[iy, ix])

    def label_of(pts):
        q = np.floor((pts - lo) / cell).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        return np.array([remap.get(int(x), -1) for x in full[q[:, 1], q[:, 0]]])

    return label_of, len(ids)


def segment(uv, cell, erode, min_cells, dilate=3):
    """점유격자 → 팽창 → 침식 → 연결성분. (형태학판 — 실측 실패, 비교용으로 남김)"""
    from scipy import ndimage
    lo = uv.min(0) - cell
    g = np.floor((uv - lo) / cell).astype(int)
    H, W = g[:, 1].max() + 2, g[:, 0].max() + 2
    occ = np.zeros((H, W), bool)
    occ[g[:, 1], g[:, 0]] = True
    # 궤적은 **얇은 리본**이다 — 그대로 침식하면 방이 아니라 전부 끊긴다.
    # 먼저 팽창해 사람이 지나다닌 영역을 방 모양 덩어리로 메운 뒤(dilate),
    # 그만큼 되깎으면서(erode) 가는 목(문간)만 끊는다.
    er = occ.copy()
    for _ in range(dilate):
        er = ndimage.binary_dilation(er)
    for _ in range(dilate + erode):
        er = ndimage.binary_erosion(er)
    lab, n = ndimage.label(er)
    # 너무 작은 성분은 방으로 치지 않는다(잡음 제거)
    for i in range(1, n + 1):
        if (lab == i).sum() < min_cells:
            lab[lab == i] = 0
    ids = [i for i in np.unique(lab) if i]
    if not ids:
        return None, 0
    remap = {v: i for i, v in enumerate(ids)}
    # 침식으로 지워진 점유 셀을 가장 가까운 씨앗으로 되돌린다
    _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    full = np.where(lab > 0, lab, lab[iy, ix])

    def label_of(pts):
        q = np.floor((pts - lo) / cell).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        v = full[q[:, 1], q[:, 0]]
        return np.array([remap.get(int(x), -1) for x in v])

    return label_of, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", nargs="+", default=["s1", "s8", "s14"])
    ap.add_argument("--cell", type=float, default=0.30, help="격자 셀 m")
    ap.add_argument("--dilate", type=int, default=3, help="선팽창 — 궤적 리본을 방 덩어리로")
    ap.add_argument("--erode", type=int, default=2, help="추가 침식(문간 끊기)")
    ap.add_argument("--min-cells", type=int, default=8, help="방으로 칠 최소 셀 수")
    ap.add_argument("--mode", default="dwell", choices=["dwell", "morph"])
    ap.add_argument("--pct", type=float, default=60.0, help="체류 분위 문턱(%)")
    ap.add_argument("--sigma", type=float, default=1.0, help="체류격자 평활")
    ap.add_argument("--modality", default="Video")
    ap.add_argument("--out", default=None,
                    help="방 배정을 rooms3d 형식으로 저장 — belief 가 그대로 읽는다")
    ap.add_argument("--sweep", action="store_true",
                    help="한 프로세스 안에서 문턱을 쓴다 — 궤적 CSV 재적재를 피한다"
                         "(설정마다 재실행하면 설정당 4분, 스윕 12개면 48분)")
    args = ap.parse_args()

    qs = json.load(open(os.path.join(D, "qa_person_1.json")))
    ev = []
    for x in qs:
        for e in ((x.get("answer_evidence") or {}).get("evidence_list") or []):
            g = norm_room(e.get("room"))
            sd = SESS5.get(e.get("video_id"))
            t = (e.get("time_span") or {}).get("start_time")
            if g and sd and t is not None:
                if args.modality and args.modality not in (e.get("modalities") or []):
                    continue
                ev.append((sd, float(t), g))

    # 궤적은 한 번만 읽는다
    traj = {}
    for sd in args.sess:
        f = os.path.join(D, sd, "closed_loop_trajectory.csv")
        if not os.path.exists(f):
            continue
        sec, P = load_traj(sd)
        gv = gravity(P); e1, e2 = floor_basis(gv)
        traj[sd] = (sec, np.stack([P @ e1, P @ e2], 1))

    if args.sweep:
        print("%-6s %-6s %-24s %-24s %s" % ("셀", "분위", "s1", "s8", "합계상한"))
        for cell in (0.30, 0.50, 0.80):
            for pct in (40, 55, 70, 80, 88, 94):
                row, tn, th, ag = [], 0, 0, []
                for sd in ("s1", "s8"):
                    pts = [(t, g) for s_, t, g in ev if s_ == sd]
                    if sd not in traj or len(pts) < 10:
                        row.append("%-24s" % "-")
                        continue
                    sec, uv = traj[sd]
                    lf, nr = segment_dwell(uv, cell, pct, args.sigma, args.min_cells)
                    if lf is None:
                        row.append("%-24s" % "방 없음")
                        continue
                    X = np.array([[np.interp(t, sec, uv[:, 0]),
                                   np.interp(t, sec, uv[:, 1])] for t, _ in pts])
                    y = [g for _, g in pts]
                    cl = {}
                    for l, g in zip(lf(X), y):
                        cl.setdefault(int(l), Counter())[g] += 1
                    h = sum(c.most_common(1)[0][1] for c in cl.values())
                    tn += len(y); th += h; ag += y
                    row.append("%-24s" % ("방%-2d 상한 %.3f (최빈 %.2f)"
                               % (nr, h / len(y),
                                  Counter(y).most_common(1)[0][1] / len(y))))
                print("%-6.2f %-6d %s %s %s"
                      % (cell, pct, row[0], row[1] if len(row) > 1 else "",
                         "**%.3f**" % (th / tn) if tn else "-"))
        print("\n(대조: k-means 방수준 k=3 상한 0.664 · 세밀 k=12(26군집) 0.745")
        print(" · 위치 1-NN s1 0.784 / s8 0.862 = 정보 상한)")
        return

    if args.out:
        # belief 가 읽는 형식(rooms3d.json)으로 초 단위 방 배정을 쓴다
        out = {}
        for sd, (sec, uv) in traj.items():
            lf, nr = (segment_dwell(uv, args.cell, args.pct, args.sigma, args.min_cells)
                      if args.mode == "dwell"
                      else segment(uv, args.cell, args.erode, args.min_cells, args.dilate))
            if lf is None:
                print("  %s 방 없음 — 건너뜀" % sd)
                continue
            fr = np.arange(0, int(sec.max()) + 1)
            fu = np.stack([np.interp(fr, sec, uv[:, 0]),
                           np.interp(fr, sec, uv[:, 1])], 1)
            lab = lf(fu)
            lab[lab < 0] = 0
            out[sd] = dict(centers=[[0.0, 0.0]] * nr, frame_room=lab.tolist(),
                           spread_m=float(np.linalg.norm(uv.max(0) - uv.min(0))),
                           dwell=dict(Counter(int(x) for x in lab)))
            print("  %-4s 방 %d개 · 프레임 %d · 체류 %s"
                  % (sd, nr, len(lab), dict(Counter(int(x) for x in lab))))
        json.dump(out, open(os.path.join(D, args.out), "w"))
        print("→ %s" % os.path.join(D, args.out))
        return

    tot_n = tot_hit = 0
    allg = []
    print("[%s] 셀 %.2f m · %s · 최소 %d셀"
          % (args.mode, args.cell,
             ("체류분위 %.0f%% · 평활 %.1f" % (args.pct, args.sigma))
             if args.mode == "dwell" else
             ("팽창 %d · 침식 %d" % (args.dilate, args.erode)), args.min_cells))
    for sd in args.sess:
        pts = [(t, g) for s_, t, g in ev if s_ == sd]
        if len(pts) < 10:
            print("  %-4s 근거 %d건 — 표본 부족" % (sd, len(pts)))
            continue
        sec, uv = traj[sd]
        if args.mode == "dwell":
            lf, nr = segment_dwell(uv, args.cell, args.pct, args.sigma, args.min_cells)
        else:
            lf, nr = segment(uv, args.cell, args.erode, args.min_cells, args.dilate)
        if lf is None:
            print("  %-4s 방이 안 남았다 — 문턱 조정 필요" % sd)
            continue
        X = np.array([[np.interp(t, sec, uv[:, 0]), np.interp(t, sec, uv[:, 1])]
                      for t, _ in pts])
        y = [g for _, g in pts]
        lab = lf(X)
        cl = {}
        for l, g in zip(lab, y):
            cl.setdefault(int(l), Counter())[g] = cl.setdefault(int(l), Counter())[g] + 1
        hit = sum(c.most_common(1)[0][1] for c in cl.values())
        mj = Counter(y).most_common(1)[0]
        tot_n += len(y); tot_hit += hit; allg += y
        print("  %-4s 방 %-2d개 · 근거 %-4d 상한 **%.3f** (최빈 %s %.3f) 순도 %s"
              % (sd, nr, len(y), hit / len(y), mj[0], mj[1] / len(y),
                 " ".join("%.2f" % (c.most_common(1)[0][1] / sum(c.values()))
                          for _, c in sorted(cl.items()))))
    if tot_n:
        mj = Counter(allg).most_common(1)[0]
        print("\n합계 근거 %d · **연결성분 상한 %.3f** · 최빈방 %.3f"
              % (tot_n, tot_hit / tot_n, mj[1] / len(allg)))
        print("(대조: k-means k=3 상한 0.664 · 위치 1-NN s1 0.784 / s8 0.862)")


if __name__ == "__main__":
    main()
