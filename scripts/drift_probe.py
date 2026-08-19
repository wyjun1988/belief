#!/usr/bin/env python3
"""궤적 드리프트를 **직접** 잰다 — 재방문 쌍의 위치 불일치.

    $P scripts/drift_probe.py --sess s1 s8

㉑ 에서 쓴 지표("같은 방인데 시간이 멀면 멀어진다")에는 **결함이 있다.** 방은
넓다 — 부엌이 3~4 m 면 한 시간 뒤 다른 자리에 서기만 해도 2 m 가 나온다.
방 크기와 드리프트가 섞여 있어 그것만으로는 포즈 문제라고 말할 수 없다.

드리프트를 분리해 재려면 **같은 자리로 돌아온 순간**을 잡아야 한다:

    ① CLIP 임베딩이 매우 비슷하고 시간이 충분히 떨어진 프레임 쌍을 후보로
    ② 두 프레임에서 SIFT 특징을 매칭하고 **호모그래피 인라이어**로 같은 장소 확인
       (CLIP 만으로는 "비슷한 부엌"과 "같은 자리"를 못 가른다)
    ③ 검증된 쌍의 **MPS 위치 거리**를 본다 — 같은 자리인데 멀면 그게 드리프트다

드리프트가 없으면 검증된 재방문 쌍의 위치 거리는 작아야 한다(< ~0.5 m).
"""
import argparse, os, subprocess, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.supermem_rooms3d import D, load_traj, gravity, floor_basis   # noqa


def grab(sd, t, out):
    """video.mp4 의 t 초 프레임을 뽑는다."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(float(t)), "-i",
                    os.path.join(D, sd, "video.mp4"), "-frames:v", "1", "-y", out],
                   check=True)
    return out


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", nargs="+", default=["s1", "s8"])
    ap.add_argument("--min-gap", type=float, default=600.0, help="후보 최소 시간차(초)")
    ap.add_argument("--cand", type=int, default=60, help="세션당 검사할 후보 쌍")
    ap.add_argument("--min-inliers", type=int, default=25,
                    help="같은 장소로 인정할 호모그래피 인라이어 수")
    args = ap.parse_args()

    sift = cv2.SIFT_create(nfeatures=2000)
    bf = cv2.BFMatcher()
    tmp = tempfile.mkdtemp()

    for sd in args.sess:
        z = np.load(os.path.join(D, sd, "index.npz"))
        E = z["emb"].astype(np.float32); ts = z["ts"].astype(float)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        sec, P = load_traj(sd)
        g = gravity(P); e1, e2 = floor_basis(g)
        uv = np.stack([P @ e1, P @ e2], 1)
        XY = np.stack([np.interp(ts, sec, uv[:, 0]), np.interp(ts, sec, uv[:, 1])], 1)

        S = E @ E.T
        gap = np.abs(ts[:, None] - ts[None]) >= args.min_gap
        S[~gap] = -1
        iu = np.triu_indices(len(E), 1)
        order = np.argsort(-S[iu])[:args.cand * 4]
        seen, pairs = set(), []
        for o in order:
            i, j = iu[0][o], iu[1][o]
            if S[i, j] < 0 or i in seen or j in seen:
                continue
            seen.add(i); seen.add(j)
            pairs.append((int(i), int(j), float(S[i, j])))
            if len(pairs) >= args.cand:
                break

        ok = []
        for i, j, s in pairs:
            try:
                a = cv2.imread(grab(sd, ts[i], os.path.join(tmp, "a.jpg")), 0)
                b = cv2.imread(grab(sd, ts[j], os.path.join(tmp, "b.jpg")), 0)
            except Exception:
                continue
            if a is None or b is None:
                continue
            ka, da = sift.detectAndCompute(a, None)
            kb, db = sift.detectAndCompute(b, None)
            if da is None or db is None or len(ka) < 10 or len(kb) < 10:
                continue
            m = bf.knnMatch(da, db, k=2)
            good = [x for x, y in m if len(m[0]) == 2 and x.distance < 0.75 * y.distance]
            if len(good) < args.min_inliers:
                continue
            src = np.float32([ka[x.queryIdx].pt for x in good]).reshape(-1, 1, 2)
            dst = np.float32([kb[x.trainIdx].pt for x in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                continue
            inl = int(mask.sum())
            if inl < args.min_inliers:
                continue
            d = float(np.linalg.norm(XY[i] - XY[j]))
            ok.append((d, inl, float(abs(ts[i] - ts[j])), s))

        print("\n[%s] 후보 %d → **기하 검증 통과 %d 쌍**" % (sd, len(pairs), len(ok)))
        if not ok:
            print("  같은 자리로 돌아온 증거가 없다 — 드리프트 판정 불가")
            continue
        ds = np.array([x[0] for x in ok])
        print("  재방문 쌍 위치 거리: 중앙 **%.2f m** · p90 %.2f · 최대 %.2f"
              % (np.median(ds), np.percentile(ds, 90), ds.max()))
        print("  (인라이어 중앙 %d · 시간차 중앙 %.0f분)"
              % (int(np.median([x[1] for x in ok])),
                 np.median([x[2] for x in ok]) / 60))
        for d, inl, dt, s in sorted(ok)[-5:]:
            print("    거리 %.2f m · 인라이어 %-4d · 시간차 %.0f분 · CLIP %.3f"
                  % (d, inl, dt / 60, s))
    print("\n→ 같은 자리(기하 검증)인데 거리가 크면 **드리프트**다.")
    print("  거리가 작으면 ㉑ 의 '방 내 거리 증가'는 드리프트가 아니라 **방 크기**다.")


if __name__ == "__main__":
    main()
