#!/usr/bin/env python3
"""SfM 포즈(seq/pose/poses_da3*.txt, c2w 4x4/행) vs HSSD GT — sim3 정렬 ATE·yaw 오차 + 평가기 입력 내보내기.

    python scripts/pose_eval_hssd.py data/seq/hs2_house_0000 data/hssd20S2/house_0000 \\
        --pose pose/poses_da3.txt --out pose_house_0000.jsonl

정렬: GT 위치(apos, y=1.5)와 SfM 위치를 Umeyama sim3(스케일 포함)로 맞춘다 — 단안 SfM 은 스케일·
전역 좌표계가 자유이므로 이 정렬은 정당(배포에선 초기 맵 앵커가 이 역할). 정렬 뒤:
  ATE 중앙/평균(m) · yaw 오차 중앙(°) · 커버리지(포즈 있는 프레임 비율)
출력 jsonl: {house, t, apos:[x,z], yaw} (우리 규약: bearing=atan2(dx,dz), 0°=+z, 시계 증가)
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kx.depth.pose_stitch import umeyama  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("seq"); ap.add_argument("house")
ap.add_argument("--pose", default="pose/poses_da3.txt")
ap.add_argument("--out", default=None)
a = ap.parse_args()
g = json.load(open(os.path.join(a.house, "gt.json"))); live = {m["t"]: m for m in g["live"]}
frames = sorted(int(f[:-4]) for f in os.listdir(os.path.join(a.seq, "rgb")) if f.endswith(".jpg"))
P = np.loadtxt(os.path.join(a.seq, a.pose))
P = P.reshape(-1, 4, 4) if P.ndim == 2 and P.shape[1] == 16 else P.reshape(-1, 4, 4)
n = min(len(P), len(frames))
ok = np.array([np.isfinite(P[i]).all() and abs(np.linalg.det(P[i][:3, :3]) - 1) < 0.2 for i in range(n)])
gt_c = np.array([[live[frames[i]]["apos"][0], 1.5, live[frames[i]]["apos"][1]] for i in range(n)])
est_c = P[:n, :3, 3]
# sim3: est → gt
s, R, t = umeyama(est_c[ok], gt_c[ok])
al = (s * (R @ est_c.T)).T + t
ate = np.linalg.norm(al - gt_c, axis=1)[ok]
# yaw: SfM 카메라 +z 전방(어댑터 규약) → 월드 방위. 정렬 회전 R 적용
fwd = np.einsum("ij,njk->nik", R, P[:n, :3, :3])[:, :, 2]          # 회전된 z축
yaw_est = np.degrees(np.arctan2(fwd[:, 0], fwd[:, 2])) % 360
yaw_gt = np.array([live[frames[i]]["yaw"] for i in range(n)])
yerr = np.abs((yaw_est - yaw_gt + 180) % 360 - 180)[ok]
print("프레임 %d · 포즈 있음 %.2f · sim3 스케일 %.3f" % (n, ok.mean(), s))
print("ATE 중앙 %.3f m · 평균 %.3f m · <0.5m %.2f | yaw 오차 중앙 %.1f° · <10° %.2f" % (
    np.median(ate), ate.mean(), (ate < 0.5).mean(), np.median(yerr), (yerr < 10).mean()))
if a.out:
    hn = os.path.basename(os.path.realpath(a.house))
    with open(a.out, "w") as f:
        for i in range(n):
            if not ok[i]: continue
            f.write(json.dumps(dict(house=hn, t=int(frames[i]), apos=[round(float(al[i, 0]), 3), round(float(al[i, 2]), 3)],
                                    yaw=round(float(yaw_est[i]), 2))) + "\n")
    print("→", a.out)
