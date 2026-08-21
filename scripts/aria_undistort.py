#!/usr/bin/env python3
"""Aria 어안(fisheye624) → 선형 보정 — **크롭이 아니라 재투영**.

크롭 실험으로는 어안의 영향을 잴 수 없다(2026-08-21 실측·사용자 지적):

| 크롭 | top-1 | GT bbox 소실 |
|---|---|---|
| 원본 | 0.463 | — |
| 70% | 0.400 | 5% |
| 55% | 0.395 | **14%** |

크롭은 **두 가지를 동시에** 한다 — 검은 모서리·강한 왜곡을 걷어내는 이득과,
**화각 축소 + GT 증거 소실**이라는 손해. 섞여 있어 왜곡 자체의 영향은 못 잰다.
55% 에서는 소실률(14%)과 하락폭(14.7%)이 거의 같아 GT 불일치로 설명된다.

**재투영은 내용을 보존한 채 왜곡만 편다** — 화각도 GT 도 안 잃는다.
`kx/adt/calib.py` 가 ADT 에 쓰던 것과 같은 경로다.

⚠️ **캘리브 출처**: HD-EPIC 은 개체별 캘리브를 안 받았다(SLAM zip 이 1.1 GB).
대신 **Nymeria 의 `online_calibration.jsonl`**(같은 Aria 모델)에서 fisheye624
파라미터를 가져온다. 개체가 다르므로 **근사**다 — 렌즈 모델과 공칭 초점거리가
같으므로 왜곡 성분은 거의 일치하지만, 개체 편차만큼 오차가 남는다.
"""
import argparse, glob, json, os, sys

import numpy as np


def rgb_fisheye_params(jsonl):
    """online_calibration.jsonl 에서 camera-rgb 의 fisheye624 파라미터를 꺼낸다."""
    r = json.loads(open(jsonl).readline())
    for c in r.get("CameraCalibrations", []):
        if c.get("Label", "").lower() == "camera-rgb":
            return np.array(c["Projection"]["Params"], float)
    return None


def build_maps(P, W, H, out_size, out_focal):
    """fisheye624 → 선형 리샘플 맵. OpenCV `remap` 용 (mx, my).

    ⚠️ **캘리브 해상도와 영상 해상도가 다르면 스케일해야 한다.** Nymeria 캘리브는
    2880×2880 기준(cx≈1455)인데 HD-EPIC MP4 는 1408×1408 이다. 그대로 쓰면 맵이
    원본 밖으로 나가 23% 만 유효했다.
    """
    src_w = 2.0 * P[1]                      # cx 로부터 원본 폭을 추정
    sc = W / src_w
    f, cx, cy = P[0] * sc, P[1] * sc, P[2] * sc
    k = P[3:9]          # 방사 k1..k6
    p = P[9:11]         # 접선
    s = P[11:15]        # thin prism
    u = (np.arange(out_size) - (out_size - 1) / 2.0) / out_focal
    xx, yy = np.meshgrid(u, u)
    r = np.sqrt(xx ** 2 + yy ** 2)
    th = np.arctan(r)                       # 핀홀 방향 → 입사각
    th2 = th * th
    # 방사 다항식 (Aria fisheye624)
    rad = th * (1 + k[0] * th2 + k[1] * th2**2 + k[2] * th2**3 +
                k[3] * th2**4 + k[4] * th2**5 + k[5] * th2**6)
    scale = np.where(r > 1e-9, rad / np.maximum(r, 1e-9), 1.0)
    xd, yd = xx * scale, yy * scale
    # 접선 + thin prism
    r2 = xd**2 + yd**2
    xd2 = xd + (2 * p[0] * xd * yd + p[1] * (r2 + 2 * xd**2)) + s[0] * r2 + s[1] * r2**2
    yd2 = yd + (p[0] * (r2 + 2 * yd**2) + 2 * p[1] * xd * yd) + s[2] * r2 + s[3] * r2**2
    mx = (f * xd2 + cx).astype(np.float32)
    my = (f * yd2 + cy).astype(np.float32)
    return mx, my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None, help="online_calibration.jsonl")
    ap.add_argument("--size", type=int, default=704)
    ap.add_argument("--focal", type=float, default=350.0)
    ap.add_argument("--test", default=None, help="시험용 mp4")
    args = ap.parse_args()

    cal = args.calib
    if not cal:
        c = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/nymeria/locbx/*/recording_head/mps/slam/online_calibration.jsonl")))
        cal = c[0] if c else None
    if not cal:
        print("캘리브 파일을 못 찾았다"); return
    P = rgb_fisheye_params(cal)
    print("fisheye624 파라미터 %d개 · f=%.1f cx=%.1f cy=%.1f" % (len(P), P[0], P[1], P[2]))
    mx, my = build_maps(P, 1408, 1408, args.size, args.focal)
    print("리샘플 맵 %dx%d · 원본 좌표 범위 x[%.0f,%.0f] y[%.0f,%.0f]"
          % (args.size, args.size, mx.min(), mx.max(), my.min(), my.max()))
    inb = ((mx >= 0) & (mx < 1408) & (my >= 0) & (my < 1408)).mean()
    print("맵이 원본 안에 드는 비율 **%.1f%%** (낮으면 화각을 넘겨 잡은 것)" % (100 * inb))
    if args.test:
        import cv2
        cap = cv2.VideoCapture(args.test)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 3000)
        ok, img = cap.read(); cap.release()
        if ok:
            out = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
            d = os.path.dirname(args.test)
            cv2.imwrite("/private/tmp/undist.jpg", out)
            print("→ /private/tmp/undist.jpg")


if __name__ == "__main__":
    main()
