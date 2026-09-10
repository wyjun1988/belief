#!/usr/bin/env python3
"""새 시뮬레이터 프레임 전처리 실험(2026-09-10): 점 잡음·과노출이 검출을 죽이는지 확인용. gt.json 은 그대로(박스 좌표 유지: 원 해상도로 되돌린다).
  python scripts/preprocess_frames.py <src_root> <dst_root> --mode bilateral|nlm|down2|gamma|gamma_bilateral [--sub map,live] [--houses h1,h2]"""
import os, sys, glob, argparse, cv2, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("dst"); ap.add_argument("--mode", default="bilateral"); ap.add_argument("--sub", default="map,live"); ap.add_argument("--houses", default="")
ap.add_argument("--gamma", type=float, default=1.5); ap.add_argument("--gain", type=float, default=0.85); a = ap.parse_args()
def proc(im):
    m = a.mode
    if m.startswith("gamma"):
        f = np.clip((im.astype(np.float32) / 255.0 * a.gain) ** a.gamma, 0, 1); im = (f * 255).astype(np.uint8)
        if m == "gamma": return im
        m = m.split("_", 1)[1]
    if m == "bilateral": return cv2.bilateralFilter(im, 7, 50, 7)
    if m == "nlm": return cv2.fastNlMeansDenoisingColored(im, None, 7, 7, 7, 21)
    if m == "down2":
        h, w = im.shape[:2]; s = cv2.resize(im, (w // 2, h // 2), interpolation=cv2.INTER_AREA); return cv2.resize(s, (w, h), interpolation=cv2.INTER_CUBIC)
    if m == "median": return cv2.medianBlur(im, 3)
    raise SystemExit("unknown mode " + a.mode)
hs = [os.path.join(a.src, h) for h in a.houses.split(",")] if a.houses else sorted(glob.glob(os.path.join(a.src, "house_*")))
n = 0
for hd in hs:
    hn = os.path.basename(hd); od = os.path.join(a.dst, hn); os.makedirs(od, exist_ok=True)
    for f in ("gt.json", "initmap_gt.json", "room_groups.json", "phantom_ids.json"):
        sp = os.path.join(os.path.realpath(hd), f)
        if os.path.exists(sp) and not os.path.exists(os.path.join(od, f)): os.symlink(sp, os.path.join(od, f))
    for sub in a.sub.split(","):
        sd = os.path.join(os.path.realpath(hd), sub); dd = os.path.join(od, sub); os.makedirs(dd, exist_ok=True)
        for p in sorted(glob.glob(os.path.join(sd, "*.jpg"))):
            q = os.path.join(dd, os.path.basename(p))
            if os.path.exists(q): continue
            im = cv2.imread(p); cv2.imwrite(q, proc(im), [cv2.IMWRITE_JPEG_QUALITY, 92]); n += 1
    print("%s: %s 처리" % (hn, a.sub), flush=True)
print("PREPROCESS_DONE %d장 (%s)" % (n, a.mode))
