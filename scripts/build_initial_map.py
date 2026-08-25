#!/usr/bin/env python3
"""**초기 맵을 관측으로 구축한다** — 매핑워크 영상 + 뎁스 + 포즈 → 정적 물체 지도.

    python scripts/build_initial_map.py --root data/thor3 --mode gtbox   # 기하 검증
    python scripts/build_initial_map.py --root data/thor3 --mode owl     # 실전 (검출)

지금까지 모든 수치(정지 지도 0.672, 앵커 방 배정표)가 **GT 맵** 위에 서 있었다.
여기서 처음으로 맵을 관측에서 만든다. 두 눈금:

  gtbox  GT bbox + GT 뎁스 + GT 포즈 → 기하 사슬만 검증 (역투영 오차 보고)
  owl    OWL 검출 + GT 뎁스 + GT 포즈 → 검출이 초기 맵을 얼마나 망치나

산출: <house>/initmap_<mode>.json — 정적 개체 목록 {type, pos, room} 와
방별 타입표. 이후 eval 이 scene_meta 대신 이걸 읽으면 GT 맵이 벗겨진다.

⚠️ AI2-THOR 뎁스는 **평면 z-버퍼**(카메라 평면까지의 미터)다. 광선 길이가 아니다.
카메라: fov 90° → f = (W/2)/tan(45°) = W/2 · horizon 10°(아래) · 높이 ≈ 1.576 m.
"""
import argparse, glob, json, os
import numpy as np

CAMY = 1.576


def rays(W, f):
    u = np.arange(W) + .5 - W/2
    return u / f


def backproject(px_, py_, d, pos, yaw, W, pitch=10.0):
    """화면 (px,py) + 평면뎁스 d → 월드 (x, z). y 는 방 판정에 불필요."""
    f = W / 2.0
    xc = (px_ + .5 - W/2) / f * d
    yc = (py_ + .5 - W/2) / f * d          # 아래가 +
    zc = d
    p = np.radians(pitch)                   # horizon 10 = 카메라가 10° 아래를 본다
    y1 = yc*np.cos(p) - zc*np.sin(p)
    z1 = yc*np.sin(p) + zc*np.cos(p)
    yw = np.radians(yaw)
    xw = xc*np.cos(yw) + z1*np.sin(yw)
    zw = -xc*np.sin(yw) + z1*np.cos(yw)
    return pos[0] + xw, pos[1] + zw


def inside(x, z, pts):
    c = False; n = len(pts)
    for i in range(n):
        x1, z1 = pts[i]; x2, z2 = pts[(i+1) % n]
        if (z1 > z) != (z2 > z) and x < (x2-x1)*(z-z1)/(z2-z1+1e-12) + x1:
            c = not c
    return c


def cluster(pts, r=0.7):
    """탐욕 군집 — 같은 개체의 여러 관측을 하나로."""
    out = []
    for x, z, w in sorted(pts, key=lambda p: -p[2]):
        for c in out:
            if (c[0]-x)**2 + (c[1]-z)**2 < r*r:
                n = c[2] + w
                c[0] = (c[0]*c[2] + x*w)/n; c[1] = (c[1]*c[2] + z*w)/n; c[2] = n
                break
        else:
            out.append([x, z, w])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", default="gtbox", choices=["gtbox", "owl"])
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--th", type=float, default=0.25, help="owl 검출 문턱")
    args = ap.parse_args()

    if args.mode == "owl":
        import torch
        from PIL import Image
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        DEV = ("cuda" if torch.cuda.is_available()
               else "mps" if torch.backends.mps.is_available() else "cpu")
        CK = "google/owlv2-base-patch16-ensemble"
        pr = Owlv2Processor.from_pretrained(CK)
        md = Owlv2ForObjectDetection.from_pretrained(CK).to(DEV).eval()
        stat = json.load(open("data/thor_static_types.json"))
        def words(t): return "".join(" "+c.lower() if c.isupper() else c for c in t).strip()
        ti = pr(text=[["a photo of a " + words(t) for t in stat]],
                images=[Image.new("RGB", (256, 256), (128,)*3)], return_tensors="pt").to(DEV)
        with torch.no_grad():
            o = md.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                         pixel_values=ti["pixel_values"], return_dict=True)
        TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)

    errs_all = []
    for hd in sorted(glob.glob(os.path.join(args.root, "house_*"))):
        rd = os.path.realpath(hd)
        wf = os.path.join(rd, "mapwalk", "walk.json")
        if not os.path.exists(wf): continue
        w = json.load(open(wf)); W = w["size"]
        g = json.load(open(os.path.join(rd, "gt.json"))); sm = g["scene_meta"]
        polys = {r: [(c[0], c[1]) for c in v] for r, v in sm["polys"].items()}
        obs = {}          # type → [(x, z, w)]
        errs = []
        frames = w["frames"][::args.stride]
        for fr in frames:
            dp = os.path.join(rd, "mapwalk", "depth", "%05d.npy" % fr["k"])
            if not os.path.exists(dp): continue
            D = np.load(dp).astype(np.float32)
            if args.mode == "gtbox":
                for oid, b in fr.get("box", {}).items():
                    if oid not in sm["static"]: continue
                    cx, cy = (b[0]+b[2])//2, (b[1]+b[3])//2
                    d = float(np.median(D[max(0,cy-2):cy+3, max(0,cx-2):cx+3]))
                    if not (0.3 < d < 12): continue
                    x, z = backproject(cx, cy, d, fr["pos"], fr["yaw"], W)
                    t = sm["static"][oid]["type"]
                    obs.setdefault(t, []).append((x, z, 1.0))
                    if "pos" in sm["static"][oid]:
                        gx, gz = sm["static"][oid]["pos"]
                        errs.append(float(np.hypot(x-gx, z-gz)))
            else:
                im = Image.open(os.path.join(rd, "mapwalk", "%05d.jpg" % fr["k"])).convert("RGB")
                pv = pr(images=[im], return_tensors="pt")["pixel_values"].to(DEV)
                with torch.no_grad():
                    fm = md.image_embedder(pixel_values=pv)[0]
                    b_, ph, pw, hdim = fm.shape
                    lg, _ = md.class_predictor(fm.reshape(1, ph*pw, hdim),
                                               TX.unsqueeze(0), MK.unsqueeze(0))
                    sc = torch.sigmoid(lg)[0].float().cpu().numpy()   # (패치, 타입)
                hit = np.argwhere(sc >= args.th)
                for pi, ci_ in hit:
                    cy = (pi // pw + .5) / ph * W; cx = (pi % pw + .5) / pw * W
                    d = float(np.median(D[max(0,int(cy)-2):int(cy)+3,
                                          max(0,int(cx)-2):int(cx)+3]))
                    if not (0.3 < d < 12): continue
                    x, z = backproject(cx, cy, d, fr["pos"], fr["yaw"], W)
                    obs.setdefault(stat[ci_], []).append((x, z, float(sc[pi, ci_])))
        inst = []
        for t, pts in obs.items():
            for x, z, wt in cluster(pts):
                if wt < (2.0 if args.mode == "gtbox" else 1.0): continue
                room = next((r for r, pp in polys.items() if inside(x, z, pp)), None)
                if room:
                    inst.append(dict(type=t, pos=[round(x, 2), round(z, 2)],
                                     room=room, w=round(float(wt), 2)))
        json.dump(inst, open(os.path.join(rd, "initmap_%s.json" % args.mode), "w"))
        # ── 채점: GT 정적 지도 대비 방별 타입표 정밀도/재현율 ──
        gtset = {(v["room"], v["type"]) for v in sm["static"].values()}
        dset = {(i["room"], i["type"]) for i in inst}
        pr_ = len(gtset & dset) / max(len(dset), 1)
        rc_ = len(gtset & dset) / max(len(gtset), 1)
        msg = "  %s · 개체 %d · (방,타입) 정밀 %.2f 재현 %.2f" \
              % (os.path.basename(hd), len(inst), pr_, rc_)
        if errs:
            errs_all += errs
            msg += " · 역투영 오차 중앙 %.2fm" % np.median(errs)
        print(msg, flush=True)
    if errs_all:
        print("전체 역투영 오차 중앙 %.2fm · 90분위 %.2fm"
              % (np.median(errs_all), np.quantile(errs_all, .9)))
    print("완료")


if __name__ == "__main__":
    main()
