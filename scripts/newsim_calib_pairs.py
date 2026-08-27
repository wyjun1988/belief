#!/usr/bin/env python3
"""새 시뮬 에피소드 → 검증기 현지 캘리브레이션 쌍 자동 생성.

    python scripts/newsim_calib_pairs.py --ep <dir> --out /tmp/nscalib [--n 200]

ego 스트림이 "봤다" 고 기록한 (물체, 시각)에서 **AABB 투영 bbox** 로 크롭을 떠
양성으로, 같은 크롭에 다른 클래스 라벨을 물어 음성으로 만든다.
exp_vlm_verify3(H100 로짓 채점)와 같은 meta.jsonl 스키마.

⚠️ 목적: §89 교훈 — 검증 문턱은 도메인마다 로짓 분포가 밀리므로, 새 에피소드가
오면 이 쌍으로 H100 1회 → 로컬 스윕으로 운용점을 다시 잡는다.
"""
import argparse, io, json, os, subprocess, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from newsim_project import project_aabb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sg = json.load(open(os.path.join(a.ep, "scene_graph.json")))
    obj = {o["id"]: o for o in sg["objects"]}
    # 목격 오라클 = **AABB 투영** (ego 델타 이벤트는 16개뿐이라 불충분).
    # 조건: bbox 24px+ · 중심 5m 이내. ⚠️ 가림 미처리 — 가려진 양성이 섞여
    # 수용률이 실제보다 낮게 측정될 수 있다(보수 편향, 캘리브레이션 용도로 허용).
    from newsim_project import project
    cams = {}
    for line in open(os.path.join(a.ep, "observed_graph_updates.jsonl")):
        d = json.loads(line)
        c = next((c for c in d.get("cameras", []) if c.get("source") == "ego"), None)
        if c: cams[int(d["time_s"])] = c
    events = {}
    nonstruct = [o for o in sg["objects"] if not o.get("is_structure")
                 and o.get("aabb_world")]
    for sec, cam in cams.items():
        for o in nonstruct:
            bb = project_aabb(o["aabb_world"], cam)
            if bb is None or min(bb[2]-bb[0], bb[3]-bb[1]) < 24: continue
            pr = project(o["transform"]["location"], cam)
            if pr is None or pr[2] > 500: continue
            events[(o["id"], sec)] = cam
    print("목격(투영) 이벤트 %d" % len(events), flush=True)
    # 1fps 프레임
    fr_dir = os.path.join(a.out, "_frames")
    if not os.path.exists(fr_dir):
        os.makedirs(fr_dir)
        subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(a.ep, "ego_left.mp4"),
                        "-vf", "fps=1", "-q:v", "3", os.path.join(fr_dir, "%04d.jpg")],
                       check=True)
    rng = np.random.default_rng(0)
    keys = list(events.keys()); rng.shuffle(keys)
    meta = []; k = 0
    classes = sorted({o["class"] for o in sg["objects"] if not o.get("is_structure")})
    for oid, sec in keys:
        cam = events[(oid, sec)]
        if k >= a.n: break
        fp = os.path.join(fr_dir, "%04d.jpg" % (sec + 1))
        if not os.path.exists(fp): continue
        bb = project_aabb(obj[oid]["aabb_world"], cam)
        if bb is None: continue
        im = Image.open(fp).convert("RGB")
        cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2
        s = max(bb[2]-bb[0], bb[3]-bb[1]) * 1.3
        crop = im.crop((int(cx-s), int(cy-s), int(cx+s), int(cy+s))).resize((336, 336))
        f1 = os.path.join(a.out, "cand_%04d.jpg" % k)
        crop.save(f1, quality=92)
        lab = obj[oid]["class"]
        alt = rng.choice([c for c in classes if c != lab])
        meta.append(dict(cand=f1, enroll=f1, label=lab, alt=str(alt), truth=1)); k += 1
        # 음성: 같은 프레임의 다른 물체 bbox 크롭에 이 라벨
        others = [(o2, s2) for (o2, s2) in keys if s2 == sec and o2 != oid]
        if others and k < a.n:
            o2, _ = others[int(rng.integers(len(others)))]
            bb2 = project_aabb(obj[o2]["aabb_world"], cam)
            if bb2:
                cx2, cy2 = (bb2[0]+bb2[2])/2, (bb2[1]+bb2[3])/2
                s2_ = max(bb2[2]-bb2[0], bb2[3]-bb2[1]) * 1.3
                c2 = im.crop((int(cx2-s2_), int(cy2-s2_), int(cx2+s2_), int(cy2+s2_))).resize((336, 336))
                f2 = os.path.join(a.out, "cand_%04d.jpg" % k)
                c2.save(f2, quality=92)
                meta.append(dict(cand=f2, enroll=f2, label=lab,
                                 alt=str(obj[o2]["class"]), truth=0)); k += 1
    open(os.path.join(a.out, "meta.jsonl"), "w").write(
        "\n".join(json.dumps(m) for m in meta))
    print("쌍 %d → %s" % (k, a.out))


if __name__ == "__main__":
    main()
