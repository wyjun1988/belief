#!/usr/bin/env python3
"""OmniGibson 렌더 프로브 — 뼈대. (RTX 노드 전용, Isaac Sim 필요)

    ~/og-venv/bin/python scripts/og_probe.py --scenes 5 --frames 200 --out /tmp/og_probe

⚠️ OmniGibson API 는 판마다 이름이 다르다 — 아래는 의도가 정확한 뼈대이며,
현지 판의 API 로 조정해도 된다. **불변 조건은 출력 형식 하나**:
    out/frames/%06d.jpg + out/meta.jsonl
    {"img": ..., "objs": [{"label": 카테고리, "ctr": [u,v], "dist": m}, ...]}
(probe_pairs.py → exp_vlm_verify3 → rtx7_sweep 이 그대로 이어진다)

의도:
  1. BEHAVIOR-1K 주거 장면 N개 로드 (Rs_int 등 int 계열 = 주거)
  2. 장면당 무작위 가시점에서 카메라 렌더: RGB + 인스턴스 세그 + 뎁스
  3. 인스턴스 → 카테고리(og 객체의 category), 화면 중심 = 세그 중앙값,
     거리 = 세그 영역 뎁스 중앙값. 벽/바닥/문 등 구조물 제외.
  4. 라벨 2종+ 프레임만 저장.
"""
import argparse, json, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--scenes", type=int, default=5)
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)

import omnigibson as og
from omnigibson.macros import gm
gm.HEADLESS = True

SCENES = ["Rs_int", "Ihlen_1_int", "Merom_1_int", "Benevolence_1_int", "Pomaria_1_int"]
SKIP = {"walls", "floors", "ceilings", "door", "window", "background"}
rng = np.random.default_rng(0)
meta = []; made = 0

for sc in SCENES[:args.scenes]:
    env = og.Environment(configs=dict(
        scene=dict(type="InteractiveTraversableScene", scene_model=sc),
        robots=[],
        env=dict(action_frequency=30, physics_frequency=30)))
    cam = og.sim.viewer_camera
    cam.image_width, cam.image_height = 768, 768
    cam.add_modality("seg_instance"); cam.add_modality("depth_linear")
    per = args.frames // args.scenes
    got = 0; tries = 0
    while got < per and tries < per * 10:
        tries += 1
        # 무작위 실내 시점: 장면 AABB 안 임의 위치 + 임의 yaw (카메라 높이 1.5)
        lo, hi = env.scene.get_aabb() if hasattr(env.scene, "get_aabb") else ((-5,-5,0),(5,5,3))
        pos = [float(rng.uniform(lo[0], hi[0])), float(rng.uniform(lo[1], hi[1])), 1.5]
        yaw = float(rng.uniform(0, 2*np.pi))
        cam.set_position_orientation(position=pos,
            orientation=[0, 0, float(np.sin(yaw/2)), float(np.cos(yaw/2))])
        og.sim.render()
        obs = cam.get_obs()[0]
        rgb, seg, dep = obs["rgb"][..., :3], obs["seg_instance"], obs["depth_linear"]
        info = cam.get_obs()[1].get("seg_instance", {})   # id → prim 경로/이름
        objs = []
        for iid in np.unique(seg):
            name = str(info.get(int(iid), "")).lower()
            if not name or any(s in name for s in SKIP): continue
            label = "".join(c for c in name.split("/")[-1] if c.isalpha())
            if not label: continue
            ys, xs = np.where(seg == iid)
            if len(xs) < 400: continue
            d = float(np.median(dep[ys, xs]))
            if not (0.3 < d < 15): continue
            objs.append(dict(label=label, ctr=[float(np.median(xs)), float(np.median(ys))],
                             dist=round(d, 2)))
        if len({o["label"] for o in objs}) < 2: continue
        fn = "frames/%06d.jpg" % made
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(args.out, fn), quality=90)
        meta.append(dict(img=fn, objs=objs)); made += 1; got += 1
    env.close()
    print("%s: %d장 (누적 %d)" % (sc, got, made), flush=True)

open(os.path.join(args.out, "meta.jsonl"), "w").write("\n".join(json.dumps(m) for m in meta))
assert made > 0, "프레임 0장"
print("프레임 %d → %s" % (made, args.out))
