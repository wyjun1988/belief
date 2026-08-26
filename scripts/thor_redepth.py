#!/usr/bin/env python3
"""기존 매핑워크의 **뎁스만 재촬영** — walk.json 의 (pos, yaw) 를 그대로 재방문.

    THOR_ROOT=data/thor5 python scripts/thor_redepth.py

⚠️ gen2 통합 판(thor5·thor6)이 renderDepthImage 누락으로 뎁스 0장이었다.
reset 후 재방문이므로 **이동 물체는 원위치가 아니다** — 타겟 픽셀의 뎁스는 그
물체가 아니라 배경(놓인 면)을 재지만, 면과 물체의 깊이 차는 방 배정 오차(~1m)
안이라 초기맵 용도로 허용한다. 정적 물체는 완전히 정확하다.
"""
import glob, json, os
import numpy as np

ROOT = os.environ.get("THOR_ROOT", "data/thor5")
import prior
from ai2thor.controller import Controller
ds = prior.load_dataset("procthor-10k")["train"]
ctrl = None
for hd in sorted(glob.glob(ROOT + "/house_*")):
    rd = os.path.realpath(hd)
    wf = os.path.join(rd, "mapwalk", "walk.json")
    if not os.path.exists(wf): continue
    dd = os.path.join(rd, "mapwalk", "depth")
    w = json.load(open(wf))
    if os.path.exists(os.path.join(dd, "%05d.npy" % w["frames"][-1]["k"])):
        continue
    os.makedirs(dd, exist_ok=True)
    g = json.load(open(os.path.join(rd, "gt.json")))
    h = ds[g["house"]]
    if ctrl is None:
        ctrl = Controller(scene=h, width=w["size"], height=w["size"], quality="Low",
                          renderDepthImage=True)
    else:
        ctrl.reset(scene=h, renderDepthImage=True)
    n = 0
    for fr in w["frames"]:
        e = ctrl.step("Teleport", position=dict(x=fr["pos"][0], y=0.9, z=fr["pos"][1]),
                      rotation=dict(x=0, y=fr["yaw"], z=0), horizon=10)
        if e.metadata["lastActionSuccess"] and e.depth_frame is not None:
            np.save(os.path.join(dd, "%05d.npy" % fr["k"]),
                    e.depth_frame.astype(np.float16)); n += 1
    print("  %s 뎁스 %d/%d" % (os.path.basename(hd), n, len(w["frames"])), flush=True)
print("완료")
