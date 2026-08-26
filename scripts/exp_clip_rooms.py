#!/usr/bin/env python3
"""매핑워크 CLIP 노드 캐시 — 방 인지 방출(emission)용.

    THOR_ROOT=data/thor5 CLIP_PREFIX=~/khcache/c5_5_ python scripts/exp_clip_rooms.py

⚠️ 왜: 정적 검출 점수로 프레임 방을 분류하면 0.431(Viterbi 로도 0.476)이다 —
오픈플랜에서 한 프레임에 여러 방 가구가 보이는 **계통 오류**라 스무딩이 안 먹힌다.
§57 에서 통했던 것은 CLIP 임베딩의 노드(방마다 여러 프레임) 최대 유사도 방출이다.
매핑워크가 방 라벨 붙은 프레임을 방마다 수백 장 주므로 노드 재료로 그대로 쓴다.
(배포에서도 매핑워크의 방 라벨은 맵 구축 시점 정보라 허용된다.)
"""
import glob, json, os
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

ROOT = os.environ.get("THOR_ROOT", "data/thor5")
OUT = os.path.expanduser(os.environ.get("CLIP_PREFIX", "/tmp/c5_"))
STRIDE = int(os.environ.get("STRIDE", "8"))
MWSTRIDE = int(os.environ.get("MWSTRIDE", "4"))
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available() else "cpu")
CK = "openai/clip-vit-base-patch16"
pr = CLIPProcessor.from_pretrained(CK)
md = CLIPModel.from_pretrained(CK).to(DEV).eval()
print("CLIP 노드 캐시 · %s" % DEV, flush=True)


def emb(paths):
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + 16]]
        with torch.no_grad():
            e = md.get_image_features(**pr(images=ims, return_tensors="pt").to(DEV))
        e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


for hd in sorted(glob.glob(ROOT + "/house_*")):
    rd = os.path.realpath(hd); hn = os.path.basename(rd)
    out = OUT + hn + ".npz"
    if os.path.exists(out): continue
    wf = os.path.join(rd, "mapwalk", "walk.json")
    if os.path.exists(wf):
        w = json.load(open(wf))
        fr = w["frames"][::MWSTRIDE]
        mwp = [os.path.join(rd, "mapwalk", "%05d.jpg" % f["k"]) for f in fr]
        mwr = [f["room"] for f in fr]
    else:
        # ⚠️ 매핑워크가 없는 판(thor4 — H100 불칸 부재로 생성 불가)은 **텔레포트 맵
        # 프레임**을 노드로 쓴다. gt.json["map"] 이 프레임별 방 라벨을 갖고 있다.
        # §57 이 실제로 쓴 구성이 이것이다(방마다 3지점 × 4방향 = 12노드).
        g0 = json.load(open(os.path.join(rd, "gt.json")))
        mp = sorted(glob.glob(os.path.join(rd, "map", "*.jpg")))
        if not mp or len(g0.get("map", [])) != len(mp): continue
        mwp = mp
        mwr = [m["room"] for m in g0["map"]]
    lv = sorted(glob.glob(os.path.join(rd, "live", "*.jpg")))[::STRIDE]
    ts = [int(os.path.basename(p)[:-4]) for p in lv]
    ME = emb(mwp); LE = emb(lv)
    np.savez_compressed(out, me=ME, mr=np.array(mwr, object),
                        le=LE, ts=np.array(ts))
    print("  %s · 노드 %d · 배회 %d" % (hn, len(ME), len(LE)), flush=True)
print("완료")
