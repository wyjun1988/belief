#!/usr/bin/env python3
"""카메라 방을 **좌표 없이** 정한다 — 매핑워크 검색(3D 재구성 불필요).

    THOR_ROOT=data/hssd20_c3_q4 A3_PREFIX=~/khcache/bench/cache/hs2_a_ \
      OUT_JSONL=~/khcache/room_retr.jsonl python scripts/room_retrieval.py

설계 복귀(2026-09-05 사용자 지적): 이 프로젝트의 답은 **방**이지 좌표가 아니다.
사다리의 '위치·포즈·카메라방' 을 실물화하면서 SfM/VGGT 로 미터 좌표를 구하려 했는데,
HSSD 렌더에서 재구성이 4채 중 3채 접힌다(§164). 좌표를 아예 구하지 않는 길:

  매핑워크 프레임에는 **사람이 준 방 라벨**이 붙어 있다(등록 시 "여기가 부엌").
  → 방마다 OWL 검출 프로파일(어휘 83차원)을 만들고,
  → live 프레임의 같은 차원 프로파일과 코사인 최근접으로 방을 정한다.

산출: jsonl {house, t, room, sim} — eval_online 의 ROOM_JSONL 이 읽는다(사다리 '카메라방:검색').
GT 0: 방 라벨은 사용자 입력, 검출은 OWL, 좌표는 등장하지 않는다.
"""
import glob, json, os, sys
import numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
ROOT = os.environ.get("THOR_ROOT", "data/hssd20_c3_q4")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "/tmp/hs2_a_"))
OUTJ = os.path.expanduser(os.environ.get("OUT_JSONL", "/tmp/room_retr.jsonl"))
TOPK = int(os.environ.get("ROOM_TOPK", "5"))       # 방 프로파일: 그 방 프레임 중 상위 K 평균
SMOOTH = int(os.environ.get("ROOM_SMOOTH", "5"))   # 시간 평활 창(프레임) — 사람은 순간이동하지 않는다

stat = json.load(open("data/thor_static_types.json"))
tg = set()
for hd in sorted(glob.glob(ROOT + "/house_*")):
    g = json.load(open(os.path.join(hd, "gt.json")))
    tg |= {v["type"] for v in g["gt0"].values()}
vocab = sorted(tg) + [s for s in stat if s not in tg]
def sp(t): return "a photo of a " + "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
ti = op(text=[[sp(v) for v in vocab]], images=[Image.new("RGB", (256, 256), (128,)*3)],
        return_tensors="pt").to(DEV)
with torch.no_grad():
    o = on.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                 pixel_values=ti["pixel_values"], return_dict=True)
TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)
print("어휘 %d · 방 프로파일 top%d · 평활 %d" % (len(vocab), TOPK, SMOOTH), flush=True)

def frames_scores(paths):
    S = []
    for i in range(0, len(paths), 4):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i+4]]
        pv = op(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
        with torch.no_grad():
            fm = on.image_embedder(pixel_values=pv)[0]
            b, ph, pw, hdim = fm.shape
            lg, _ = on.class_predictor(fm.reshape(b, ph * pw, hdim),
                                       TX.unsqueeze(0).expand(b, -1, -1), MK.unsqueeze(0).expand(b, -1))
        S.append(torch.sigmoid(lg).amax(1).float().cpu().numpy())
    return np.concatenate(S) if S else np.zeros((0, len(vocab)), np.float32)

def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)

fo = open(OUTJ, "w"); tot = hit = 0
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    g = json.load(open(os.path.join(hd, "gt.json")))
    mp = g.get("map") or []
    mfs = sorted(glob.glob(os.path.join(hd, "map", "*.jpg")))[:len(mp)]
    if not mfs: print("  %s map 없음 — 건너뜀" % hn, flush=True); continue
    MS = frames_scores(mfs)                                   # (M, V)
    rooms = [m["room"] for m in mp[:len(MS)]]                 # 사용자 입력 라벨
    prof, rlist = [], []
    for r in sorted(set(rooms)):
        idx = [i for i, x in enumerate(rooms) if x == r]
        if not idx: continue
        v = MS[idx]
        # 방 프로파일 = 그 방 프레임들의 성분별 상위 K 평균(한 장의 우연을 덜 탄다)
        k = min(TOPK, len(v))
        prof.append(np.sort(v, axis=0)[-k:].mean(0)); rlist.append(r)
    P = unit(np.array(prof, np.float32))
    z = np.load(A3P + hn + ".npz", allow_pickle=True)         # live 프레임 검출(같은 어휘)
    LS = unit(z["s"].astype(np.float32)); ts = z["ts"]
    sim = LS @ P.T                                            # (L, R)
    if SMOOTH > 1:                                            # 시간 평활 — 이동 평균
        c = np.cumsum(np.vstack([np.zeros((1, sim.shape[1])), sim]), 0)
        w = SMOOTH; L = len(sim)
        sm = np.array([(c[min(i + w // 2 + 1, L)] - c[max(i - w // 2, 0)]) /
                       (min(i + w // 2 + 1, L) - max(i - w // 2, 0)) for i in range(L)])
        sim = sm
    pick = sim.argmax(1)
    gtr = [g["live"][int(t)]["room"] if isinstance(g["live"], dict) else g["live"][int(i)]["room"]
           for i, t in enumerate(ts)] if False else None
    lv = {m["t"]: m for m in g["live"]} if isinstance(g["live"], list) else g["live"]
    ok = n = 0
    for i, t in enumerate(ts):
        r = rlist[pick[i]]
        m = lv.get(int(t)) if isinstance(lv, dict) else None
        if m is not None and m.get("room") is not None:
            n += 1; ok += (r == m["room"])
        fo.write(json.dumps({"house": hn, "t": int(t), "room": r,
                             "sim": float(sim[i, pick[i]])}) + "\n")
    tot += n; hit += ok
    print("  %s 방 %d · live %d · GT 일치 %.2f" % (hn, len(rlist), len(ts), ok / max(n, 1)), flush=True)
fo.close()
print("검색 기반 카메라방 → %s · 전체 GT 일치 %.3f (n=%d)" % (OUTJ, hit / max(tot, 1), tot), flush=True)
