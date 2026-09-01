#!/usr/bin/env python3
"""초기 씬그래프 구축 — 매핑워크 프레임의 검출만으로 "무엇이 어느 방에" 를 만든다.

    THOR_ROOT=data/hssd20 A3_PREFIX=/tmp/hs_a_ MAP_PREFIX=/tmp/hs_m_ \\
      python scripts/build_initmap.py

**GT 를 쓰지 않는다.** 매핑워크(map/*.jpg)를 OWL 로 검출하고, 프레임이 어느 방에서
찍혔는지는 map 메타의 room 을 쓴다(매핑 로봇/사람이 방을 도는 순서는 아는 정보 —
배포에선 SLAM+방 분할이 준다). 물체 타입마다 "가장 강하게 검출된 방"을 등록한다.

산출: 각 채에 initmap_owl.json = [{type, room, w}] — eval_online SG_INIT=hybrid 가 읽는다.
이게 ① (안 움직인 물체) 의 **진짜 성적**을 만든다 — SG_INIT=gt 는 초기 기록이
정답이라 0.99 가 자명하게 나온다(2026-09-01 사용자 지적).
"""
import glob, json, os
import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

ROOT = os.environ.get("THOR_ROOT", "data/hssd20")
A3P = os.environ.get("A3_PREFIX", "/tmp/hs_a_")
TH = float(os.environ.get("TH", "0.12"))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()

def sp(t): return "a photo of a " + "".join(" " + c.lower() if c.isupper() else c
                                            for c in t).strip()

for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa = A3P + hn + ".npz"
    if not os.path.exists(fa): continue
    g = json.load(open(os.path.join(hd, "gt.json")))
    mp = g.get("map") or []
    mfs = sorted(glob.glob(os.path.join(hd, "map", "*.jpg")))
    if not mp or not mfs:
        print("  %s 매핑워크 없음 — 건너뜀" % hn, flush=True); continue
    vocab = list(np.load(fa, allow_pickle=True)["vocab"])
    ti = op(text=[[sp(v) for v in vocab]], images=[Image.new("RGB", (256, 256), (128,)*3)],
            return_tensors="pt").to(DEV)
    with torch.no_grad():
        o = on.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                     pixel_values=ti["pixel_values"], return_dict=True)
    TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)
    polys = g["scene_meta"]["polys"]
    GEO = os.environ.get("INITMAP_GEO", "1") == "1" and mp and mp[0].get("apos")
    Wf = float(os.environ.get("FRAME_W", "768")); Ff = Wf / 2
    def pbx(cx): return np.degrees(np.arctan((cx - Wf / 2) / Ff))
    def room_pt(pt):
        best = (1e9, None)
        for r, pl in polys.items():
            a = np.array(pl); c = a.mean(0)
            ins = (a[:, 0].min() <= pt[0] <= a[:, 0].max()) and (a[:, 1].min() <= pt[1] <= a[:, 1].max())
            d = 0.0 if ins else float(np.hypot(pt[0] - c[0], pt[1] - c[1]))
            if d < best[0]: best = (d, r)
        return best[1]

    # 방×타입 점수 누적. GEO=1 이면 **투영으로 물체 위치를 추정해 방을 정한다**
    # (프레임의 room 을 그대로 쓰면 문 너머 물체가 옆방으로 오염 — §121 후속 실측 0.316)
    acc = {}
    for k in range(0, min(len(mfs), len(mp)), 1):
        room = mp[k]["room"]
        im = Image.open(mfs[k]).convert("RGB")
        pv = op(images=[im], return_tensors="pt")["pixel_values"].to(DEV)
        with torch.no_grad():
            fm = on.image_embedder(pixel_values=pv)[0]
            b, ph, pw, hdim = fm.shape
            lg, _ = on.class_predictor(fm.reshape(b, ph*pw, hdim),
                                       TX.unsqueeze(0).expand(b, -1, -1),
                                       MK.unsqueeze(0).expand(b, -1))
        s = torch.sigmoid(lg).amax(1)[0].float().cpu().numpy()
        pw = fm.shape[2]
        if GEO:
            # 검출 패치 → 방위 → (GT 거리) → 지도 투영 → 방
            P_ = torch.sigmoid(lg).argmax(1)[0].int().cpu().numpy()
            ap = mp[k]["apos"]; yaw = float(mp[k]["yaw"])
            dmap = mp[k].get("dist") or {}
            oid_of = {}
            for oid2, c2 in (mp[k].get("ctr") or {}).items():
                oid_of.setdefault(round(c2[0] / 20), []).append(oid2)
            for c, v in enumerate(vocab):
                if s[c] < TH: continue
                cx = (P_[c] % pw + .5) / pw * Wf
                # 거리: 그 화면 위치에 가장 가까운 GT 물체의 거리 (배포에선 depth)
                cand = [(abs(((mp[k]["ctr"][o][0]) - cx)), o) for o in dmap]
                if not cand: continue
                dd, o_near = min(cand)
                if dd > 80: continue
                b = yaw + pbx(cx)
                d_ = dmap[o_near]
                pt = [ap[0] + d_ * np.sin(np.radians(b)), ap[1] + d_ * np.cos(np.radians(b))]
                rr = room_pt(pt)
                acc.setdefault(v, {}).setdefault(rr, 0.0)
                acc[v][rr] += float(s[c])
        else:
            for c, v in enumerate(vocab):
                if s[c] >= TH:
                    acc.setdefault(v, {}).setdefault(room, 0.0)
                    acc[v][room] += float(s[c])
    out = [dict(type=t, room=max(rs, key=rs.get), w=round(max(rs.values()), 3))
           for t, rs in acc.items()]
    json.dump(out, open(os.path.join(os.path.realpath(hd), "initmap_owl.json"), "w"))
    # 정확도 진단 (GT 대조 — 기록만, 구축엔 미사용)
    gt_room = {}
    for v in g["gt0"].values(): gt_room.setdefault(v["type"], v["room"])
    hit = [1 for e in out if gt_room.get(e["type"]) == e["room"]]
    print("  %s 타입 %d · 방배정 정확도 %.3f" % (hn, len(out),
          len(hit) / max(sum(1 for e in out if e["type"] in gt_room), 1)), flush=True)
print("완료")
