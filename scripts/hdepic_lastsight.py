#!/usr/bin/env python3
"""**마지막 목격**만으로 "그 물건 어디 있어" 에 답할 수 있는가 — 부재 판정 없이.

    $P scripts/hdepic_lastsight.py --root /Volumes/External_SSD/khronos/hdepic

설계 재검토(2026-08-21, 사용자 지적):

    거실에 있던 물건을 내가 부엌으로 옮겼다. 다시 거실에 가면 없는 것을 본다.
    이때 "어디 있어" 로 검색하면 **거실(있음) → 부엌(옮김) → 거실(부재)** 이 다 올라온다.
    시간순으로 **마지막에 물건이 보인 곳**이 부엌이므로, 그게 곧 답이다.
    **"거실에 없다" 를 따로 판정할 이유가 없다.**

즉 부재 판정은 **"어디 있어" 의 답이 아니라 "없어졌다" 의 답**이다. 두 과제가 다르다:

| 상황 | 마지막 목격으로 되나 |
|---|---|
| 내가 옮겼고 옮긴 곳을 봤다 | **된다** — 부재 불필요 |
| 남이 옮겼다 / 안 보는 새 옮겨졌다 | 안 된다 — 새 위치 관측이 없다 |
| "아직 거기 있나?" 를 물음 | 안 된다 — 부재 확인이 답 자체 |

여기서 재는 것: **마지막 목격 규칙의 정확도**. HD-EPIC 은 물체별 관측마다 `fixture`
가 붙어 있으므로 GT 가 명확하다.

  방식 A **GT 마지막 목격** — 관측 기록의 마지막 fixture. 상한(지각층 무관)
  방식 B **검색 기반**   — 질의 시각 이전 프레임에서 그 물체를 검색해 상위
                          프레임의 시각을 찾고, 그 시각의 GT fixture 를 답으로
  방식 C **씬그래프 미갱신** — 처음 본 fixture 를 그대로 답(대조군)

B 가 A 에 가까우면 **재검색만으로 씬그래프를 갱신할 수 있다**는 뜻이고,
C 와 비슷하면 검색이 갱신에 못 쓰인다는 뜻이다.
"""
import argparse, json, os, sys, tempfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FPS = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/External_SSD/khronos/hdepic")
    ap.add_argument("--part", default="P03")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--topk", type=int, default=5, help="검색 상위 k 프레임에서 시각 추정")
    ap.add_argument("--min-obs", type=int, default=3, help="관측이 이보다 적으면 제외")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch, cv2
    from PIL import Image
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                              CLIPTextModelWithProjection, CLIPTokenizer)
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cn = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    tk = CLIPTokenizer.from_pretrained(cm)
    tn = CLIPTextModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()

    def emb_img(paths, bs=32):
        out = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            with torch.no_grad():
                e = cn(**cp(images=ims, return_tensors="pt").to(args.device)).image_embeds
            out.append(e.cpu().numpy().astype(np.float32))
        E = np.concatenate(out)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def emb_txt(ws):
        with torch.no_grad():
            t = tk(["a photo of a " + w for w in ws], padding=True,
                   truncation=True, return_tensors="pt").to(args.device)
            e = tn(**t).text_embeds.cpu().numpy().astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    assoc = json.load(open(os.path.join(args.root, "assoc_info.json")))
    mask = json.load(open(os.path.join(args.root, "mask_info.json")))
    vdir = os.path.join(args.root, "Videos", args.part)
    tmp = tempfile.mkdtemp()
    rows = []

    vids = sorted(v for v in assoc if v.startswith(args.part)
                  and os.path.exists(os.path.join(vdir, v + ".mp4")))
    for vi_, vid in enumerate(vids):
        mp4 = os.path.join(vdir, vid + ".mp4")
        mi = mask.get(vid, {})
        objs = {}
        for oid, o in assoc[vid].items():
            nm = o.get("name")
            obs = sorted((mi[m]["frame_number"] / FPS, mi[m]["fixture"])
                         for t in o.get("tracks", []) for m in t.get("masks", [])
                         if m in mi and mi[m].get("fixture"))
            if nm and len(obs) >= args.min_obs and len({f for _, f in obs}) > 1:
                objs[oid] = (nm, obs)          # **fixture 가 바뀐 물체만** (갱신이 필요한 경우)
        if not objs:
            continue
        cap = cv2.VideoCapture(mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0
        if dur < 60:
            cap.release(); continue
        secs, paths = [], []
        for i, s in enumerate(np.arange(0, dur, args.every)):
            f = int(round(s * fps))
            if not (0 <= f < total):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            p = os.path.join(tmp, "L%03d_%05d.jpg" % (vi_, i))
            cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            secs.append(float(s)); paths.append(p)
        cap.release()
        if len(paths) < 20:
            continue
        ts = np.array(secs); E = emb_img(paths)
        names = [nm for nm, _ in objs.values()]
        T = emb_txt(names)

        for qi, (oid, (nm, obs)) in enumerate(objs.items()):
            # 질의 시각 = 마지막 관측 직후(그 시점에 "어디 있어" 를 묻는다)
            qt = obs[-1][0] + 1.0
            gt_last = obs[-1][1]                 # 정답 = 마지막 목격 fixture
            gt_first = obs[0][1]                 # 갱신 안 한 씬그래프의 답
            m = ts <= qt
            if m.sum() < 10 or gt_last == gt_first:
                continue
            tt = ts[m]
            sim = E[m] @ T[qi]
            top = tt[np.argsort(-sim)[:args.topk]]
            # 검색이 고른 시각 중 **가장 늦은 것**에 해당하는 GT fixture
            def fx_at(t):
                prev = [f for s_, f in obs if s_ <= t]
                return prev[-1] if prev else obs[0][1]
            pred_ret = fx_at(float(top.max()))
            rows.append(dict(vid=vid, obj=nm, n_obs=len(obs),
                             gt=gt_last, first=gt_first, ret=pred_ret,
                             a_ok=True, b_ok=pred_ret == gt_last,
                             c_ok=gt_first == gt_last))
        print("  %-22s 대상 %-3d (누적 %d)" % (vid[4:], len(objs), len(rows)), flush=True)

    if not rows:
        print("판정 없음")
        return
    n = len(rows)
    print("\n**fixture 가 바뀐 물체 %d건** (갱신이 필요한 경우만)" % n)
    print("  방식 A  GT 마지막 목격        %.3f  ← 상한(지각층 무관)" % 1.0)
    print("  방식 B  **검색 기반 마지막 목격** %.3f" % np.mean([r["b_ok"] for r in rows]))
    print("  방식 C  씬그래프 미갱신(첫 위치) %.3f  ← 대조군" % np.mean([r["c_ok"] for r in rows]))
    print("\n→ B 가 A(1.0)에 가까우면 **재검색만으로 씬그래프를 갱신할 수 있다**.")
    print("  B 가 C 수준이면 검색이 갱신에 못 쓰인다.")
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
