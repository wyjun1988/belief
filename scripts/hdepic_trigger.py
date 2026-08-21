#!/usr/bin/env python3
"""**장소 매칭 트리거**의 정밀도 — 갱신을 언제 돌릴지 정하는 신호.

    $P scripts/hdepic_trigger.py --root /Volumes/External_SSD/khronos/hdepic

설계(2026-08-21, 사용자 제안):

    씬그래프를 만들 때 각 장소마다 **latent 키**를 남긴다. 기록 중에는 **30초 주기**로
    현재 latent 를 구해 키와 대조하고, **매칭되는 순간이 키프레임**이다. 그때만
    무거운 것(seg·검출)을 돌려 씬그래프를 갱신한다.

왜 이 구조가 나은가:
- **라우터 비용이 사실상 0** — 12시간에 1,440장 = CLIP 배치로 27초.
  손-물체 접촉 검출은 상시 모델이 필요해 비교가 안 된다.
- **사건이 아니라 상태를 본다** — "언제 놓았나" 를 짚을 필요가 없다.
  `조리대 방문 → 컵 없음` + `싱크 방문 → 컵 있음` 두 관측이면 **옮김이 유도된다**.
  re-ID 없이 성립한다.
- 질의 때 "마지막 목격을 검색으로 찾는" 일이 없어진다(그것이 0.253 으로 실패했다).

⚠️ **위험: 장소 매칭이 355곳 기준 top-1 51% 였다.** 트리거가 틀리면 엉뚱한 방의
상태를 써넣어 **그래프를 오염시킨다**(갱신은 읽기가 아니라 쓰기라 누적된다).
다만 실사용 조건은 훨씬 유리하다 — 여기서 그 셋을 하나씩 켜며 잰다:

  A **단순 최근접**   키 중 가장 닮은 것 (기준선)
  B **+ 연속성**      직전 판정에 가산점 (순간이동 안 한다)
  C **+ 자기검증**    그 장소에 있어야 할 물체가 실제로 보이는지 확인
  D **B+C**

GT: HD-EPIC 은 관측마다 `fixture` 가 붙어 있으므로 "그 시각에 어느 가구 앞인가" 를
채점할 수 있다. ⚠️ **키는 앞부분에서만 만들고 뒷부분에서 채점한다**(누수 방지).
"""
import argparse, json, os, sys, tempfile
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FPS = 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/External_SSD/khronos/hdepic")
    ap.add_argument("--part", default="P03")
    ap.add_argument("--period", type=float, default=30.0, help="트리거 주기(초)")
    ap.add_argument("--key-frac", type=float, default=0.4,
                    help="앞 이 비율로 키를 만들고 나머지로 채점(누수 방지)")
    ap.add_argument("--cont", type=float, default=0.05, help="연속성 가산점")
    ap.add_argument("--min-vis", type=int, default=3, help="장소로 칠 최소 관측 수")
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
    tot = {k: [0, 0] for k in ("A", "B", "C", "D")}
    n_place = []

    vids = sorted(v for v in assoc if v.startswith(args.part)
                  and os.path.exists(os.path.join(vdir, v + ".mp4")))
    for vi_, vid in enumerate(vids):
        mi = mask.get(vid, {})
        # 시각별 fixture (GT) + 그 장소의 물체 목록(자기검증용)
        obs = []
        fx_objs = defaultdict(Counter)
        for oid, o in assoc[vid].items():
            nm = o.get("name")
            for t in o.get("tracks", []):
                for m in t.get("masks", []):
                    r = mi.get(m)
                    if r and r.get("fixture"):
                        obs.append((r["frame_number"] / FPS, r["fixture"]))
                        if nm:
                            fx_objs[r["fixture"]][nm] += 1
        if len(obs) < 20:
            continue
        obs.sort()
        keep = {f for f, c in Counter(f for _, f in obs).items() if c >= args.min_vis}
        obs = [(t, f) for t, f in obs if f in keep]
        if len(keep) < 2:
            continue
        cap = cv2.VideoCapture(os.path.join(vdir, vid + ".mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0
        if dur < 120:
            cap.release(); continue
        # 트리거 표집 + 키 생성용 프레임(관측 시각)
        want = sorted({round(t, 1) for t in np.arange(0, dur, args.period)} |
                      {round(t, 1) for t, _ in obs})
        secs, paths = [], []
        for i, s in enumerate(want):
            f = int(round(s * fps))
            if not (0 <= f < total):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            p = os.path.join(tmp, "T%03d_%05d.jpg" % (vi_, i))
            cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            secs.append(float(s)); paths.append(p)
        cap.release()
        if len(paths) < 30:
            continue
        ts = np.array(secs); E = emb_img(paths)
        split = ts.min() + (ts.max() - ts.min()) * args.key_frac

        # ── 장소 키: **앞부분 관측만** 으로 만든다(누수 방지)
        keys, knames = [], []
        for fxn in sorted(keep):
            tt = [t for t, f in obs if f == fxn and t <= split]
            if len(tt) < 2:
                continue
            idx = np.unique(np.searchsorted(ts, np.array(tt)))
            idx = idx[(idx >= 0) & (idx < len(ts))]
            if len(idx) < 2:
                continue
            v = E[idx].mean(0); v /= np.linalg.norm(v) + 1e-9
            keys.append(v); knames.append(fxn)
        if len(keys) < 2:
            continue
        K = np.stack(keys)
        n_place.append(len(keys))
        # 자기검증용 — 장소별 대표 물체
        OBJW = [[w for w, _ in fx_objs[f].most_common(4)] for f in knames]
        flat = sorted({w for ws in OBJW for w in ws})
        T = emb_txt(flat) if flat else None
        wi = {w: i for i, w in enumerate(flat)}

        # ── 채점: 뒷부분의 트리거 시각마다 GT fixture 와 대조
        prev = None
        for i, t in enumerate(ts):
            if t <= split or abs(t % args.period) > 0.2:
                continue
            near = [f for tt, f in obs if abs(tt - t) <= args.period / 2]
            if not near:
                continue
            gt = Counter(near).most_common(1)[0][0]
            if gt not in knames:
                continue
            sim = K @ E[i]
            a = int(np.argmax(sim))
            # B 연속성
            simB = sim.copy()
            if prev is not None:
                simB[prev] += args.cont
            b = int(np.argmax(simB))
            # C 자기검증 — 그 장소에 있어야 할 물체가 실제로 보이는가
            if T is not None:
                z = T @ E[i]
                bonus = np.array([np.mean([z[wi[w]] for w in ws]) if ws else 0.0
                                  for ws in OBJW], np.float32)
                bonus = (bonus - bonus.mean()) / (bonus.std() + 1e-9)
                simC = sim + 0.02 * bonus
                simD = simB + 0.02 * bonus
            else:
                simC, simD = sim, simB
            c = int(np.argmax(simC)); d = int(np.argmax(simD))
            for k, pick in (("A", a), ("B", b), ("C", c), ("D", d)):
                tot[k][0] += int(knames[pick] == gt); tot[k][1] += 1
            prev = d
        print("  %-22s 장소 %-2d · 채점 %d (누적 %d)"
              % (vid[4:], len(keys), tot["A"][1], tot["A"][1]), flush=True)

    if not tot["A"][1]:
        print("판정 없음")
        return
    n = tot["A"][1]
    ch = 1.0 / np.mean(n_place)
    print("\n트리거 판정 %d회 · 장소 평균 %.1f곳 · **우연 %.3f**" % (n, np.mean(n_place), ch))
    print("\n%-24s %-8s %s" % ("방식", "정확도", "우연대비"))
    for k, nm in (("A", "단순 최근접"), ("B", "+연속성"),
                  ("C", "+자기검증"), ("D", "**연속성+자기검증**")):
        acc = tot[k][0] / tot[k][1]
        print("%-24s %-8.3f %.1f배" % (nm, acc, acc / ch))
    print("\n→ 90%대면 이 설계가 선다. 60%대면 그래프 오염 때문에 못 쓴다.")
    if args.out:
        json.dump({k: v for k, v in tot.items()}, open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
