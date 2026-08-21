#!/usr/bin/env python3
"""부재 판정의 **장소 신호**와 **문맥 선택**을 갈라 잰다.

    $P scripts/nymeria_absence_ablate.py

부재 판정은 두 단계다:
    ① **그 장소 프레임을 찾는다**
    ② 그 안에서 키워드 존재도를 재고 전/후 하락을 본다

지금까지 ①을 **물체 조합**(키워드 뺀 PMI 이웃)으로 해왔다. 그런데 같은 데이터에서
장소 식별이 **물체 조합 0.644 vs CLIP latent 0.900** 이었다(㉛). ①을 latent 로
갈아끼우면 부재도 오를 것이라는 것이 이 실험의 가설이다(사용자 제안 1).

⚠️ 앞서 잰 "마스크 latent"(효과 없음)와는 **다른 것**이다. 그건 앵커에서 물체를
빼 **편향을 없애려던** 것이고(마스크가 패치의 3%뿐이라 무효였다), 이건 **장소 찾기
신호 자체를 교체**하는 것이다.

그리고 문맥으로 쓸 물체를 **큰 것 위주로** 고른다(사용자 제안 2). 작은 물체를
문맥으로 쓰면 **그것도 없어졌을 수 있어** 장소 앵커로 못 믿는다 — 조리대·냉장고·
소파처럼 안 움직이는 것이 안정적이다.

비교:
  장소 신호  ① geo(궤적 군집 — 상한, 포즈가 필요) · ② latent(방 키) · ③ 물체 조합
  문맥 선택  전체 어휘 vs **큰 물체(가구)만**
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.nymeria_graph import D, house_frame, traj_of        # noqa: E402
from scripts.absence_evidence import PLACES                      # noqa: E402

# 안 움직이는 큰 물체 = 안정적인 장소 앵커. PLACES 에 Nymeria 어휘 몇 개를 더한다.
BIG = set(PLACES) | {"kitchen counter", "wardrobe", "window", "door", "oven",
                     "microwave", "refrigerator", "tv", "chair"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--det", default="owl_det.json")
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--topm", type=int, default=4, help="문맥 이웃 수")
    ap.add_argument("--noise-frac", type=float, default=0.25,
                    help="한 세션에서 소수 방 검출이 이 비율 미만이면 잡음으로 무시")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scipy.cluster.vq import kmeans2
    from scipy.stats import mannwhitneyu
    det = json.load(open(os.path.join(D, args.det)))
    seqs = {}
    for sd in sorted(glob.glob(os.path.join(D, "loc49", "*"))):
        if os.path.isdir(sd):
            try:
                seqs[os.path.basename(sd)] = traj_of(sd)
            except Exception:
                pass
    _, e1, e2, ctr = house_frame(seqs)
    P = np.concatenate([p for _, p in seqs.values()])
    U = np.stack([P @ e1, P @ e2], 1) - ctr
    cen, _ = kmeans2(U, args.k, minit="++", seed=0, iter=60)

    rows = []
    for s, (sec, p) in seqs.items():
        u = np.stack([p @ e1, p @ e2], 1) - ctr
        for r in det.get(s, []):
            i = int(np.argmin(np.abs(sec - r["sec"])))
            rm = int(np.argmin(np.linalg.norm(u[i] - cen, axis=1)))
            rows.append(dict(s=s, sec=float(r["sec"]), geo=rm,
                             det={w: v for w, v in r["det"].items()
                                  if v >= args.score_thr}))
    sess = sorted({r["s"] for r in rows})
    print("세션 %d · 검출 프레임 %d · 방 %d" % (len(sess), len(rows), args.k))

    # ── CLIP latent 로 방 키 (①-b 와 같은 방식)
    import cv2, torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cn = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    feats = np.zeros((len(rows), 512), np.float32)
    have = np.zeros(len(rows), bool)
    for s in sess:
        mp4 = os.path.join(D, "loc49_rgb", s + ".mp4")
        if not os.path.exists(mp4):
            continue
        cap = cv2.VideoCapture(mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = [i for i, r in enumerate(rows) if r["s"] == s]
        ims, keep = [], []
        for i in idx:
            f = int(round(rows[i]["sec"] * fps))
            if not (0 <= f < tot):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            ims.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            keep.append(i)
        cap.release()
        for j in range(0, len(ims), 32):
            with torch.no_grad():
                e = cn(**cp(images=ims[j:j+32], return_tensors="pt").to(
                    args.device)).image_embeds.cpu().numpy().astype(np.float32)
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
            for t, i in enumerate(keep[j:j+32]):
                feats[i] = e[t]; have[i] = True
        print("  %-40s 프레임 %d" % (s[:40], len(keep)), flush=True)
    print("latent 확보 %d/%d" % (int(have.sum()), len(rows)))

    # 방 키 = geo 라벨별 latent 평균(씬그래프 구축 시 남기는 것에 해당)
    kl = sorted({r["geo"] for r in rows})
    K = np.stack([feats[[i for i, r in enumerate(rows)
                         if r["geo"] == L and have[i]]].mean(0) for L in kl])
    K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
    lat_room = np.array([kl[int(np.argmax(K @ feats[i]))] if have[i] else -1
                         for i in range(len(rows))])

    # PMI (물체 조합) — 문맥용
    vocab = sorted({w for r in rows for w in r["det"]})
    vi = {w: i for i, w in enumerate(vocab)}
    Pm = np.zeros((len(vocab), len(rows)), bool)
    for j, r in enumerate(rows):
        for w in r["det"]:
            Pm[vi[w], j] = True
    pr = Pm.mean(1)
    J = (Pm.astype(np.float32) @ Pm.astype(np.float32).T) / len(rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.log(J / (pr[:, None] * pr[None] + 1e-9) + 1e-9)
    G[~np.isfinite(G)] = 0.0
    np.fill_diagonal(G, -np.inf)

    # ── 물체별 세션 이력
    # ⚠️ **"방이 하나라도 다르면 이동" 은 틀린 라벨이다.** 검출 오류 1건에 라벨이
    # 뒤집힌다 — 실측: `knife` 가 부엌 132 · 거실 1 · 거실2 1 로 잡혀 "이동" 이 됐고,
    # `kitchen counter`(움직일 리 없는 것)도 부엌 136 · 거실 7 로 "이동" 이 됐다.
    # 그 결과 이동 25 · 정적 6 이라는 **뒤집힌 비율**이 나왔다(집에서는 대부분 안 움직인다).
    #
    # 두 가지를 고친다:
    #   ① **다수결 방** — 세션에서 가장 많이 검출된 방만 그 세션의 위치로.
    #      게다가 **소수 방 검출이 전체의 --noise-frac 미만이면 무시**한다.
    #   ② **다중 개체 제외** — 모든 세션에서 여러 방에 꾸준히 잡히면 `tv`·`sofa` 처럼
    #      **개체가 여럿**이다(거실마다 하나씩). 같은 물건이 아니므로 뺀다.
    #      ⚠️ Nymeria 에는 물체 GT·개체 ID 가 없다(메타는 세션 정보뿐) — 검출만으로
    #      가려야 한다.
    obj = defaultdict(list)
    multi = set()
    for w in {w for r in rows for w in r["det"]}:
        per = []
        for s in sess:
            c = Counter()
            for r in rows:
                if r["s"] == s:
                    for ww in r["det"]:
                        if ww == w:
                            c[r["geo"]] += 1
            if c:
                per.append(c)
        if not per:
            continue
        # ② 여러 방에 **꾸준히**(과반 세션에서 2개 방 이상) 잡히면 다중 개체
        if sum(1 for c in per if len(c) >= 2) >= max(2, len(per) * 0.5):
            multi.add(w); continue
        for si, s in enumerate(sess):
            c = Counter()
            for r in rows:
                if r["s"] == s:
                    for ww in r["det"]:
                        if ww == w:
                            c[r["geo"]] += 1
            if not c:
                continue
            tot = sum(c.values())
            top, n1 = c.most_common(1)[0]
            # ① 다수결이 뚜렷할 때만 채택(소수 방은 잡음으로 본다)
            if n1 / tot >= 1.0 - args.noise_frac:
                obj[w].append((si, top))
    print("다중 개체로 제외 %d종: %s" % (len(multi), sorted(multi)[:10]))

    def score(place_mode, big_only):
        mv, st = [], []
        for w, hist in obj.items():
            if len(hist) < 3:
                continue
            rms = [rm for _, rm in hist]
            r0 = rms[0]
            if place_mode == "geo":
                sel = [i for i, r in enumerate(rows)
                       if r["geo"] == r0 and sess.index(r["s"]) > hist[0][0]]
            elif place_mode == "latent":
                sel = [i for i, r in enumerate(rows)
                       if lat_room[i] == r0 and sess.index(r["s"]) > hist[0][0]]
            else:                                    # 물체 조합
                ki = vi[w]
                cand = np.argsort(-G[ki])
                ctx = [j for j in cand
                       if G[ki, j] > 0.3 and (not big_only or vocab[j] in BIG)][:args.topm]
                if not ctx:
                    continue
                cs = Pm[ctx].any(0)
                sel = [i for i in range(len(rows))
                       if cs[i] and sess.index(rows[i]["s"]) > hist[0][0]]
            if len(sel) < 5:
                continue
            v = float(np.median([rows[i]["det"].get(w, 0.0) for i in sel]))
            (mv if len(set(rms)) > 1 else st).append(v)
        if len(mv) < 3 or len(st) < 3:
            return None
        u, p = mannwhitneyu(st, mv, alternative="greater")
        return u / (len(mv) * len(st)), p, len(mv), len(st)

    print("\n%-22s %-8s %-9s %s" % ("장소 신호 · 문맥", "AUC", "p", "이동/정적"))
    out = {}
    for pm, big, nm in (("geo", False, "geo(궤적) — 상한"),
                        ("latent", False, "**latent(방 키)**"),
                        ("pmi", False, "물체 조합(전체)"),
                        ("pmi", True, "**물체 조합(큰 물체만)**")):
        r = score(pm, big)
        if r:
            print("%-22s %-8.3f %-9.3f %d/%d" % (nm, r[0], r[1], r[2], r[3]))
            out[nm] = r
        else:
            print("%-22s 표본 부족" % nm)
    if args.out:
        json.dump({k: list(v) for k, v in out.items()}, open(args.out, "w"),
                  ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
