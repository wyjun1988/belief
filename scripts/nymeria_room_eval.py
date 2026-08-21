#!/usr/bin/env python3
"""Nymeria 로 **방 단위** 네 지표를 교차 확인한다 — SuperMemory 결과의 검증.

    $P scripts/nymeria_room_eval.py

SuperMemory 방 단위 실측(㉛)에서 네 지표가 다 살아났다:
장소 0.615 · 검색 0.863 · 마지막목격 0.661 · 부재 AUC 0.688.

다만 유보가 있었다 — 표본 118 물체 · 2세션이고, 방 6개 중 5개가 `kitchen` 으로
대응돼 **실질은 2분류**에 가까웠다. Nymeria 는 조건이 다르다:

| | SuperMemory | **Nymeria** |
|---|---|---|
| 집 | 개방형 | **벽으로 갈림** |
| 방 | 군집 6개(실질 2분류) | **거실2 · 부엌1 = 3개** |
| 세션 | 3개 | **11개** |
| 방 belief 기존 실측 | s8 0.86 | **+43%**(k=2) |

⚠️ Nymeria 는 **세션 간**이 자연스러운 축이다(같은 집을 11번 방문). 물체가 세션
사이에 옮겨지므로 부재·마지막목격이 **하루 스케일**에 가깝다 — 다른 데이터에는
없던 조건이다.
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.nymeria_graph import D, house_frame, traj_of        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="방 군집 수(씬그래프가 3개)")
    ap.add_argument("--det", default="owl_det.json")
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--latent", action="store_true",
                    help="CLIP latent 키로도 장소 식별을 잰다(SuperMemory 와 같은 방식)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from scipy.cluster.vq import kmeans2
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
    print("세션 %d · 방 %d개" % (len(seqs), args.k))

    # 세션별 프레임 → (방, 검출)
    rows = []                                   # (세션, 초, 방, {물체:점수})
    for s, (sec, p) in seqs.items():
        u = np.stack([p @ e1, p @ e2], 1) - ctr
        for r in det.get(s, []):
            i = int(np.argmin(np.abs(sec - r["sec"])))
            rm = int(np.argmin(np.linalg.norm(u[i] - cen, axis=1)))
            rows.append((s, float(r["sec"]), rm,
                         {w: v for w, v in r["det"].items() if v >= args.score_thr}))
    if not rows:
        print("검출 없음"); return
    sess = sorted({r[0] for r in rows})
    print("검출 프레임 %d · 방 분포 %s" % (len(rows), dict(Counter(r[2] for r in rows))))

    # ── 물체별 방 이력 (세션 순서 = 시간 순서)
    obj = defaultdict(list)                     # 물체 → [(세션idx, 방)]
    for si, s in enumerate(sess):
        by = defaultdict(Counter)
        for r in rows:
            if r[0] != s:
                continue
            for w in r[3]:
                by[w][r[2]] += 1
        for w, c in by.items():
            obj[w].append((si, c.most_common(1)[0][0]))
    print("물체 %d종" % len(obj))

    # ── ①-b **CLIP latent 키**로 장소 식별 (SuperMemory 와 같은 방식)
    #     ⚠️ 물체 조합 방식(①-a)과 나란히 재야 공정하다 — SuperMemory 에서는
    #     latent 키를 썼는데 여기서 물체 조합만 쓰면 방식 차이가 데이터 차이로 보인다.
    if args.latent:
        import cv2, torch
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        cm = "openai/clip-vit-base-patch16"
        cp = CLIPImageProcessor.from_pretrained(cm)
        cn = CLIPVisionModelWithProjection.from_pretrained(
            cm, use_safetensors=True).to(args.device).eval()
        import tempfile
        tmp = tempfile.mkdtemp()
        feats, flab, fses = [], [], []
        for s_ in sess:
            mp4 = os.path.join(D, "loc49_rgb", s_ + ".mp4")
            if not os.path.exists(mp4):
                continue
            cap = cv2.VideoCapture(mp4)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            ims, labs_ = [], []
            for r in rows:
                if r[0] != s_:
                    continue
                fno = int(round(r[1] * fps))
                if not (0 <= fno < tot):
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                ok, img = cap.read()
                if not ok:
                    continue
                ims.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                labs_.append(r[2])
            cap.release()
            for i in range(0, len(ims), 32):
                with torch.no_grad():
                    e = cn(**cp(images=ims[i:i+32], return_tensors="pt").to(
                        args.device)).image_embeds.cpu().numpy().astype(np.float32)
                feats.append(e)
            flab += labs_; fses += [s_] * len(labs_)
            print("  %-40s 프레임 %d" % (s_[:40], len(labs_)), flush=True)
        if feats:
            F = np.concatenate(feats)
            F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
            flab = np.array(flab); fses = np.array(fses)
            a2, b2 = [], []
            for ho in sorted(set(fses.tolist())):
                tr = fses != ho; te = fses == ho
                if te.sum() < 5 or len(set(flab[tr].tolist())) < 2:
                    continue
                ks = sorted(set(flab[tr].tolist()))
                K = np.stack([F[tr][flab[tr] == k].mean(0) for k in ks])
                K = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
                pr = np.array([ks[int(np.argmax(K @ F[i]))] for i in np.nonzero(te)[0]])
                a2.append(float((pr == flab[te]).mean()))
                b2.append(Counter(flab[te].tolist()).most_common(1)[0][1] / te.sum())
            if a2:
                print("\n①-b **장소 식별(CLIP latent)** %.3f · 최빈방 %.3f"
                      % (np.mean(a2), np.mean(b2)))

    # ── ①-a 장소 식별 — 물체 조합으로 방을 맞히는가(세션 하나 빼고 학습)
    acc, base = [], []
    for ho in range(len(sess)):
        tr = [r for r in rows if r[0] != sess[ho]]
        te = [r for r in rows if r[0] == sess[ho]]
        if len(te) < 5:
            continue
        prof = defaultdict(Counter)
        for r in tr:
            for w in r[3]:
                prof[r[2]][w] += 1
        ks = sorted(prof)
        if len(ks) < 2:
            continue
        vocab = sorted({w for c in prof.values() for w in c})
        M = np.array([[prof[k][w] for w in vocab] for k in ks], float)
        M = M / (M.sum(1, keepdims=True) + 1e-9)
        wi = {w: i for i, w in enumerate(vocab)}
        ok = 0
        for r in te:
            v = np.zeros(len(vocab))
            for w in r[3]:
                if w in wi:
                    v[wi[w]] = 1
            ok += int(ks[int(np.argmax(M @ v))] == r[2])
        acc.append(ok / len(te))
        base.append(Counter(r[2] for r in te).most_common(1)[0][1] / len(te))
    if acc:
        print("\n①-a **장소 식별(물체 조합)** %.3f · 최빈방 %.3f · 우연 %.3f"
              % (np.mean(acc), np.mean(base), 1.0 / args.k))

    # ── ②③ 증거 검색 · 마지막 목격 (방 단위)
    hit, last = [], []
    for w, hist in obj.items():
        if len(hist) < 2:
            continue
        # 검색: 그 물체 검출이 가장 강한 프레임 상위 k
        cand = [(r[3].get(w, 0.0), r) for r in rows if w in r[3]]
        if len(cand) < args.topk:
            continue
        cand.sort(key=lambda x: -x[0])
        top = [r for _, r in cand[:args.topk]]
        gtr = {rm for _, rm in hist}
        hit.append(any(r[2] in gtr for r in top))
        # 마지막 목격: 상위 중 가장 늦은 세션의 방 = 마지막 세션의 방인가
        latest = max(top, key=lambda r: (sess.index(r[0]), r[1]))
        last.append(latest[2] == hist[-1][1])
    if hit:
        print("② **증거 검색(방 일치)** hit@%d %.3f (물체 %d)"
              % (args.topk, float(np.mean(hit)), len(hit)))
        print("③ **마지막 목격(방)** %.3f" % float(np.mean(last)))

    # ── ④ 부재: 세션 간 방이 바뀐 물체 vs 안 바뀐 물체
    mv, st = [], []
    for w, hist in obj.items():
        if len(hist) < 3:
            continue
        rms = [rm for _, rm in hist]
        r0 = rms[0]
        later = [r for r in rows if r[2] == r0 and sess.index(r[0]) > hist[0][0]]
        if len(later) < 5:
            continue
        v = float(np.median([r[3].get(w, 0.0) for r in later]))
        (mv if len(set(rms)) > 1 else st).append(v)
    if len(mv) >= 3 and len(st) >= 3:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(st, mv, alternative="greater")
        print("④ **부재(방 이탈)** 이동 %d · 정적 %d · AUC **%.3f** (p=%.3f)"
              % (len(mv), len(st), u / (len(mv) * len(st)), p))
    else:
        print("④ 부재 — 표본 부족(이동 %d · 정적 %d)" % (len(mv), len(st)))
    print("\n(대조: SuperMemory 방 단위 — 장소 0.615 · 검색 0.863 · 마지막 0.661 · 부재 0.688)")
    if args.out:
        json.dump(dict(place=float(np.mean(acc)) if acc else None,
                       hit=float(np.mean(hit)) if hit else None,
                       last=float(np.mean(last)) if last else None),
                  open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
