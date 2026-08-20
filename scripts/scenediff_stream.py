#!/usr/bin/env python3
"""영상을 이어 붙여 **장소 수준** 부재로 만든다 — 장소를 떠먹여주지 않는 조건.

    $P scripts/scenediff_stream.py --root <benchmark/data>

㉔ 의 AUC 0.655 는 **쌍 안에서만** 비교한 값이다. "이 두 영상을 비교하라" 고
장소를 떠먹여준 셈이라, 우리 설계의 핵심인 **"문맥(물체 조합)으로 장소를 먼저
찾는다"** 단계가 아예 작동하지 않았다. ㉓ 에서 그 단계의 정밀도가 0.37 이었으므로
그것을 포함하면 얼마나 깎이는지가 이 실험의 질문이다.

구성 — 실사용과 같은 모양으로 잇는다:

    세션 A = 모든 쌍의 **video1**(전) 프레임을 이어 붙인 것   ← "예전의 집"
    세션 B = 모든 쌍의 **video2**(후) 프레임을 이어 붙인 것   ← "지금의 집"

그러면 장소가 N개인 건초더미가 되고, 질의는 "그 물건 아직 거기 있나" 가 된다.
**다른 장소가 섞여 있는 것은 함정이 아니라 과제다** — 그 장면을 찾아내는 것까지가
이 시스템의 일이다.

그래서 한 걸음 더 간다: `--unseen R` 로 일부 쌍의 **video2 를 세션 B 에서 빼면**,
그 장소는 "오늘 가보지 않은 방" 이 된다. 그 방 물건에 대한 정답은 "없어졌다" 가
아니라 **"판정 보류"** 다(증거의 부재 ≠ 부재의 증거). 문맥 게이트가 이걸 제대로
기권하는지가 세 번째 채점 항목이다 — 기권 못 하면 안 가본 방을 전부 도난 신고한다.

채점은 우리 파이프라인 그대로:
    ① 키워드를 뺀 문맥(PMI 이웃)으로 A·B 각각에서 **장소 프레임**을 고른다
    ② 그 안에서만 키워드 검출도를 재고 A→B 하락을 본다
양성 = Removed · 음성 = Moved(씬에 남아 있음).
"""
import argparse, glob, json, os, pickle, re, subprocess, sys, tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.scenediff_absence import base_label, frames_of, video_of   # noqa: E402


def pmi_from(P):
    """P[word, frame] 이진 존재 → PMI 그래프."""
    n = P.shape[1]
    p = P.sum(1) / max(n, 1)
    J = (P.astype(np.float32) @ P.astype(np.float32).T) / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.log(J / (p[:, None] * p[None] + 1e-9) + 1e-9)
    G[~np.isfinite(G)] = 0.0
    np.fill_diagonal(G, -np.inf)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", default="all", choices=["all", "sdk", "sdv"])
    ap.add_argument("--limit", type=int, default=999)
    ap.add_argument("--nframes", type=int, default=8, help="영상당 표본 프레임")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--det-thr", type=float, default=0.05)
    ap.add_argument("--topm", type=int, default=4, help="문맥 이웃 수")
    ap.add_argument("--topf", type=int, default=8, help="장소 프레임 수")
    ap.add_argument("--ctx-gate", type=float, default=1.0, help="문맥 z 문턱")
    ap.add_argument("--pres-z", type=float, default=1.5, help="PMI 용 존재 z 문턱")
    ap.add_argument("--unseen", type=float, default=0.0,
                    help="이 비율의 쌍은 video2 를 세션 B 에서 뺀다 — '오늘 안 가본 방'."
                         " 그 방 물건의 정답은 '없어짐' 이 아니라 **판정 보류** 다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ── ① 쌍을 모아 어휘와 프레임 목록을 만든다
    pairs = []
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if len(pairs) >= args.limit or not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        sdk = bool(re.match(r"^P\d", name))
        if (args.subset == "sdk" and not sdk) or (args.subset == "sdv" and sdk):
            continue
        f = os.path.join(d, "segments.pkl")
        v1, v2 = video_of(d, 1), video_of(d, 2)
        if not (os.path.exists(f) and v1 and v2):
            continue
        try:
            seg = pickle.load(open(f, "rb"))
        except Exception:
            continue
        objs = [(o, "Removed" if not o.get("in_video2") else "Moved")
                for o in (seg.get("objects") or []) if o.get("in_video1")]
        if not objs:
            continue
        pairs.append((name, sdk, d, v1, v2, objs))
    if not pairs:
        print("쓸 수 있는 쌍이 없다")
        return

    # 조건① — 같은 기본 이름이 **전체 풀**에 둘 이상이면 개체를 못 가른다.
    # 쌍 안에서만 보던 ㉔ 보다 엄격하다(이어 붙이면 다른 장소의 동명 물체도 섞인다).
    allb = Counter(base_label(o["label"]) for _, _, _, _, _, os_ in pairs for o, _ in os_)
    targets = [(n, sdk, o, st) for n, sdk, _, _, _, os_ in pairs for o, st in os_
               if allb[base_label(o["label"])] == 1]
    vocab = sorted({base_label(o["label"]) for _, _, _, _, _, os_ in pairs for o, _ in os_})
    print("쌍 %d · 어휘 %d · 조건① 통과 대상 %d (전체 %d)"
          % (len(pairs), len(vocab), len(targets), sum(allb.values())))

    # ── ② 세션 A(전) / B(후) 프레임을 뽑는다
    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    mdl = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(mdl)
    net = Owlv2ForObjectDetection.from_pretrained(mdl).to(args.device).eval()
    tmp = tempfile.mkdtemp()

    def score_frames(imgs):
        S = np.zeros((len(vocab), len(imgs)), np.float32)
        for i, p in enumerate(imgs):
            im = Image.open(p).convert("RGB")
            with torch.no_grad():
                inp = proc(text=[vocab], images=im, return_tensors="pt").to(args.device)
                o = net(**inp)
                r = proc.post_process_grounded_object_detection(
                    o, threshold=args.det_thr,
                    target_sizes=torch.tensor([im.size[::-1]]).to(args.device))[0]
            for lb, sc in zip(r["labels"].tolist(), r["scores"].tolist()):
                S[lb, i] = max(S[lb, i], float(sc))
        return S

    rng = np.random.default_rng(args.seed)
    nun = int(round(args.unseen * len(pairs)))
    unseen = set(np.array([p[0] for p in pairs])[rng.permutation(len(pairs))[:nun]].tolist())
    if unseen:
        print("오늘 안 가본 방 %d곳(세션 B 에서 제외): %s"
              % (len(unseen), sorted(unseen)[:4]))
    SA, SB, owner_a, owner_b = [], [], [], []
    for k, (name, sdk, d, v1, v2, _) in enumerate(pairs):
        fa = frames_of(v1, args.nframes, tmp, "a%03d" % k)
        fb = [] if name in unseen else frames_of(v2, args.nframes, tmp, "b%03d" % k)
        if len(fa) < 3:
            continue
        SA.append(score_frames(fa)); owner_a += [name] * len(fa)
        if len(fb) >= 3:
            SB.append(score_frames(fb)); owner_b += [name] * len(fb)
        print("  %-44s A %2d · B %2d 프레임 (누적 %d/%d)"
              % (name[:44], len(fa), len(fb), len(owner_a), len(owner_b)))
    if not SA:
        print("프레임 없음")
        return
    A = np.concatenate(SA, 1); B = np.concatenate(SB, 1)
    owner_a = np.array(owner_a); owner_b = np.array(owner_b)
    print("\n세션 A %d프레임 · 세션 B %d프레임 · 어휘 %d" % (A.shape[1], B.shape[1], len(vocab)))

    # z 정규화(우리 파이프라인과 동일) + PMI 그래프는 A·B 합쳐서
    Z = np.concatenate([A, B], 1)
    Zz = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-9)
    Az, Bz = Zz[:, :A.shape[1]], Zz[:, A.shape[1]:]
    G = pmi_from(Zz > args.pres_z)
    vi = {w: i for i, w in enumerate(vocab)}

    def side(Zs, ki, ctx):
        cs = Zs[ctx].max(0)
        ok = np.nonzero(cs >= args.ctx_gate)[0]
        if len(ok) < 3:
            return None, None
        sel = ok[np.argsort(-cs[ok])[:min(args.topf, len(ok))]]
        return float(np.median(Zs[ki, sel])), sel

    rows, abstain = [], []
    for name, sdk, o, st in targets:
        w = base_label(o["label"])
        ki = vi[w]
        ctx = [j for j in np.argsort(-G[ki])[:args.topm] if G[ki, j] > 0.3]
        if not ctx:
            continue
        za, sa = side(Az, ki, ctx)
        zb, sb = side(Bz, ki, ctx)
        if za is None or zb is None:
            # 장소를 세션 B 에서 못 찾았다 = **판정 보류**. 안 가본 방이면 정답,
            # 가본 방인데 못 찾았으면 문맥 게이트의 재현율 손실이다.
            abstain.append((name, w, st, name in unseen))
            continue
        # 장소 찾기 정확도 — 고른 프레임이 실제로 그 쌍의 것이었나
        pa = float((owner_a[sa] == name).mean()); pb = float((owner_b[sb] == name).mean())
        rows.append(dict(pair=name, sdk=sdk, word=w, status=st, z_before=za,
                         z_after=zb, drop=za - zb, place_prec_a=pa, place_prec_b=pb,
                         ctx=[vocab[j] for j in ctx]))
    report(rows, args, abstain, unseen)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


def report(rows, args, abstain=(), unseen=()):
    from scipy.stats import mannwhitneyu
    print("\n판정 가능 %d개 · 판정 보류 %d개" % (len(rows), len(abstain)))
    if unseen:
        # 안 가본 방 물건은 **보류가 정답**이다
        un = [a for a in abstain if a[3]]
        tot_un = len(un) + sum(1 for r in rows if r["pair"] in unseen)
        print("Ⓓ 기권 판정 — 안 가본 방 물건 %d개 중 **%d개 보류(정답)** = %.0f%%"
              % (tot_un, len(un), 100 * len(un) / max(tot_un, 1)))
        seen_ab = [a for a in abstain if not a[3]]
        print("   (가본 방인데 보류한 것 %d개 = 문맥 게이트 재현율 손실)" % len(seen_ab))
    if not rows:
        return
    pa = np.array([r["place_prec_a"] for r in rows])
    print("Ⓐ 문맥이 고른 장소 프레임의 정밀도: A 중앙 **%.2f** · B 중앙 %.2f"
          % (np.median(pa), np.median([r["place_prec_b"] for r in rows])))
    rows = [r for r in rows if r["pair"] not in unseen]     # 채점은 가본 방만
    for tag, sel in (("전체", rows), ("SD-K", [r for r in rows if r["sdk"]]),
                     ("SD-V", [r for r in rows if not r["sdk"]])):
        R = [r for r in sel if r["status"] == "Removed"]
        M = [r for r in sel if r["status"] == "Moved"]
        print("\n[%s] Removed %d · Moved %d" % (tag, len(R), len(M)))
        if len(R) < 3 or len(M) < 3:
            print("  표본 부족")
            continue
        for thr in (-9, 0.0, 0.5):
            P = [r["drop"] for r in R if r["z_before"] >= thr]
            N = [r["drop"] for r in M if r["z_before"] >= thr]
            if len(P) < 3 or len(N) < 3:
                print("  조건②≥%-5s n=%d/%d 표본 부족" % (thr, len(P), len(N)))
                continue
            u, p = mannwhitneyu(P, N, alternative="greater")
            print("  조건②≥%-5s n=%-6s AUC **%.3f** · p=%.4f"
                  % ("없음" if thr < -1 else "%.1f" % thr,
                     "%d/%d" % (len(P), len(N)), u / (len(P) * len(N)), p))


if __name__ == "__main__":
    main()
