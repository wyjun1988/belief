#!/usr/bin/env python3
"""부재를 **하나의 타임라인**에서 찾는다 — 장소도 시점도 스스로 찾아야 한다.

    $P scripts/scenediff_timeline.py --root <benchmark/data>

㉔ 의 AUC 0.655 는 "이 두 영상을 비교하라" 고 **장소를 떠먹여준** 값이다. 실사용은
그렇지 않다. 과제를 정확히 이렇게 정의한다:

    ① 있음 → 다른방 → **없음** → 다른방      정답 = **없음**
    ② 있음 → 다른방 → 다른방                 정답 = **있음**

①은 물건이 **있던 프레임과 없어진 프레임을 둘 다** 찾아야 맞힌다.
②는 그 장소를 다시 안 갔으므로 **마지막으로 본 상태(있음)** 가 답이다 — 기권이
아니라 마지막 상태로 답한다.

**다른 방은 함정이 아니라 과제의 일부다** — 아무 프레임이나 집어오지 않는다는 것을
확인시키는 역할이다. ①과 ②는 **같은 물체 · 같은 '있음' 근거 · 같은 방해 장면**을
쓰고 재방문 유무만 다르므로, 차이는 오직 "없어진 순간을 찾았는가" 에서 온다.

판정 알고리즘(우리 파이프라인 그대로):
    ⓐ 키워드를 뺀 문맥(PMI 이웃)으로 타임라인 전체에서 **그 장소 프레임**을 고른다
    ⓑ 그중 **가장 마지막 방문 구간**만 남긴다 (시간순 — 최신 관측이 현재 상태다)
    ⓒ 그 구간에서 키워드 검출도가 문턱 미만이면 '없음', 이상이면 '있음'
"""
import argparse, glob, json, os, pickle, re, sys, tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.scenediff_absence import base_label, frames_of, video_of   # noqa: E402
from scripts.absence_evidence import PLACES                            # noqa: E402


def pmi_from(P):
    n = P.shape[1]
    p = P.sum(1) / max(n, 1)
    J = (P.astype(np.float32) @ P.astype(np.float32).T) / max(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.log(J / (p[:, None] * p[None] + 1e-9) + 1e-9)
    G[~np.isfinite(G)] = 0.0
    np.fill_diagonal(G, -np.inf)
    return G


def visits_of(idx, max_gap):
    """고른 프레임 인덱스를 **방문 구간들**로 쪼갠다(간격 max_gap 이하면 같은 방문)."""
    if len(idx) == 0:
        return []
    idx = np.sort(idx)
    cut = np.nonzero(np.diff(idx) > max_gap)[0]
    out, s0 = [], 0
    for c in cut:
        out.append(idx[s0:c + 1]); s0 = c + 1
    out.append(idx[s0:])
    return out


def last_visit(idx, max_gap):
    """마지막 방문 구간만."""
    v = visits_of(idx, max_gap)
    return v[-1] if v else np.array([], int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--subset", default="all", choices=["all", "sdk", "sdv"])
    ap.add_argument("--limit", type=int, default=999)
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--ndistract", type=int, default=3, help="구간마다 끼울 다른 방 수")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--det-thr", type=float, default=0.05)
    ap.add_argument("--topm", type=int, default=4)
    ap.add_argument("--ctx-gate", type=float, default=1.0)
    ap.add_argument("--pres-z", type=float, default=1.5)
    ap.add_argument("--max-gap", type=int, default=2, help="같은 방문으로 볼 프레임 간격")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--place", default="latent", choices=["latent", "pmi", "gt"],
                    help="장소를 무엇으로 찾는가. latent=CLIP 프레임 임베딩 유사도"
                         "(우리 검색층과 같은 것) · pmi=물체 조합 · gt=상한")
    ap.add_argument("--anchor-q", type=float, default=0.70,
                    help="앵커 자기유사도의 이 분위를 절대 문턱으로 쓴다."
                         " **0.70 이 실측 최적**(균형정확도 0.807) — 스윕:"
                         " 0.25→0.699 · 0.55→0.783 · 0.70→**0.807** · 0.80→0.768"
                         " · 0.94→0.608. 올릴수록 방해 장면 오검이 급감하지만(61%→8%)"
                         " 0.80 부터는 진짜 재방문까지 잘려 다시 나빠진다. 낮출수록"
                         " 관대(재현↑·정밀↓). **퍼센타일 게이트를 대체한 것** —"
                         " 상위 N%% 로 자르면 방해 구간에서도 반드시 뽑혀 마지막"
                         " 방문이 늘 타임라인 끝이 된다")
    ap.add_argument("--visit-mode", default="last", choices=["last", "recency"],
                    help="last=마지막 방문만(현행) · recency=여러 방문을 최신 가중으로"
                         " 누적. 마지막 방문은 프레임이 적어 잡음에 흔들린다")
    ap.add_argument("--visit-tau", type=float, default=1.0,
                    help="recency: 방문 단위 감쇠. 작을수록 마지막 방문만 본다")
    ap.add_argument("--vote-k", type=int, default=3,
                    help="앵커 프레임 중 상위 몇 개와의 유사도를 평균할지(프레임 투표)")
    ap.add_argument("--cache", default=None,
                    help="클립별 OWL 점수 캐시(npz). 판정 로직만 바꿔 재실험할 때"
                         " 지각층을 다시 돌리지 않는다")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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
        rem = [o for o in (seg.get("objects") or [])
               if o.get("in_video1") and not o.get("in_video2")]
        pairs.append((name, sdk, v1, v2, rem))
    if len(pairs) < args.ndistract + 2:
        print("쌍이 부족하다")
        return

    # 조건① — 같은 기본 이름이 **전체 풀**에 둘 이상이면 개체를 못 가른다.
    allb = Counter(base_label(o["label"]) for _, _, _, _, rem in pairs for o in rem)
    # ⚠️ 변화한 물체 라벨만 어휘로 쓰면 **문맥이 주변 가구가 아니라 다른 장면의
    # 물체**가 된다(실측: `white bag` 의 이웃이 `yellow shoes`·`game controller`).
    # 그러면 어느 장면이나 문맥이 비슷해져 장소를 못 가른다 — 장소 적중 0.00 의 원인.
    # ADT 에서는 씬 물체 119개(가구 포함)를 썼다. 여기서도 장소 어휘를 더한다.
    vocab = sorted(set(allb) | set(PLACES))
    targets = [(n, sdk, o) for n, sdk, _, _, rem in pairs for o in rem
               if allb[base_label(o["label"])] == 1]
    print("쌍 %d · 어휘 %d · 조건① 통과 대상 %d (전체 %d)"
          % (len(pairs), len(vocab), len(targets), sum(allb.values())))

    clips, embs = {}, {}
    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        vocab = list(z["vocab"])
        for k in z.files:
            if k.startswith("S|"):
                clips[tuple(k[2:].split("|"))] = z[k]
            elif k.startswith("E|"):
                embs[tuple(k[2:].split("|"))] = z[k]
        print("캐시에서 %d클립 · 어휘 %d 적재" % (len(clips), len(vocab)))
        return run(args, pairs, targets, vocab, clips, embs)

    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    mdl = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(mdl)
    net = Owlv2ForObjectDetection.from_pretrained(mdl).to(args.device).eval()
    # ⚠️ `CLIPModel.from_pretrained` 는 이 저장소 환경에서 막힌다 — 가중치가 .bin
    # 뿐이고 torch 2.2.2 에서 transformers 가 torch.load 를 거부한다(CVE-2025-32434).
    # 저장소의 기존 CLIP 경로와 같이 **safetensors 로 강제**한다.
    cm = "openai/clip-vit-base-patch16"
    cproc = CLIPImageProcessor.from_pretrained(cm)
    cnet = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    tmp = tempfile.mkdtemp()

    def embed(imgs):
        """장소 찾기를 latent 로 하기 위한 프레임 임베딩 — 우리 검색층과 같은 것."""
        ims = [Image.open(p).convert("RGB") for p in imgs]
        with torch.no_grad():
            inp = cproc(images=ims, return_tensors="pt").to(args.device)
            e = cnet(**inp).image_embeds.cpu().numpy().astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    def score(imgs):
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

    # 클립별 점수를 **한 번만** 계산하고, 타임라인은 인덱싱으로 조립한다
    for k, (name, sdk, v1, v2, _) in enumerate(pairs):
        for w, v in (("v1", v1), ("v2", v2)):
            f = frames_of(v, args.nframes, tmp, "%03d%s" % (k, w))
            if len(f) >= 3:
                clips[(name, w)] = score(f)
                embs[(name, w)] = embed(f)
        print("  %-44s v1 %s · v2 %s (%d/%d)"
              % (name[:44], clips.get((name, "v1"), np.zeros((0, 0))).shape[1:],
                 clips.get((name, "v2"), np.zeros((0, 0))).shape[1:],
                 k + 1, len(pairs)))
    if not clips:
        print("프레임 없음")
        return
    if args.cache:
        d = {"S|%s|%s" % k: v for k, v in clips.items()}
        d.update({"E|%s|%s" % k: v for k, v in embs.items()})
        np.savez_compressed(args.cache, vocab=np.array(vocab, object), **d)
        print("→ 캐시 %s" % args.cache)
    return run(args, pairs, targets, vocab, clips, embs)


def run(args, pairs, targets, vocab, clips, embs):

    # z 정규화·PMI 는 **전체 풀**에서 한 번 (씬그래프 사전에 해당)
    ALL = np.concatenate(list(clips.values()), 1)
    mu, sd = ALL.mean(1, keepdims=True), ALL.std(1, keepdims=True) + 1e-9
    Zc = {k: (v - mu) / sd for k, v in clips.items()}
    G = pmi_from(np.concatenate(list(Zc.values()), 1) > args.pres_z)
    vi = {w: i for i, w in enumerate(vocab)}

    def judge(tl, em, ki, ctx, n0, owner, name):
        """타임라인 → (마지막 방문 구간의 키워드 값, 구간 길이, 구간 인덱스)

        **장소를 어떻게 찾는가**가 이 함수의 전부다.

          latent  물건을 본 근거 구간(앞머리 n0 프레임)을 앵커로, **그 이후** 프레임 중
                  그 장소로 보이는 것을 고른다. 씬그래프가 "이 물건을 마지막으로 본
                  프레임" 을 갖고 있으므로 앵커는 과거 정보다 — 인과를 안 어긴다.
          pmi     키워드를 뺀 물체 조합.
          gt      정답 장소를 그대로 준다 — 상한.

        ⚠️ **퍼센타일 게이트를 쓰면 안 된다.** 종전에는 상위 20% 로 잘랐는데, 그러면
        방해 장면만 있는 구간에서도 반드시 뭔가 뽑혀 **마지막 방문이 늘 타임라인 끝**
        이 됐다(장소 적중률이 pmi·latent 모두 정확히 0.00 이었던 원인).

        대신 **앵커 자기유사도로 절대 문턱을 교정한다**: 같은 장소를 찍은 앵커
        프레임끼리의 유사도 분포가 "이 장소는 이 정도로 닮는다" 의 기준이다.
        GT 없이 앵커만으로 정해지므로 실사용에서 그대로 쓸 수 있다.

        표현은 **CLIP** 을 쓴다. 4090 실측에서 CLIP CLS 가 DINO-VLAD 와 top-1 동률
        (51%)이고 recall@3 는 오히려 높다(63% vs 61%). 우리가 **이미 저장하는 것**이라
        추가 비용이 0 이다.
        """
        after = np.arange(n0, tl.shape[1])          # 근거 구간 이후만 본다(인과)
        if len(after) == 0:
            return None, 0, after
        if args.place == "gt":
            ok = np.array([i for i in after if owner[i][0] == name])
        elif args.place == "latent":
            A = em[:n0]
            if len(A) < 2:
                return None, 0, np.array([], int)
            # 앵커 자기유사도 → "이 장소는 이 정도로 닮는다"
            SS = A @ A.T
            iu = np.triu_indices(len(A), 1)
            base = SS[iu]
            thr = float(np.quantile(base, args.anchor_q))
            # **프레임 투표** — 앵커 평균 하나가 아니라 앵커 프레임 각각과 비교해
            # 상위 몇 개의 평균을 쓴다. 평균 벡터는 시선이 흩어진 앵커에서 뭉개진다.
            S = em[after] @ A.T                     # [after, anchor]
            k = min(args.vote_k, A.shape[0])
            sim = np.sort(S, 1)[:, -k:].mean(1)
            ok = after[sim >= thr]
        else:
            cs = tl[ctx][:, after].max(0)
            ok = after[cs >= args.ctx_gate]
        if len(ok) == 0:
            return None, 0, np.array([], int)
        vs = visits_of(ok, args.max_gap)
        if not vs:
            return None, 0, np.array([], int)
        if args.visit_mode == "last":
            sel = vs[-1]
            return float(np.median(tl[ki, sel])), len(sel), sel
        # **여러 방문 누적** — 마지막 방문 하나는 프레임이 적어 잡음에 흔들린다.
        # 방문마다 값을 내고 **최신일수록 크게** 가중해 합친다. 현재 상태를 묻는
        # 것이므로 과거 방문은 참고만 해야 한다(τ 는 방문 단위).
        vals = np.array([np.median(tl[ki, v]) for v in vs], float)
        ages = np.arange(len(vs) - 1, -1, -1, dtype=float)   # 마지막 방문이 0
        w = np.exp(-ages / max(args.visit_tau, 1e-6))
        sel = vs[-1]
        return float((vals * w).sum() / w.sum()), len(sel), sel

    rng = np.random.default_rng(args.seed)
    rows = []
    names = [p[0] for p in pairs]
    for name, sdk, o in targets:
        w = base_label(o["label"]); ki = vi[w]
        ctx = [j for j in np.argsort(-G[ki])[:args.topm] if G[ki, j] > 0.3]
        if not ctx or (name, "v1") not in Zc or (name, "v2") not in Zc:
            continue
        others = [n for n in names if n != name]
        pick = rng.permutation(len(others))[:args.ndistract * 2]
        d1 = [others[i] for i in pick[:args.ndistract]]
        d2 = [others[i] for i in pick[args.ndistract:]]

        def build(blocks):
            mats, ems, owner = [], [], []
            for nm, wv in blocks:
                key = (nm, wv)
                if key not in Zc:
                    continue
                mats.append(Zc[key]); ems.append(embs[key])
                owner += [(nm, wv)] * Zc[key].shape[1]
            return np.concatenate(mats, 1), np.concatenate(ems, 0), owner

        # ① 있음 → 다른방 → 없음 → 다른방   (정답: 없음)
        b1 = [(name, "v1")] + [(n, "v1") for n in d1] + [(name, "v2")] + [(n, "v2") for n in d2]
        # ② 있음 → 다른방 → 다른방          (정답: 있음)
        b2 = [(name, "v1")] + [(n, "v1") for n in d1] + [(n, "v2") for n in d2]
        for case, blocks, truth in ((1, b1, "없음"), (2, b2, "있음")):
            tl, em, owner = build(blocks)
            n0 = Zc[(name, "v1")].shape[1]      # 앞머리 = 물건을 본 근거 구간
            val, nsel, sel = judge(tl, em, ki, ctx, n0, owner, name)
            if val is None:
                # **재방문을 못 찾았다 → 마지막으로 본 상태로 답한다 = '있음'.**
                # 기권이 아니다. ②는 이것이 정답이고, ①은 이것이 오답이다.
                rows.append(dict(pair=name, sdk=sdk, word=w, case=case, truth=truth,
                                 val=None, revisit=False, place_hit=0.0))
                continue
            # 마지막 방문이 실제로 그 장소였나 (①은 v2, ②는 v1 이 정답 블록)
            # ①은 v2(없어진 뒤), ②는 재방문이 없으니 근거 구간(v1)이 정답
            want = (name, "v2") if case == 1 else (name, "v1")
            hit = float(np.mean([owner[i] == want for i in sel]))
            rows.append(dict(pair=name, sdk=sdk, word=w, case=case, truth=truth,
                             val=val, revisit=True, nsel=int(nsel), place_hit=hit,
                             ctx=[vocab[j] for j in ctx]))
    report(rows, args.place)
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


def report(rows, mode="?"):
    """정답 규칙: 재방문을 찾았으면 그 구간의 키워드로 판정, 못 찾았으면 '있음'."""
    n1 = [r for r in rows if r["case"] == 1]
    n2 = [r for r in rows if r["case"] == 2]
    print("\n[장소 찾기 = %s] 타임라인 %d개 (①없음 %d · ②있음 %d)"
          % (mode, len(rows), len(n1), len(n2)))
    if not n1 or not n2:
        return
    for tag, s_ in (("①(정답 없음)", n1), ("②(정답 있음)", n2)):
        rv = [r for r in s_ if r["revisit"]]
        print("  %s 재방문 찾음 %d/%d (%.0f%%) · 그중 정답 장소 비율 중앙 %.2f"
              % (tag, len(rv), len(s_), 100 * len(rv) / len(s_),
                 np.median([r["place_hit"] for r in rv]) if rv else float("nan")))
    # ②에서 재방문을 못 찾으면 '있음' 이라 **자동 정답**이다. ①은 자동 오답.
    vals = [r["val"] for r in rows if r["val"] is not None]
    if len(vals) < 6:
        print("  판정 표본 부족")
        return
    print("\n  %-8s %-9s %-9s %-9s %s" % ("문턱", "①정답률", "②정답률", "균형정확도", ""))
    best = None
    for t in np.percentile(vals, [10, 25, 50, 75, 90]):
        def acc(s_, want_absent):
            c = 0
            for r in s_:
                said_absent = (r["val"] is not None) and (r["val"] < t)
                c += int(said_absent == want_absent)
            return c / len(s_)
        a1, a2 = acc(n1, True), acc(n2, False)
        ba = (a1 + a2) / 2
        print("  %-8.3f %-9.2f %-9.2f %-9.3f %s"
              % (t, a1, a2, ba, "**우연 초과**" if ba > 0.5 else ""))
        if best is None or ba > best[0]:
            best = (ba, t, a1, a2)
    print("  → 최고 균형정확도 **%.3f** (문턱 %.3f · ① %.2f · ② %.2f)" % best)


if __name__ == "__main__":
    main()
