#!/usr/bin/env python3
"""부재 증거 — "물건이 없어진 것"을 검색으로 확인하고 씬그래프를 고친다.

    $P scripts/absence_evidence.py --seq <adt-seq>          # GT 로 기제 채점
    $P scynthesis/absence_evidence.py --supermem            # SuperMemory 적용

설계(사용자 아이디어 4단계):

    ① 씬그래프 구성        1fps 스트림에서 물체 존재판정 → 동시출현(PMI) 그래프
    ② 확장 질의 구성       keyword K → E(K) = K + 문맥(이웃 물체·장소)
    ③ **keyword 제거 검색** C(K) = E(K) − K → K 가 **있어야 할 장소**의 프레임을 찾는다
    ④ 부재 확인 + 갱신     그 프레임들에서 K 의 존재도를 재고, 낮으면 '그 자리에 없다'

왜 keyword 를 빼는가: K 로 검색하면 K 가 보이는 프레임만 올라와 **부재를 볼 수 없다**
(생존 편향). 문맥만으로 장소를 찾으면 K 가 있든 없든 그 장소가 잡히고, 그 안에서
K 의 존재를 물어야 부재가 관측 가능해진다. 이것이 v2 원안의 미해결 항목
("없어진 것은 업데이트가 안 된다")에 대한 답이다.

채점(ADT): GT 이동구간이 있는 물체는 이동 후 원래 장소에서 **부재해야** 하고,
정적 물체는 계속 **존재해야** 한다. 이 둘을 가르는 능력을 AUC 로 잰다.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PLACES = ["kitchen drawer", "kitchen cabinet", "counter", "kitchen island", "refrigerator",
          "stove", "sink", "dining table", "sofa", "coffee table", "bookshelf", "tv stand",
          "nightstand", "dresser", "closet", "desk", "wall", "floor", "shelf", "bed"]


def clip_text(texts, device="mps"):
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    nm = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(nm)
    txt = CLIPTextModelWithProjection.from_pretrained(nm, use_safetensors=True).eval().to(device)
    out = []
    for i in range(0, len(texts), 256):
        with torch.no_grad():
            tt = tok(texts[i:i + 256], padding=True, truncation=True, return_tensors="pt").to(device)
            e = txt(**tt).text_embeds
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out)


def presence(E, vocab, device="mps", z_thr=1.5):
    """① 물체 존재판정 — 프레임별 CLIP z점수. GT 를 쓰지 않는다."""
    V = clip_text(["a photo of a " + w for w in vocab], device)
    S = V @ E.T                                        # (vocab, frame)
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    return Z, (Z > z_thr)


def pmi_graph(P, vocab):
    """① 동시출현 PMI 그래프 — 물체 사이 문맥 관계."""
    n = P.shape[1]
    pa = P.mean(1)
    G = np.zeros((len(vocab), len(vocab)))
    for i in range(len(vocab)):
        co = (P[i:i + 1] & P).mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            G[i] = np.log(np.maximum(co, 1e-9) / np.maximum(pa[i] * pa, 1e-9))
    np.fill_diagonal(G, -9)
    return G


def context_of(G, vocab, k_idx, topm=4, min_pmi=0.3):
    """② 확장 질의의 문맥부 — PMI 이웃 상위 topm."""
    nb = np.argsort(-G[k_idx])[:topm]
    return [j for j in nb if G[k_idx, j] > min_pmi]


def absence_score(Z, P, G, vocab, k_idx, frames_a, frames_b, topm=4, topf=12,
                  ctx_gate=1.0, min_frames=5):
    """③④ keyword 를 뺀 문맥 질의로 장소 프레임을 찾고, 그 안에서 K 의 존재를 잰다.

    frames_a(이전) / frames_b(이후) 두 구간에서 각각 계산해 **변화**를 본다 —
    절대 존재도는 CLIP 편향이 섞이지만, 같은 물체의 전후 차이는 그것이 상쇄된다.
    """
    ctx = context_of(G, vocab, k_idx, topm)
    if not ctx:
        return None
    def side(fr):
        if len(fr) < 3:
            return None
        # ⚠️ '증거의 부재' 와 '부재의 증거' 를 갈라야 한다. 문맥 점수가 문턱을 넘는
        # 프레임 = **그 장소를 실제로 봤다**. 그런 프레임이 없으면 부재를 주장할 수
        # 없다(실측: 게이트 없이는 착용자가 안 간 장소의 정적 물체가 부재로 오판돼
        # AUC 0.604 에 머물렀다).
        fr = np.array(fr)
        cs = Z[ctx][:, fr].max(0)
        ok = fr[cs >= ctx_gate]
        if len(ok) < min_frames:
            return None                                  # 장소 미방문 → 판정 보류
        csel = Z[ctx][:, ok].max(0)
        sel = ok[np.argsort(-csel)[:min(topf, len(ok))]]
        return float(np.median(Z[k_idx, sel]))
    za, zb = side(frames_a), side(frames_b)
    if za is None or zb is None:
        return None
    return dict(ctx=[vocab[j] for j in ctx], z_before=za, z_after=zb, drop=za - zb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Apartment_release_decoration_seq137_M1292")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--device", default="mps")
    ap.add_argument("--topm", type=int, default=4, help="문맥 이웃 수")
    ap.add_argument("--topf", type=int, default=12, help="장소 프레임 수")
    ap.add_argument("--ctx-gate", type=float, default=1.0,
                    help="문맥 z 문턱 — 이 값 이상인 프레임만 '그 장소를 봤다' 로 인정."
                         " 부재 주장의 전제조건(없으면 판정 보류)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    z = np.load(os.path.join(sd, "clip_frames.npz"))
    E, fidx = z["emb"].astype(np.float32), z["idx"]
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]

    # 어휘: GT 카테고리(이름만 빌린다 — 위치·존재 정보는 안 쓴다) + 장소
    cats = sorted({(r.get("category") or "").strip() for r in gt.values() if r.get("category")})
    vocab = sorted(set([c for c in cats if c and len(c) > 2]) | set(PLACES))
    print("① 씬그래프: 프레임 %d · 어휘 %d개" % (len(E), len(vocab)))
    Z, P = presence(E, vocab, args.device)
    print("   존재판정 밀도 %.1f개/프레임" % P.sum(0).mean())
    G = pmi_graph(P, vocab)
    vi = {w: i for i, w in enumerate(vocab)}

    # 채점 대상: 이동 물체(부재해야 함) vs 정적 물체(존재해야 함)
    movers, statics = {}, []
    for k, r in gt.items():
        c = (r.get("category") or "").strip()
        if c not in vi:
            continue
        if r.get("moves"):
            e = max(m["end_idx"] for m in r["moves"])
            s = min(m["start_idx"] for m in r["moves"])
            movers[c] = (s, e)
        elif r.get("motion_type") == "static":
            statics.append(c)
    statics = [c for c in set(statics) if c not in movers]
    print("② 확장질의 · ③ keyword 제거 검색 · ④ 부재 판정")
    print("   이동 물체 %d개 · 정적 대조 %d개" % (len(movers), len(statics)))

    rows = []
    for c, (s, e) in movers.items():
        fa = [i for i, f in enumerate(fidx) if f < s]
        fb = [i for i, f in enumerate(fidx) if f > e]
        r = absence_score(Z, P, G, vocab, vi[c], fa, fb, args.topm, args.topf,
                          args.ctx_gate)
        if r:
            r.update(obj=c, kind="이동", span=(s, e))
            rows.append(r)
    # 정적 대조: 같은 시점 분할(중간)로 전후 비교
    mid = int(np.median(fidx))
    for c in statics:
        fa = [i for i, f in enumerate(fidx) if f < mid]
        fb = [i for i, f in enumerate(fidx) if f >= mid]
        r = absence_score(Z, P, G, vocab, vi[c], fa, fb, args.topm, args.topf,
                          args.ctx_gate)
        if r:
            r.update(obj=c, kind="정적", span=None)
            rows.append(r)

    mv = [r["drop"] for r in rows if r["kind"] == "이동"]
    st = [r["drop"] for r in rows if r["kind"] == "정적"]
    print("\n부재 신호(z 하락) — 이동 %d개 중앙 %+.2f · 정적 %d개 중앙 %+.2f"
          % (len(mv), np.median(mv) if mv else 0, len(st), np.median(st) if st else 0))
    if mv and st:
        # AUC: 이동 물체의 하락이 정적보다 큰가
        pairs = [(a > b) + 0.5 * (a == b) for a in mv for b in st]
        auc = float(np.mean(pairs))
        print("**AUC %.3f** (0.5 = 무작위, 1.0 = 완전 분리)" % auc)
        thr = float(np.percentile(st, 90))
        det = float(np.mean([a > thr for a in mv]))
        print("정적 오탐 10%% 문턱(%.2f) 에서 이동 검출률 **%.2f**" % (thr, det))
    print("\n물체별:")
    for r in sorted(rows, key=lambda r: -r["drop"])[:12]:
        print("  %-18s %s z %+.2f→%+.2f (하락 %+.2f) 문맥=%s"
              % (r["obj"], r["kind"], r["z_before"], r["z_after"], r["drop"],
                 ",".join(r["ctx"][:3])))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
