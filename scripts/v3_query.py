#!/usr/bin/env python3
"""v3 질의 프런트엔드 — 자연어 → Qwen 키워드 추출 → 확장/배제 질의.

    $P scripts/v3_query.py --stage extract      # 질문별 키워드·이웃 JSON 생성(캐시)
    $P scripts/v3_query.py --stage eval         # 자동 키워드로 검색 채점 vs 수동 기준

아키텍처(v3 확정): Qwen3.5-4B 는 두 곳에서만 쓴다 — 앞(키워드 추출)과 뒤(답변).
검색 로직은 우리 코드가 고정 실행한다. 추출은 문법 제약으로 JSON 을 강제하고
thinking 은 끈다(전역 정책).

산출 질의 두 종:
  확장 질의  E = keyword + 이웃(같이 있을 물건·장소)  → 검색 재현율 강화
  배제 질의  C = 이웃만(keyword 제외)                → 부재 증거용(생존 편향 회피)
"""
import argparse
import json
import os
import re
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
SERVER = os.environ.get("LLAMA_SERVER",
                        "http://192.168.219.123:8080/v1/chat/completions")

GRAMMAR = r'''
root ::= "{" ws "\"keyword\"" ws ":" ws str ws "," ws "\"neighbors\"" ws ":" ws arr ws "}"
arr ::= "[" ws str (ws "," ws str){0,4} ws "]"
str ::= "\"" [a-z ]{1,40} "\""
ws ::= [ \n]?
'''

PROMPT = """You are the query planner of a household memory assistant.
From the user's question, extract:
- "keyword": the single object being asked about (1-3 lowercase words)
- "neighbors": 2-5 lowercase words for objects/places it is usually near
  (use hints from the question itself, then common sense)
Question: %s
Respond with JSON only."""


def ask_json(question, timeout=120):
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT % question}],
        "temperature": 0, "max_tokens": 120,
        "grammar": GRAMMAR,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(SERVER, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    return json.loads(txt)


def stage_extract(args):
    from scripts.supermem_answer import questions
    out_p = os.path.join(D, "v3_keywords.json")
    done = json.load(open(out_p)) if os.path.exists(out_p) else {}
    Q = questions()
    for i, (x, _) in enumerate(Q):
        qid = str(x["question_id"])
        if qid in done:
            continue
        try:
            done[qid] = ask_json(x["question"])
        except Exception as e:
            print("Q%s 오류: %s" % (qid, str(e)[:80]))
            continue
        if i % 10 == 0:
            json.dump(done, open(out_p, "w"), ensure_ascii=False, indent=1)
            print("%d/%d 예: %s → %s" % (i, len(Q), x["question"][:40], done[qid]))
    json.dump(done, open(out_p, "w"), ensure_ascii=False, indent=1)
    print("추출 완료 %d건 → %s" % (len(done), out_p))


def stage_eval(args):
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from scripts.supermem_answer import load_index, questions
    kw = json.load(open(os.path.join(D, "v3_keywords.json")))
    E, ts, sid = load_index()
    starts = json.load(open(os.path.join(D, "session_starts.json")))
    abst = np.array([starts[v] + t for v, t in zip(sid, ts)], float)
    Q = [(x, ev) for x, ev in questions() if str(x["question_id"]) in kw]
    qabs = np.array([x["metadata"]["primary_video_start_time"]
                     + (((x.get("question_evidence") or {}).get("time_spans") or [{}])[0]
                        .get("start_time") or 0) for x, _ in Q])
    evabs = [[(starts[v] + a, starts[v] + b) for v, a, b in ev] for _, ev in Q]
    M = abst[None, :] <= qabs[:, None]

    dev = torch.device("mps")
    nm = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(nm)
    txt = CLIPTextModelWithProjection.from_pretrained(nm, use_safetensors=True).eval().to(dev)

    def emb(texts):
        with torch.no_grad():
            tt = tok(texts, padding=True, truncation=True, return_tensors="pt").to(dev)
            return torch.nn.functional.normalize(txt(**tt).text_embeds, dim=-1).cpu().numpy()

    k15 = np.ones(15) / 15

    def score(texts):
        S = emb(texts) @ E.T
        S = S - S.mean(0, keepdims=True)
        S = np.apply_along_axis(lambda r: np.convolve(r, k15, mode="same"), 1, S)
        if getattr(args, "recency_tau", 0) > 0:
            dt = np.clip(qabs[:, None] - abst[None, :], 0, None)
            S = S * np.exp(-dt / (args.recency_tau * 3600.0))
        return np.where(M, S, -np.inf)

    def hits(SS, k=5, tol=30):
        tot = 0
        ol = [0, 0]
        for i, (x, _) in enumerate(Q):
            picked = []
            for j in np.argsort(-SS[i]):
                if not np.isfinite(SS[i][j]):
                    break
                if all(abs(abst[j] - abst[p]) > 30 for p in picked):
                    picked.append(j)
                if len(picked) >= k:
                    break
            ok = any(a - tol <= abst[j] <= b + tol for j in picked for a, b in evabs[i])
            tot += ok
            if x["metadata"]["skill"] == "object_location_memory":
                ol[0] += ok
                ol[1] += 1
        return tot / len(Q), ol[0] / max(ol[1], 1)

    owlZ = None
    if getattr(args, "owl", False):
        # OWLv2 재순위 — CLIP 유사도 대신 **키워드가 실제로 검출된 프레임**을 올린다.
        # CLIP 은 "비슷해 보이는" 프레임을, OWLv2 는 "그 물건이 있는" 프레임을 고른다.
        import re as _re
        from scripts.supermem_answer import SESS
        from scripts.owl_presence import load_owl, owl_z, report_src
        order = []
        for vid, sd in SESS.items():
            z2 = np.load(os.path.join(D, sd, "index.npz"))
            order += [(sd, i) for i in range(len(z2["ts"]))]
        owl = load_owl({sd: os.path.join(D, "owl_sm_%s.json" % sd) for sd in SESS.values()})
        print("OWLv2 지각층: 세션 %d · 검출프레임 %d"
              % (len(owl), sum(len(v) for v in owl.values())))
        def _norm(w):
            t = [y for y in _re.findall(r"[a-z]+", w.lower()) if len(y) > 1]
            return " ".join(t[-2:]) if t else w.lower()
        kwl = [_norm(kw[str(x["question_id"])]["keyword"]) for x, _ in Q]
        owlZ, osrc = owl_z(owl, order, kwl, E=E, device="mps", thr=args.owl_thr)
        report_src(osrc, "검색 키워드")
        owlZ = np.apply_along_axis(lambda r: np.convolve(r, k15, mode="same"), 1, owlZ)

    base = score(["a photo of " + x["question"] for x, _ in Q])
    print("기준(원문 질문):      전체 %.2f · 물체위치 %.2f" % hits(base))
    ext = score(["a photo of " + kw[str(x["question_id"])]["keyword"] + ", "
                 + " ".join(kw[str(x["question_id"])]["neighbors"]) for x, _ in Q])
    print("확장 질의(자동 추출): 전체 %.2f · 물체위치 %.2f" % hits(ext))
    both = np.maximum(base, ext)
    print("원문+확장 최대결합:   전체 %.2f · 물체위치 %.2f" % hits(both))
    if owlZ is not None:
        ow = np.where(M, owlZ, -np.inf)
        print("**OWLv2 단독**:       전체 %.2f · 물체위치 %.2f" % hits(ow))
        for a in (0.3, 0.5, 1.0):
            print("**확장+OWLv2 (a=%.1f)**: 전체 %.2f · 물체위치 %.2f"
                  % ((a,) + hits(np.where(M, ext + a * owlZ, -np.inf))))

        # ── 선별(gate) — 점수로 섞지 말고 **후보를 좁히는 데** 쓴다 ────────────
        # 검색이 찾는 것은 "물건이 보이는 프레임"이 아니라 "그 물건을 거기 둔 순간"
        # 이다. OWLv2 는 재현율이 높아 보이는 프레임을 너무 많이 올린다(그래서
        # 점수 가산은 오히려 나빠졌다). 대신 OWLv2 로 **후보를 거르고 순위는
        # CLIP 에 맡기면**, 높은 재현율이 잡음이 아니라 필터로 쓰인다.
        print("\n[선별] OWLv2 로 후보를 좁히고 순위는 CLIP 확장질의가 정한다")
        for g in (0.5, 1.0, 1.5):
            gated = np.where(owlZ >= g, ext, -np.inf)
            keep = float(np.mean(np.isfinite(gated) & M))
            print("  문턱 z≥%.1f (후보 %4.1f%%): 전체 %.2f · 물체위치 %.2f"
                  % ((g, 100 * keep) + hits(np.where(M, gated, -np.inf))))
        for frac in (0.05, 0.10, 0.25):
            # 질문마다 상위 frac 프레임만 후보로 남긴다(질문별 분위수 — 물체마다
            # 검출 빈도가 달라 절대 문턱은 공평하지 않다)
            thr = np.quantile(owlZ, 1 - frac, axis=1, keepdims=True)
            gated = np.where(owlZ >= thr, ext, -np.inf)
            print("  상위 %4.0f%% 후보:        전체 %.2f · 물체위치 %.2f"
                  % ((100 * frac,) + hits(np.where(M, gated, -np.inf))))
        # 등장 순간 — 검출 점수가 **올라가는** 구간. 물건이 화면에 새로 들어온
        # 시점이 '거기 둔 순간'에 더 가깝다는 가설.
        onset = owlZ - np.roll(owlZ, 30, axis=1)
        onset[:, :30] = 0
        print("  등장 순간(상승 구간):    전체 %.2f · 물체위치 %.2f"
              % hits(np.where(M, ext + 0.5 * onset, -np.inf)))
        print("  등장 순간 단독:          전체 %.2f · 물체위치 %.2f"
              % hits(np.where(M, onset, -np.inf)))

        # ── 방향을 뒤집은 선별 — **CLIP 이 고른 후보 안에서만** OWLv2 로 고른다.
        # 전역 필터가 실패한 이유는 명확하다: OWLv2 는 GT 근거 프레임을 약하게만
        # 표시한다(근거구간 검출률 0.43 vs 전체 0.33). 전역으로 거르면 근거의
        # 57% 를 통째로 버린다. CLIP 의 재현율을 살린 채 순위만 손보면 그 손실이 없다.
        print("\n[역방향 선별] CLIP 상위 N개 후보 안에서 OWLv2 점수로 재정렬")
        for N in (10, 20, 40):
            SS = np.full_like(ext, -np.inf)
            for i in range(ext.shape[0]):
                valid = np.nonzero(np.isfinite(ext[i]) & M[i])[0]
                if len(valid) == 0:
                    continue
                top = valid[np.argsort(-ext[i][valid])[:N]]
                # 후보 안에서 OWLv2 순위로 재배열 (동점은 CLIP 순위 유지)
                SS[i, top] = owlZ[i, top] + 1e-6 * ext[i, top]
            print("  N=%-3d 후보 재정렬:      전체 %.2f · 물체위치 %.2f" % ((N,) + hits(SS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["extract", "eval"])
    ap.add_argument("--recency-tau", type=float, default=0.0,
                    help="최근성 가중 τ(시간) — 점수에 exp(-Δt/τ). 프레임 풀이 커질 때 필수")
    ap.add_argument("--owl", action="store_true",
                    help="검색에 OWLv2 재순위를 얹는다 (data/supermem/owl_sm_*.json)")
    ap.add_argument("--owl-thr", type=float, default=0.0,
                    help="OWLv2 근접 게이트 — 이 점수 미만 검출은 버린다."
                         " Nymeria 실측으로 약한 검출이 위치를 오염시킨다")
    args = ap.parse_args()
    (stage_extract if args.stage == "extract" else stage_eval)(args)


if __name__ == "__main__":
    main()
