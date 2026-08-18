#!/usr/bin/env python3
"""부재 증거를 QA 에 적용 — 선택지 재순위 + 씬그래프 갱신.

    $P scripts/absence_qa.py                       # SuperMemory 물체·위치 문항
    $P scripts/absence_qa.py --update-graph out.json

사용자 아이디어 4단계를 질의응답에 붙인다:

    ① 씬그래프      1fps 스트림의 물체 존재판정 + PMI 문맥 그래프
    ② 확장 질의     질문의 keyword K(찾는 물건) + 선택지가 지목하는 장소 L
    ③ K 제거 검색   L 의 문맥만으로 프레임을 찾는다 — K 가 있든 없든 그 장소가 잡힌다
    ④ 부재 판정     그 프레임들에서 K 의 존재도를 재고, 선택지별로 점수를 매긴다.
                    "가장 최근에 K 가 실제로 보인 장소" 가 답이고, 어디서도 안 보이면
                    부재 이벤트로 기록해 그래프를 갱신한다

핵심은 ③이다. K 를 넣어 검색하면 K 가 보이는 프레임만 올라와(생존 편향) **부재가
관측 불가능**하다. 장소만으로 찾고 그 안에서 K 를 물어야 "그 자리에 없다" 가 보인다.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.absence_evidence import clip_text, pmi_graph, presence   # noqa: E402
from scripts.supermem_answer import load_index, questions            # noqa: E402

D = os.path.join(ROOT, "data", "supermem")
STOP = set("the a an i my me you he she it they them his her their of to in on at was were "
           "is are am did do does done what where who when why how that this these those "
           "after before with for and or as from left right just thinking about need want "
           "get got take took put place placed leave store stored you your".split())


def keyphrase(t, n=4):
    ws = [w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2]
    return " ".join(ws[-n:]) if ws else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--topm", type=int, default=4)
    ap.add_argument("--topf", type=int, default=10)
    ap.add_argument("--ctx-gate", type=float, default=1.0)
    ap.add_argument("--skill", default="object_location_memory")
    ap.add_argument("--owl", action="store_true",
                    help="지각층을 CLIP → OWLv2 로 교체 (data/supermem/owl_sm_*.json)")
    ap.add_argument("--update-graph", default=None)
    args = ap.parse_args()

    E, ts, sid = load_index()
    from scripts.supermem_answer import SESS
    order = []
    for vid, sd in SESS.items():
        z = np.load(os.path.join(D, sd, "index.npz"))
        order += [(sd, i) for i in range(len(z["ts"]))]
    starts = json.load(open(os.path.join(D, "session_starts.json")))
    abst = np.array([starts[s] + t for s, t in zip(sid, ts)], float)
    Q = [(x, ev) for x, ev in questions()
         if x["metadata"]["skill"] == args.skill
         and not x["choices"][x["correct_option_index"]].startswith("This question can not")]
    print("① 씬그래프 구성 — 프레임 %d" % len(E))

    # 어휘: 질문·선택지의 명사 + 장소 어휘
    from scripts.absence_evidence import PLACES
    words = set(PLACES)
    for x, _ in Q:
        for t in [x["question"]] + list(x["choices"]):
            for w in re.findall(r"[a-z]+", t.lower()):
                if w not in STOP and len(w) > 3:
                    words.add(w)
    vocab = sorted(words)
    if args.owl:
        from scripts.owl_presence import load_owl, owl_z, report_src
        owl = load_owl({sd: os.path.join(D, "owl_sm_%s.json" % sd) for sd in SESS.values()})
        print("OWLv2 지각층: 세션 %d · 검출프레임 %d"
              % (len(owl), sum(len(v) for v in owl.values())))
        Z, src = owl_z(owl, order, vocab, E=E, device=args.device)
        report_src(src, "부재QA")
        P = Z > 1.5
    else:
        Z, P = presence(E, vocab, args.device)
    G = pmi_graph(P, vocab)
    vi = {w: i for i, w in enumerate(vocab)}
    print("   어휘 %d · 존재판정 %.1f개/프레임" % (len(vocab), P.sum(0).mean()))

    def q_abs(x):
        """②③④ — 선택지별 (K 존재도, 장소 관측 여부)"""
        qabs = x["metadata"]["primary_video_start_time"] + \
            (((x.get("question_evidence") or {}).get("time_spans") or [{}])[0].get("start_time") or 0)
        past = np.nonzero(abst <= qabs)[0]
        # keyword: 질문의 마지막 명사구 = 찾는 물건
        kws = [w for w in re.findall(r"[a-z]+", x["question"].lower())
               if w in vi and w not in STOP]
        if not kws or len(past) < 10:
            return None
        kv = [vi[w] for w in kws]
        out = []
        for ci, ch in enumerate(x["choices"]):
            # ③ 선택지의 장소 표현만으로 검색 (keyword 는 넣지 않는다)
            lw = [w for w in re.findall(r"[a-z]+", ch.lower())
                  if w in vi and w not in STOP and w not in kws]
            if not lw:
                out.append((ci, None, 0))
                continue
            li = [vi[w] for w in lw]
            cs = Z[li][:, past].max(0)
            ok = past[cs >= args.ctx_gate]
            if len(ok) < 5:
                out.append((ci, None, len(ok)))          # 장소 미관측 → 판정 보류
                continue
            sel = ok[np.argsort(-Z[li][:, ok].max(0))[:args.topf]]
            # ④ 그 장소 프레임에서 keyword 존재도 (여러 단어면 최대)
            pk = float(np.median(Z[kv][:, sel].max(0)))
            # 최근성: 가장 최근 관측 프레임의 시각
            recent = float(abst[sel].max())
            out.append((ci, pk, len(ok), recent))
        return out

    n = ok_abs = ok_rand = 0
    absent_events = []
    for x, _ in Q:
        r = q_abs(x)
        if r is None:
            continue
        cand = [(t[1], t[0], t[3]) for t in r if t[1] is not None]
        if not cand:
            continue
        n += 1
        # 존재도 최고 선택지, 동점이면 더 최근
        best = max(cand, key=lambda t: (t[0], t[2]))
        ok_abs += (best[1] == x["correct_option_index"])
        ok_rand += 1.0 / len(x["choices"])
        if best[0] < 0.5:      # 어디서도 안 보임 → 부재 이벤트
            kw = keyphrase(x["question"], 2)
            absent_events.append(dict(question_id=x["question_id"], keyword=kw,
                                      best_z=best[0], places_checked=len(cand)))
    print("\n② 확장질의 · ③ keyword 제거 검색 · ④ 부재 판정")
    print("판정 가능 문항 %d/%d" % (n, len(Q)))
    if n:
        print("**부재증거 단독 선택 정확도 %.2f** (무작위 %.2f)" % (ok_abs / n, ok_rand / n))
    print("부재 이벤트(어느 후보에서도 관측 안 됨) %d건" % len(absent_events))
    for e in absent_events[:5]:
        print("   Q%-5s '%s' 최고 z %.2f (후보 %d곳 확인)"
              % (e["question_id"], e["keyword"], e["best_z"], e["places_checked"]))

    if args.update_graph:
        # ④ 씬그래프 갱신 산출물 — 부재로 판정된 물체·장소 쌍
        json.dump(dict(absent_events=absent_events, n_scored=n,
                       accuracy=ok_abs / max(n, 1)),
                  open(args.update_graph, "w"), ensure_ascii=False, indent=1)
        print("→ %s" % args.update_graph)


if __name__ == "__main__":
    main()
