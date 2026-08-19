#!/usr/bin/env python3
"""SuperMemory **가구 단위** belief — v3 목표의 2차 수위.

    $P scripts/supermem_belief_furn.py

방 단위는 이 데이터셋에서 막혔다 — 개방형 원룸이라 GT 라벨('Apartment kitchen' /
'Apartment living area')이 사람의 느슨한 기능적 구획이고, 우리 3D 군집(물리 공간)과
어긋난다(순도 0.81, 명명 후 0.51). 그러나 **가구 단위는 다르다**: 정답 선택지의
65%가 "스토브 옆 부엌 서랍", "침실 옷장 문 위 고리" 처럼 **가구를 명시**한다.
개방형이든 아니든 서랍은 서랍이다.

    ① GT      정답 선택지에서 가구 표현 추출(drawer/shelf/bed/cabinet/…)
    ② 후보    같은 가구 어휘를 CLIP 으로 프레임에서 검출 → 후보 집합
    ③ 신호    last-known(마지막 목격 가구) · 부재 게이트(그 가구 보는데 물건 없음)
    ④ 채점    질의 시각 기준 top-1 / top-3

방 단위와 달리 여기서는 **3D 위치가 필요 없다** — 가구는 외형으로 식별된다.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")

FURN = ["drawer", "cabinet", "counter", "kitchen island", "refrigerator", "shelf",
        "dining table", "sofa", "bed", "nightstand", "dresser", "desk", "sink",
        "stove", "microwave", "oven", "rack", "basket", "trash can", "hook",
        "closet", "pantry", "cart", "stand"]
SESS5 = {
    "Person_1_session_1_01312026_glasses_1266": "s1",
    "Person_1_session_8_03102026_glasses_1264": "s8",
    "Person_1_session_14_03152026_glasses_1266": "s14",
    "Person_1_session_19_03292026_glasses_1266sm": "s19",
    "Person_1_session_20_03292026_glasses_1284": "s20",
}


def gt_furn(text):
    """정답 선택지 → 가구 라벨(가장 구체적인 것 우선)."""
    t = text.lower()
    hits = [f for f in FURN if f in t]
    if not hits:
        return None
    return max(hits, key=len)          # 'kitchen island' > 'island'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--z-obj", type=float, default=1.0)
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump", default=None,
                    help="문항별 판정을 JSON 으로 남긴다 — 지각층마다 판정 가능 문항이"
                         " 달라(CLIP 137 vs OWLv2 55) 교집합 비교가 필요하다")
    ap.add_argument("--owl", action="store_true",
                    help="지각층을 CLIP → OWLv2 로 교체 (data/supermem/owl_sm_*.json)."
                         " 어휘 밖 단어는 CLIP 폴백하고 비율을 보고한다")
    ap.add_argument("--furn-synonyms", default=None,
                    help="가구 개념의 표면형 메타데이터. 개념 점수를 표면형 최댓값으로"
                         " 합친다 — 검출을 옮기는 게 아니라 여러 말로 물어본 결과를 합친다")
    ap.add_argument("--owl-thr", type=float, default=0.0,
                    help="OWLv2 근접 게이트 — 이 점수 미만 검출은 버린다."
                         " Nymeria 실측으로 약한 검출이 위치를 오염시킨다")
    args = ap.parse_args()

    from scripts.absence_evidence import clip_text

    Es, tss, sids, order = [], [], [], []
    for vid, sd in SESS5.items():
        f = os.path.join(D, sd, "index.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        Es.append(z["emb"].astype(np.float32))
        tss.extend(z["ts"])
        sids.extend([vid] * len(z["ts"]))
        order += [(sd, i) for i in range(len(z["ts"]))]      # OWLv2 프레임 키와 정렬
    E = np.concatenate(Es)
    ts, sid = np.array(tss), np.array(sids)
    starts = json.load(open(os.path.join(D, "session_starts.json")))
    abst = np.array([starts[v] + t for v, t in zip(sid, ts)], float)
    kwj = json.load(open(os.path.join(D, "v3_keywords.json")))

    owl = None
    if args.owl:
        from scripts.owl_presence import load_owl, owl_z, report_src
        owl = load_owl({sd: os.path.join(D, "owl_sm_%s.json" % sd) for sd in SESS5.values()})
        print("OWLv2 지각층: 세션 %d · 검출프레임 %d"
              % (len(owl), sum(len(v) for v in owl.values())))

    # 가구 후보 검출
    if owl and args.furn_synonyms:
        syn = json.load(open(args.furn_synonyms))
        from scripts.owl_presence import owl_matrix, zscore
        rows = []
        for f in FURN:
            M = owl_matrix(owl, order, syn[f]["surface"], args.owl_thr)
            rows.append(M.max(0))                      # 개념 = 표면형 최댓값
        FZ = zscore(np.stack(rows))
        ns = sum(len(syn[f]["surface"]) for f in FURN)
        print("   가구 표면형 집계: 개념 %d ← 표면형 %d" % (len(FURN), ns))
    elif owl:
        FZ, fsrc = owl_z(owl, order, FURN, E=E, device=args.device, thr=args.owl_thr)
        report_src(fsrc, "가구")
    else:
        FV = clip_text(["a photo of a " + f for f in FURN], args.device)
        FS = FV @ E.T
        FZ = (FS - FS.mean(1, keepdims=True)) / (FS.std(1, keepdims=True) + 1e-9)

    q = json.load(open(os.path.join(D, "qa_person_1.json")))
    Q = []
    for x in q:
        if x["metadata"]["skill"] != "object_location_memory":
            continue
        g = gt_furn(x["choices"][x["correct_option_index"]])
        if g and str(x["question_id"]) in kwj:
            Q.append((x, g))
    if args.limit:
        Q = Q[:args.limit]
    print("가구 단위 질의 %d문항 · GT 분포 %s"
          % (len(Q), dict(Counter(g for _, g in Q).most_common(8))))

    kws = [kwj[str(x["question_id"])]["keyword"] for x, _ in Q]
    if owl:
        import re as _re
        def _norm(w):
            t = [x for x in _re.findall(r"[a-z]+", w.lower()) if len(x) > 1]
            return " ".join(t[-2:]) if t else w.lower()
        KZ, ksrc = owl_z(owl, order, [_norm(k) for k in kws], E=E, device=args.device, thr=args.owl_thr)
        report_src(ksrc, "키워드")
    else:
        KV = clip_text(["a photo of a " + k for k in kws], args.device)
        KS = KV @ E.T
        KZ = (KS - KS.mean(1, keepdims=True)) / (KS.std(1, keepdims=True) + 1e-9)

    r_lk = r_ab = r_pop = 0
    n = 0
    dump = []
    top3_lk = top3_ab = 0
    pop = Counter(g for _, g in Q)
    for i, (x, gtf) in enumerate(Q):
        qabs = x["metadata"]["primary_video_start_time"] + \
            (((x.get("question_evidence") or {}).get("time_spans") or [{}])[0]
             .get("start_time") or 0)
        past = np.nonzero(abst <= qabs)[0]
        if len(past) < 20:
            continue
        det = past[KZ[i, past] >= args.z_obj]
        if len(det) == 0:
            continue
        n += 1
        # ③-A last-known: 물건이 마지막 목격된 프레임에서 가장 강한 가구
        lk_scores = FZ[:, det[-min(len(det), 5):]].mean(1)
        order_lk = np.argsort(-lk_scores)
        # ③-B 부재 게이트: 최근에 그 가구를 봤는데 물건이 없으면 감쇠
        recent = past[-min(len(past), 600):]
        adj = lk_scores.copy()
        for fi in range(len(FURN)):
            vis = recent[FZ[fi, recent] >= args.gate]
            if len(vis) < 5:
                continue
            pk = float(np.median(KZ[i, vis[np.argsort(-FZ[fi, vis])[:10]]]))
            adj[fi] *= args.gamma if pk < 0.3 else 1.0
        order_ab = np.argsort(-adj)
        gi = FURN.index(gtf) if gtf in FURN else -1
        if gi < 0:
            continue
        r_lk += int(order_lk[0] == gi)
        r_ab += int(order_ab[0] == gi)
        top3_lk += int(gi in order_lk[:3])
        top3_ab += int(gi in order_ab[:3])
        r_pop += int(pop.most_common(1)[0][0] == gtf)
        dump.append(dict(q=x["question_id"], gt=gtf,
                         lk1=int(order_lk[0] == gi), lk3=int(gi in order_lk[:3]),
                         ab1=int(order_ab[0] == gi), ab3=int(gi in order_ab[:3])))
    if not n:
        print("판정 가능 문항 없음")
        return
    print("\n판정 가능 %d문항 · 가구 후보 %d개" % (n, len(FURN)))
    print("**가구 단위 belief**            top-1    top-3")
    print("  무작위(%d후보)              %.2f     %.2f" % (len(FURN), 1 / len(FURN), 3 / len(FURN)))
    print("  최빈 가구(%s)          %.2f     —" % (pop.most_common(1)[0][0], r_pop / n))
    print("  **last-known**             %.2f     %.2f" % (r_lk / n, top3_lk / n))
    print("  **+ 부재 게이트**           %.2f     %.2f" % (r_ab / n, top3_ab / n))
    if args.dump:
        json.dump(dump, open(args.dump, "w"), ensure_ascii=False)
        print("→ %s (문항별 판정)" % args.dump)


if __name__ == "__main__":
    main()
