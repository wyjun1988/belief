#!/usr/bin/env python3
"""SuperMemory 부재를 **사람이 단 물체 단위 GT** 로 다시 잰다.

    $P scripts/supermem_absence_gt.py

⚠️ **이 스크립트가 존재하는 이유 — 종전 부재 라벨이 부재가 아니었다.**
`supermem_room_eval.py` ④ 는 물체를 **질문 키워드**(question[:40])로 묶고
"근거 방이 2개 이상이면 이동" 이라고 했다. 실제로 뽑혀 나온 "이동" 사례:

    "I'm trying to remember the sequence of m…"  → bedroom, kitchen
    "I'm trying to retrace my steps for my ch…"  → bedroom, kitchen
    "I need to fix a loose screw on this chai…"  → entrance, kitchen

**어느 것도 물체 이동이 아니다** — 근거가 두 방에 걸친 질문일 뿐이다.
그렇게 나온 AUC 0.688(p=0.037)은 철회한다. Nymeria 도 같은 정의를 썼고
라벨을 고치자 0.640(p=0.023) → 0.615(p=0.120, n=19)로 무너졌다.

**고친 것**:
  ① 물체를 **질문 키워드가 아니라 물체구**로 묶는다. `object_location_memory`
     206문항은 "Where did I leave the <물체>?" 꼴이라 정규식으로 뽑힌다(187/206).
     같은 물체가 여러 문항·여러 세션에 걸쳐 나온다(`yellow toy car` 8회).
  ② 방은 `answer_evidence[].room` — **사람이 단 GT** 다(검출 잡음 없음).
  ③ 대명사성 구(`things`·`everything`·`it`)는 물체가 아니므로 뺀다.

이 라벨의 이동/정적 비율은 **13/18** 로 정상이다(집에서는 대부분 안 움직인다).
종전 정의는 25/6 으로 뒤집혀 있었다 — 그 자체가 라벨이 깨졌다는 신호였다.

**측정은 물체 자신의 전/후 하락으로 한다.** 처음에 "첫 근거 방에서 이후 존재도"
절대값으로 쟀더니 AUC 0.596(p=0.215)이 나왔는데, 표를 보면 이유가 명확했다 —
정적인 `black knife sheath` 가 0.1737 로 **최하위**였다. CLIP 이 그 어휘를 못 잡을
뿐이다. **물체 간 절대 유사도 비교는 부재가 아니라 CLIP 어휘 편향을 재는 것이다**
(S-EMBER 에서 "절대 hit@k 비교 금지" 로 이미 배운 함정).

그래서 물체마다 **자기 자신을 기준**으로 잰다:

    하락 = median(원래 방 · 있던 구간) − median(원래 방 · 떠난 뒤)

이동 물체는 하락이 커야 하고 정적 물체는 0 근처여야 한다. 어휘 편향은 뺄셈으로
상쇄된다. 시점은 세션을 넘는 전역 순서 (세션번호, 초) 로 잡는다 —
s1(01/31) < s8(03/10) < s14(03/15).

⚠️ 같은 시각(±`--tie-sec`)에 두 방이 태그된 근거는 **이동이 아니다**(근거 하나가
두 방으로 중복 표기된 것). `white dash cam box` 가 (1,1056,entrance)·(1,1056,kitchen)
으로 "이동" 이 됐었다. 제외한다.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
from scripts.absence_evidence import clip_text                   # noqa: E402

# 프레임 임베딩을 가진 세션과 그 시간 순서
SESS_NO = {"s1": 1, "s8": 8, "s14": 14}
NORM = {"kitchen": "kitchen", "living": "living_room", "dining": "living_room",
        "bedroom": "bedroom", "closet": "bedroom", "hallway": "entrance",
        "entrance": "entrance", "balcony": "balcony", "bathroom": "bathroom"}
# 물체가 아닌 것 — 대명사·집합명사
STOP = {"it", "them", "things", "everything", "stuff", "these", "those",
        "that", "this", "the rest", "anything", "something"}

PATS = [
    re.compile(r"\bwhere did i (?:leave|put|place|store|set|drop|throw|stash|"
               r"last see|last put|last leave|find)\s+"
               r"(?:the|my|that|this|those|these|a|an)?\s*(.+?)\s*"
               r"(?:\?|$| after| before| when| earlier| last)", re.I),
    re.compile(r"\bwhere i (?:left|put|stored|placed)\s+"
               r"(?:the|my|that|a|an)?\s*(.+?)\s*(?:\?|$|,)", re.I),
    re.compile(r"\bwhere(?:'s| is| are)\s+(?:the|my|that)?\s*(.+?)\s*\?", re.I),
]


def norm_room(r):
    if not r:
        return None
    r = str(r).strip().lower()
    return next((v for k, v in NORM.items() if k in r), None)


def obj_of(q):
    for s in re.split(r"(?<=[.!?])\s+", q):
        for p in PATS:
            m = p.search(s)
            if m:
                o = m.group(1).strip().lower().rstrip("?.,")
                return o if 2 < len(o) < 40 and o not in STOP else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", nargs="+", default=["s1", "s8", "s14"])
    ap.add_argument("--min-frames", type=int, default=5)
    ap.add_argument("--tie-sec", type=float, default=5.0,
                    help="같은 시각으로 볼 허용 오차 — 두 방 중복 표기 판정")
    ap.add_argument("--signal", choices=["clip", "owl"], default="owl",
                    help="존재도 신호 — clip(전체 프레임 latent) / owl(검출 점수)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from scipy.stats import mannwhitneyu

    qs = json.load(open(os.path.join(D, "qa_person_1.json")))
    r3 = json.load(open(os.path.join(D, "rooms3d.json")))

    # ── ① 물체 단위 GT 이력  물체 → [(세션번호, 초, 방)]
    hist = defaultdict(list)
    for x in qs:
        if x["metadata"].get("skill") != "object_location_memory":
            continue
        o = obj_of(x["question"])
        if not o:
            continue
        for e in ((x.get("answer_evidence") or {}).get("evidence_list") or []):
            rm = norm_room(e.get("room"))
            t = (e.get("time_span") or {}).get("start_time")
            sn = re.search(r"session_(\d+)_", e.get("video_id", ""))
            if rm and t is not None and sn:
                hist[o].append((int(sn.group(1)), float(t), rm))
    hist = {o: sorted(set(v)) for o, v in hist.items() if len(set(v)) >= 2}
    print("물체 %d종 (근거 2건 이상)" % len(hist))

    # ── ② 세션별 프레임·방 대응
    per = {}
    for sd in args.sess:
        f = os.path.join(D, sd, "index.npz")
        if sd not in r3 or not os.path.exists(f):
            continue
        z = np.load(f)
        E = z["emb"].astype(np.float32); ts = z["ts"].astype(float)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        lab = np.array(r3[sd]["frame_room"])
        m = min(len(E), len(lab))
        E, ts, lab = E[:m], ts[:m], lab[:m]
        # 군집 L → GT 방 이름: 근거 시각 ±5초에 걸리는 방의 다수결
        c2n = {}
        for L in sorted(set(lab.tolist())):
            near = [rm for v in hist.values() for sn, t, rm in v
                    if sn == SESS_NO[sd]
                    and len(np.nonzero((lab == L) & (np.abs(ts - t) <= 5))[0])]
            if near:
                c2n[L] = Counter(near).most_common(1)[0][0]
        per[sd] = dict(E=E, ts=ts, lab=lab, c2n=c2n)
        print("  %-4s 프레임 %d · 방대응 %s" % (sd, m, c2n))

    names = sorted(hist)

    # ── 존재도 신호
    # ⚠️ **CLIP 전체 프레임 latent 은 물체 부재를 못 잡는다.** 실측: 전/후 하락이
    # ±0.01 로 잡음 수준이고 AUC 0.414(우연 이하)였다. 전체 프레임 벡터는 **방**을
    # 인코딩하지 그 안 물체 하나의 유무를 인코딩하지 않는다. SceneDiff 에서
    # 0.655(p=0.010)가 나온 것도 검출 기반이었다. 그래서 기본값은 owl 이다.
    if args.signal == "clip":
        Q = clip_text(names, args.device)
        for sd, P in per.items():
            P["sc"] = P["E"] @ Q.T                      # (프레임, 물체)
    else:
        # 물체구 → OWL 어휘 매칭: **토큰 경계**로 연속 부분열인 것 중 가장 긴 것.
        # ⚠️ 단순 부분문자열이면 `watering can` 이 `ring` 에 걸린다.
        V = json.load(open(os.path.join(D, "owl_vocab.json")))
        vs = set(V)
        match = {}
        for o in names:
            tk = o.split()
            cand = [" ".join(tk[i:i + n]) for n in range(len(tk), 0, -1)
                    for i in range(len(tk) - n + 1)]
            cand = [c for c in cand if c in vs]
            if cand:
                mx = max(len(c.split()) for c in cand)
                match[o] = [c for c in cand if len(c.split()) == mx]
        print("어휘 매칭 %d/%d" % (len(match), len(names)))
        for sd, P in per.items():
            det = json.load(open(os.path.join(D, "owl_sm_%s.json" % sd)))
            ks = sorted(det)
            n = len(P["ts"])
            if len(ks) < n:
                print("  ⚠️ %s 검출 %d < 프레임 %d" % (sd, len(ks), n))
            M = np.zeros((n, len(names)), np.float32)
            for i in range(min(n, len(ks))):
                d = det[ks[i]]
                for qi, o in enumerate(names):
                    ws = match.get(o)
                    if ws:
                        M[i, qi] = max((d.get(w, 0.0) for w in ws), default=0.0)
            P["sc"] = M
        names = [o for o in names if o in match]
        keep = [i for i, o in enumerate(sorted(hist)) if o in match]
        for sd, P in per.items():
            P["sc"] = P["sc"][:, keep]

    # ── ③ 물체 자신의 전/후 하락
    def frames(r0, lo, hi):
        """원래 방 r0 프레임 중 전역시각이 (lo, hi] 인 것들의 유사도."""
        out = []
        for sd, P in per.items():
            sn = SESS_NO[sd]
            cl = [L for L, nm in P["c2n"].items() if nm == r0]
            if not cl:
                continue
            g = sn * 1e5 + P["ts"]          # 전역 시각 = 세션번호·1e5 + 초
            sel = np.isin(P["lab"], cl) & (g > lo) & (g <= hi)
            idx = np.nonzero(sel)[0]
            if len(idx):
                out.append((sd, idx))
        return out

    mv, st, rows = [], [], []
    for qi, o in enumerate(names):
        v = hist[o]
        s0, t0, r0 = v[0]
        # 같은 시각에 두 방이 찍힌 근거는 중복 표기 — 방을 하나로 못 정하므로 뺀다
        amb = any(a[0] == b[0] and abs(a[1] - b[1]) <= args.tie_sec and a[2] != b[2]
                  for a in v for b in v)
        if amb:
            continue
        rooms = [r for _, _, r in v]
        moved = len(set(rooms)) > 1
        if moved:
            # 원래 방 마지막 근거 → 다른 방 첫 근거
            last = max(s * 1e5 + t for s, t, r in v if r == r0)
            nx = [s * 1e5 + t for s, t, r in v if r != r0 and s * 1e5 + t > last]
            if not nx:
                continue                      # 원래 방이 마지막 — 떠난 뒤가 없다
            pre_hi, post_lo = last, min(nx)
        else:
            mid = v[len(v) // 2]
            pre_hi = post_lo = mid[0] * 1e5 + mid[1]
        LO, HI = -1.0, 1e18

        def med(sp):
            a = [per[sd]["sc"][idx, qi] for sd, idx in sp]
            return (float(np.median(np.concatenate(a))), len(np.concatenate(a))) \
                if a else (None, 0)

        pre, npre = med(frames(r0, LO, pre_hi))
        post, npost = med(frames(r0, post_lo, HI))
        if pre is None or post is None:
            continue
        if npre < args.min_frames or npost < args.min_frames:
            continue
        drop = pre - post
        (mv if moved else st).append(drop)
        rows.append((o, "이동" if moved else "정적", r0, npre, npost,
                     round(pre, 4), round(post, 4), round(drop, 4)))

    print("\n%-24s %-5s %-11s %6s %6s %8s %8s %8s"
          % ("물체", "라벨", "원래 방", "전", "후", "전 존재도", "후 존재도", "하락"))
    for o, tag, r0, a, b, pre, post, drop in sorted(rows, key=lambda r: (r[1], -r[7])):
        print("  %-22s %-5s %-11s %6d %6d %8.4f %8.4f %+8.4f"
              % (o[:22], tag, r0, a, b, pre, post, drop))

    print("\n이동 %d · 정적 %d" % (len(mv), len(st)))
    res = dict(mv=mv, st=st, n_obj=len(hist))
    if len(mv) >= 3 and len(st) >= 3:
        u, p = mannwhitneyu(mv, st, alternative="greater")
        auc = u / (len(mv) * len(st))
        print("**부재 AUC %.3f (p=%.3f)**  — 이동 하락 중앙 %+.4f vs 정적 %+.4f"
              % (auc, p, float(np.median(mv)), float(np.median(st))))
        res.update(auc=auc, p=p)
    else:
        print("표본 부족 — 판정 불가")
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
