#!/usr/bin/env python3
"""**방 단위**로 네 가지를 잰다 — 우리 설계 기준(belief 는 방 단위)에 맞춘 측정.

    $P scripts/supermem_room_eval.py

⚠️ **왜 SuperMemory 인가**: 그동안 장소 식별을 **가구 구획** 단위로 쟀는데
(HD-EPIC 의 `counter.005` vs `counter.006`) 우리 기준은 **방**이다.
"노트북 어디 있어" 의 답은 **"부엌"** 이지 "조리대 6번" 이 아니다.

다른 데이터는 방 단위 측정이 **불가능**했다:

| 데이터 | 방 | 판정 |
|---|---|---|
| HD-EPIC | **1개**(전 참가자가 부엌만) | 불가 |
| ADT | 시퀀스당 **1개**(2분간 한 방에만 머묾) | 불가 |
| **SuperMemory** | **6개** · 프레임 고루 분포 | **가능** |

네 가지:
  ① **장소 식별** — 프레임 → 방 (앞 40% 로 키 생성, 뒤에서 채점)
  ② **증거 검색** — 물체 질의 → 그 물체가 있는 방의 프레임이 오는가
  ③ **마지막 목격** — 검색 상위 중 가장 늦은 프레임의 방
  ④ **부재** — 방이 바뀐 물체를 원래 방에서 못 보게 되는가

방 라벨은 `rooms3d.json`(3D 궤적 군집)에서 온다 — 우리가 씬그래프 구축 때 만든 것이라
실사용과 같은 출처다. GT 는 QA 의 `answer_evidence.room`.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
SESS = {"Person_1_session_1_01312026_glasses_1266": "s1",
        "Person_1_session_8_03102026_glasses_1264": "s8",
        "Person_1_session_14_03152026_glasses_1266": "s14"}
NORM = {"kitchen": "kitchen", "apartment kitchen": "kitchen",
        "living room": "living_room", "apartment living area": "living_room",
        "an apartment living area": "living_room", "living area": "living_room",
        "bedroom": "bedroom", "hallway": "entrance", "entrance": "entrance"}


def norm_room(r):
    if not r:
        return None
    r = str(r).strip().lower()
    return NORM.get(r) or next((v for k, v in NORM.items() if k in r), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sess", nargs="+", default=["s1", "s8", "s14"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r3 = json.load(open(os.path.join(D, "rooms3d.json")))
    qs = json.load(open(os.path.join(D, "qa_person_1.json")))
    from scripts.absence_evidence import clip_text

    agg = defaultdict(list)
    for sd in args.sess:
        if sd not in r3:
            continue
        f = os.path.join(D, sd, "index.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        E = z["emb"].astype(np.float32); ts = z["ts"].astype(float)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        lab = np.array(r3[sd]["frame_room"])
        m = min(len(E), len(lab))
        E, ts, lab = E[:m], ts[:m], lab[:m]
        n = len(E); split = int(n * 0.4)

        # ── ① 장소 식별
        keys, kl = [], []
        for L in sorted(set(lab.tolist())):
            idx = np.nonzero((lab == L) & (np.arange(n) < split))[0]
            if len(idx) >= 5:
                v = E[idx].mean(0); v /= np.linalg.norm(v) + 1e-9
                keys.append(v); kl.append(L)
        if len(keys) < 2:
            continue
        K = np.stack(keys)
        te = np.arange(split, n)
        pred = np.array([kl[int(np.argmax(K @ E[i]))] for i in te])
        acc = float((pred == lab[te]).mean())
        maj = Counter(lab[te].tolist()).most_common(1)[0][1] / len(te)
        agg["place"].append((acc, maj, 1.0 / len(kl), len(te)))

        # ── 물체별 GT 방 (QA 근거에서)
        vid = next((v for v, s in SESS.items() if s == sd), None)
        obj = defaultdict(list)                      # 이름 → [(초, 방)]
        for x in qs:
            for e in ((x.get("answer_evidence") or {}).get("evidence_list") or []):
                if SESS.get(e.get("video_id")) != sd:
                    continue
                rm = norm_room(e.get("room"))
                t = (e.get("time_span") or {}).get("start_time")
                kw = (x.get("keywords") or x.get("question", ""))[:40]
                if rm and t is not None and kw:
                    obj[kw].append((float(t), rm))
        if not obj:
            continue
        # 방 군집 → GT 방 이름 대응 (다수결)
        c2n = {}
        for L in kl:
            near = [rm for k_, v_ in obj.items() for t, rm in v_
                    if len(np.nonzero((lab == L) & (np.abs(ts - t) <= 5))[0])]
            if near:
                c2n[L] = Counter(near).most_common(1)[0][0]
        if len(set(c2n.values())) < 2:
            continue

        names = sorted(obj)
        Q = clip_text(names, args.device)
        hit, last_ok = [], []
        for qi, nm in enumerate(names):
            ev = sorted(obj[nm])
            sim = E @ Q[qi]
            top = np.argsort(-sim)[:args.topk]
            gtr = {rm for _, rm in ev}
            # ② 상위 k 중 하나라도 GT 방과 맞는가
            hit.append(any(c2n.get(int(lab[i])) in gtr for i in top))
            # ③ 상위 중 가장 늦은 프레임의 방 = 마지막 근거의 방인가
            last_ok.append(c2n.get(int(lab[int(max(top))])) == ev[-1][1])
        agg["hit"].append(float(np.mean(hit)))
        agg["last"].append(float(np.mean(last_ok)))
        agg["n"].append(len(names))

        # ── ④ 부재: 방이 바뀐 물체 vs 안 바뀐 물체
        for nm in names:
            ev = sorted(obj[nm])
            rms = [r for _, r in ev]
            if len(ev) < 2:
                continue
            sim = E @ Q[names.index(nm)]
            r0 = ev[0][1]
            cl = [L for L, v in c2n.items() if v == r0]
            inr = np.nonzero(np.isin(lab, cl) & (np.arange(n) >= split))[0]
            if len(inr) < 5:
                continue
            v = float(np.median(sim[inr]))
            (agg["mv"] if len(set(rms)) > 1 else agg["st"]).append(v)
        print("  %-4s 방 %d개 · 물체 %d · 대응 %s" % (sd, len(kl), len(names), c2n))

    if agg["place"]:
        a = np.array([x[0] for x in agg["place"]]); mj = np.array([x[1] for x in agg["place"]])
        ch = np.array([x[2] for x in agg["place"]])
        print("\n① **장소 식별(방)** %.3f · 최빈방 %.3f · 우연 %.3f"
              % (a.mean(), mj.mean(), ch.mean()))
    if agg["hit"]:
        print("② **증거 검색(방 일치)** hit@%d %.3f (물체 %d)"
              % (args.topk, float(np.mean(agg["hit"])), int(np.sum(agg["n"]))))
        print("③ **마지막 목격(방)** %.3f" % float(np.mean(agg["last"])))
    if len(agg["mv"]) >= 3 and len(agg["st"]) >= 3:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(agg["st"], agg["mv"], alternative="greater")
        print("④ **부재(방 이탈)** 이동 %d · 정적 %d · AUC **%.3f** (p=%.3f)"
              % (len(agg["mv"]), len(agg["st"]),
                 u / (len(agg["mv"]) * len(agg["st"])), p))
    else:
        print("④ 부재 — 표본 부족(이동 %d · 정적 %d)" % (len(agg["mv"]), len(agg["st"])))
    if args.out:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in agg.items()}, open(args.out, "w"), default=float)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
