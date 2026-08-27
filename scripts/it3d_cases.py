#!/usr/bin/env python3
"""IT3DEgo 실사 — **우리 3경우 문항** 자동 생성·채점. (아이맥, 캐시 재사용)

    python scripts/it3d_cases.py

질문은 언제나 "X 지금 어디?" 하나. GT 로 경우를 가른다:

  경우2  마지막 목격 이후 그 자리를 다시 봤고 물체가 **있음**  → 그 자리 답이 정답
  경우1  다시 봤는데 **없음**(이동함)                        → "부재" 판정이 정답
  경우3  다시 안 봄                                        → 마지막 목격 자리 답이 정답

⚠️ 카메라 궤적이 없으므로 **증인 물체**로 재방문을 판정한다: 옛 자리 0.6m 안에
머문 다른 물체가 그 시각 이후 bbox 로 관측되면 카메라가 그 자리를 다시 본 것.
"""
import glob, os, sys
import numpy as np
from collections import Counter
sys.path.insert(0, ".")
from scripts.it3d_absence import load_ann, base_label

R_NEAR = 0.6
cases = Counter(); rows = []
for f in sorted(glob.glob("data/it3dego/cache_all/*.all.npz")):
    vn = os.path.basename(f)[:-len(".all.npz")]
    ad = os.path.join("data/it3dego/ann/annotations", vn)
    if not os.path.isdir(ad): continue
    z = np.load(f, allow_pickle=True)
    ts, S, E = z["ts"], z["owl"], z["emb"].astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    labs, segs, box = load_ann(ad)
    words = [base_label(l) for l in labs]
    # 3D 구간
    seg3 = {}
    for line in open(os.path.join(ad, "3d_center_annot.txt")):
        p = line.split()
        if len(p) < 7: continue
        seg3.setdefault(int(p[2]), []).append(
            (int(p[0]), int(p[1]), np.array([float(p[3]), float(p[4]), float(p[5])])))
    for oi in seg3: seg3[oi].sort()
    for oi, w in enumerate(words):
        if words.count(w) > 1: continue
        sgs = seg3.get(oi, [])
        bts = np.array(sorted(box.get(oi, [])), dtype=np.int64)
        if not sgs or len(bts) < 3: continue
        moved = len(sgs) >= 2 and np.linalg.norm(sgs[-1][2] - sgs[-2][2]) > R_NEAR
        if moved:
            t_move = sgs[-1][0]; old_pos = sgs[-2][2]
            witness = False
            for oj, sj in seg3.items():
                if oj == oi: continue
                if any(np.linalg.norm(s[2] - old_pos) < R_NEAR and s[1] >= t_move for s in sj):
                    bj = np.array(sorted(box.get(oj, [])), dtype=np.int64)
                    if len(bj) and (bj > t_move).any(): witness = True; break
            case = "1_부재확인" if witness else "3_미확인"
            t_q = ts[-1]
        else:
            half = ts[len(ts)//2]
            case = "2_재확인" if (bts > half).any() else "3_미확인"
            t_q = ts[-1]
        cases[case] += 1
        _ = None
        # ── 우리 시스템 답 ──
        sc = S[:, oi]
        th = np.quantile(sc, 0.95)
        cand = np.where(sc >= th)[0]
        if not len(cand): continue
        recent = sorted(cand, key=lambda i: -ts[i])[:3]
        t_last = ts[max(recent, key=lambda i: ts[i])]
        # ⚠️ 순환 제거: "마지막 목격 이후 점수 하락" 은 정의상 참이다(목격을 점수로
        # 정의했으므로). **자리 게이팅** — 같은 자리를 다시 본 프레임에서만 비교.
        sig = E[recent].mean(0); sig = sig / (np.linalg.norm(sig) + 1e-9)
        simv = E @ sig
        gate = float(np.quantile(simv[recent], .5)) * 0.97
        after = np.where((ts > t_last) & (simv >= gate))[0]
        drop = (float(np.quantile(sc[recent], .9) - np.quantile(sc[after], .9))
                if len(after) >= 4 else None)
        rows.append(dict(case=case, drop=drop, recent=[int(ts[i]) for i in recent],
                         seg_now=(int(sgs[-1][0]), int(sgs[-1][1])),
                         seg_old=(int(sgs[-2][0]), int(sgs[-2][1])) if len(sgs) >= 2 else None))
        continue
        fired = False

print("=== IT3DEgo 3경우 문항 (47영상) · n=%d ===" % len(rows))
print("  구성: " + " · ".join("%s %d" % (c, cases[c]) for c in sorted(cases)))
print("\n  %-7s %-11s %-11s %-11s %s" % ("문턱", "2_재확인", "3_미확인", "1_부재", "전체"))
ds = [r["drop"] for r in rows if r["drop"] is not None]
for th in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 9.0):
    ok2 = Counter(); n2 = Counter()
    for r in rows:
        fired = r["drop"] is not None and r["drop"] > th
        n2[r["case"]] += 1
        if r["case"] == "1_부재확인":
            ok2[r["case"]] += fired
        else:
            lo, hi = r["seg_now"] if r["case"] == "2_재확인" else (r["seg_old"] or r["seg_now"])
            ok2[r["case"]] += (not fired) and any(lo <= t <= hi for t in r["recent"])
    tot = sum(ok2.values()) / max(sum(n2.values()), 1)
    print("  %-7s %-11.3f %-11.3f %-11.3f **%.3f**"
          % ("없음" if th > 8 else "%.2f" % th,
             ok2["2_재확인"]/max(n2["2_재확인"],1), ok2["3_미확인"]/max(n2["3_미확인"],1),
             ok2["1_부재확인"]/max(n2["1_부재확인"],1), tot))
