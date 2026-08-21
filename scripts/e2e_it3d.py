#!/usr/bin/env python3
"""**엔드투엔드 · IT3DEgo** — SuperMemory 와 같은 지표로, 표본을 20배로.

    $P scripts/e2e_it3d.py --cache <all.npz 들> --ann <annotations>

SuperMemory 엔드투엔드는 답한 것의 정밀도 0.698(다수결 0.649)로 **다수결과 구별이
안 됐다.** 다만 판정 가능한 표본이 100건대고 이동 물체가 6~11건뿐이라 결론을 못 박는다.
IT3DEgo 는 개체 ID·이동 시각이 GT라 **질의 수백 건**을 만들 수 있다.

### 질의 만들기

물체가 위치 L₀ → L₁ → L₂ … 로 옮겨간다. **각 구간의 중간 시각을 질의 시점 T** 로
삼는다(2번째 구간부터 — 그 전에 목격 이력이 있어야 한다).
그 시점의 진짜 위치는 GT 로 안다.

    기록 = T 이전 프레임
    ① 검색   물체구로 마지막 목격 프레임 li 를 찾는다
             **GT 검증**: 그 프레임에 실제로 그 물체 2D bbox 가 있는가
    ② 장소   li 의 latent 가 곧 장소 키다 (IT3DEgo 엔 방 이름이 없다)
             측정용 GT: li 시각에 물체가 있던 location_id
    ③ 재방문 li 와 닮은 프레임이 T 까지 다시 나오는가 (앵커 자기유사도 문턱)
    ④ 부재   거기서 검출이 목격 때 대비 떨어졌는가

    정답 = (그 자리에 있다/없다) 판정이 **T 시점 실제 위치**와 맞는가

⚠️ **부재 층이 필요한 질의와 아닌 질의를 갈라야 한다.** IT3DEgo 는 사람이 물건을
눈앞에서 옮기므로, 새 자리가 **관측된** 경우가 많다. 그러면 검색이 최신 목격을
찾아내 그것만으로 답이 맞는다 — 부재 판정은 할 일이 없다("항상 마지막 본 곳"
다수결이 0.63인 이유가 이것이다).

우리 시나리오("안 보는 사이에 누가 옮겼다")에 해당하는 것은 **새 자리가 아직
관측되지 않은** 질의다. 그 부분집합에서만 부재 층의 값어치가 드러난다.
그래서 `obs_new`(현재 구간에 그 물체 2D bbox 가 있었나)로 갈라 각각 보고한다.

⚠️ SuperMemory 에서 배운 것 두 가지를 그대로 적용한다:
  · 부재 판정에 **고정 문턱 × 최댓값**을 쓰면 안 된다 → 자기 기준 전/후 중앙값 비율
  · **조건②** — 목격 때조차 안 잡히면 답하지 않고 **기권**한다(㉞ 에서 이게 병목)
"""
import argparse, glob, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.it3d_absence import base_label, load_ann              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", required=True, help="*.all.npz 가 있는 디렉터리들")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--anchor-win", type=float, default=0.0,
                    help="앵커 시간창(100ns). 0이면 창 없음. "
                         "⚠️ 60초로 좁혔더니 앵커가 서로 거의 같은 프레임이라 "
                         "자기유사도 문턱이 치솟아 **재방문 없음이 85%%** 가 됐다. "
                         "앵커는 시간창이 아니라 **장소 키**로 뽑아야 한다.")
    ap.add_argument("--anchor-m", type=int, default=8)
    ap.add_argument("--anchor-q", type=float, default=0.70)
    ap.add_argument("--cond2", type=float, default=0.10)
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--bbox-tol", type=float, default=2e6)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--oracle", nargs="*", default=[], choices=["evidence", "place"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = {}
    for d in args.cache:
        for f in sorted(glob.glob(os.path.join(d, "*.all.npz"))):
            files[os.path.basename(f)[:-len(".all.npz")]] = f
    print("영상 %d개" % len(files))
    if not files:
        return
    from scripts.absence_evidence import clip_text

    rows = []
    for vn, f in sorted(files.items()):
        ad = os.path.join(args.ann, vn)
        if not os.path.isdir(ad):
            continue
        z = np.load(f, allow_pickle=True)
        ts, E, S = z["ts"], z["emb"], z["owl"]
        labs, segs, box = load_ann(ad)
        words = [base_label(l) for l in labs]
        dup = {w for w in words if words.count(w) > 1}     # 조건①
        Q = clip_text(words, args.device)

        for oi, segl in segs.items():
            if oi >= len(words) or words[oi] in dup:
                continue
            bts = box.get(oi, np.array([], np.int64))
            if len(bts) == 0:
                continue
            for si in range(1, len(segl)):
                t0, t1, L_now = segl[si]
                T = (t0 + t1) // 2                          # 질의 시점
                rec = np.nonzero(ts <= T)[0]
                if len(rec) < 50:
                    continue
                # ① 검색 — 마지막 목격
                if "evidence" in args.oracle:
                    prev = bts[bts <= segl[si - 1][1]]
                    if len(prev) == 0:
                        continue
                    li = int(rec[np.argmin(np.abs(ts[rec] - prev[-1]))])
                else:
                    sc = E[rec] @ Q[oi] + S[rec, oi]
                    top = rec[np.argsort(-sc)[:args.topk]]
                    li = int(top[np.argmax(ts[top])])
                seen_ok = bool(len(bts) and np.any(np.abs(bts - ts[li]) <= args.bbox_tol))
                # 검색된 순간 물체가 GT 상 있던 자리
                L_at = next((l for a, b, l in segl if a <= ts[li] <= b), None)
                if "place" in args.oracle:
                    L_at = segl[si - 1][2]
                if L_at is None:
                    continue
                truly_here = (L_at == L_now)

                # ③ 재방문 — li 근방을 앵커로 삼아 자기유사도 문턱
                near = rec[ts[rec] <= ts[li]]
                if args.anchor_win > 0:
                    near = near[np.abs(ts[near] - ts[li]) <= args.anchor_win]
                if len(near) < 3:
                    continue
                A = near[np.argsort(-(E[near] @ E[li]))[:args.anchor_m]]
                SS = E[A] @ E[A].T
                iu = np.triu_indices(len(A), 1)
                if len(iu[0]) == 0:
                    continue
                thr = float(np.quantile(SS[iu], args.anchor_q))
                cand = rec[ts[rec] > ts[li]]
                cand = np.setdiff1d(cand, A)
                if len(cand) == 0:
                    state = "b"
                    after = np.array([], int)
                else:
                    sim = np.sort(E[cand] @ E[A].T, 1)[:, -3:].mean(1)
                    after = cand[sim >= thr]
                    state = None
                # ④ 부재 — 자기 기준 전/후
                s_bef = float(np.median(S[A, oi]))
                if state != "b" and len(after) < args.min_frames:
                    state = "b"
                if state != "b":
                    if s_bef < args.cond2:
                        state = "u"
                    else:
                        s_aft = float(np.median(S[after, oi]))
                        state = "c" if s_aft < args.ratio * s_bef else "a"
                # 새 자리가 T 까지 **관측됐는가** — 부재 층이 필요한지 가른다
                obs_new = bool(len(bts) and np.any((bts >= t0) & (bts <= T)))
                rows.append(dict(video=vn, obj=labs[oi], word=words[oi], T=int(T),
                                 obs_new=obs_new,
                                 state=state, seen_ok=seen_ok, truly_here=bool(truly_here),
                                 n_after=int(len(after)), s_before=s_bef,
                                 L_at=int(L_at), L_now=int(L_now)))
        print("  %-20s 누적 질의 %d" % (vn, len(rows)), flush=True)

    n = len(rows)
    if not n:
        print("질의 없음"); return
    ans = [r for r in rows if r["state"] in ("a", "b")]
    ab = [r for r in rows if r["state"] == "u"]
    scored = [r for r in rows if r["state"] != "u"]
    gone = [r for r in scored if not r["truly_here"]]
    prec = sum(r["truly_here"] for r in ans) / len(ans) if ans else float("nan")
    rec_ = sum(1 for r in gone if r["state"] == "c") / len(gone) if gone else float("nan")
    base = sum(r["truly_here"] for r in scored) / len(scored) if scored else float("nan")
    print("\n질의 %d · 영상 %d · 오라클 %s"
          % (n, len({r["video"] for r in rows}), ",".join(args.oracle) or "없음"))
    print("  ① 검색이 실제 목격 프레임을 짚은 비율 %.3f"
          % (sum(r["seen_ok"] for r in rows) / n))
    print("  ③ 재방문 없음(b) %d/%d (%.0f%%)"
          % (sum(1 for r in rows if r["state"] == "b"), n,
             100.0 * sum(1 for r in rows if r["state"] == "b") / n))
    print("  ④ 기권(u) %d/%d (%.0f%%)" % (len(ab), n, 100.0 * len(ab) / n))
    print("  **위치를 답한 %d건의 정밀도 %.3f  (다수결 %.3f)**" % (len(ans), prec, base))
    print("  **실제로 없는 %d건 중 '없다' 로 넘긴 비율(재현) %.3f**" % (len(gone), rec_))
    print("  상태 분포 %s" % dict(Counter(r["state"] for r in rows)))
    print("\n  ── 새 자리가 관측됐는가로 가르면 (부재 층이 필요한 쪽은 '미관측')")
    for tag, sub in (("관측됨", [r for r in rows if r["obs_new"]]),
                     ("**미관측**", [r for r in rows if not r["obs_new"]])):
        a = [r for r in sub if r["state"] in ("a", "b")]
        sc = [r for r in sub if r["state"] != "u"]
        g = [r for r in sc if not r["truly_here"]]
        if not a or not sc:
            continue
        print("    %-10s 질의 %3d · 정밀도 %.3f (다수결 %.3f) · 부재재현 %s · b %d%%"
              % (tag, len(sub), sum(r["truly_here"] for r in a) / len(a),
                 sum(r["truly_here"] for r in sc) / len(sc),
                 ("%.3f" % (sum(1 for r in g if r["state"] == "c") / len(g))) if g else "—",
                 int(100 * sum(1 for r in sub if r["state"] == "b") / len(sub))))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
