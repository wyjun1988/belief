#!/usr/bin/env python3
"""EgoLife 로 **증거 검색**과 **최신 데이터 검색**을 잰다.

    $P scripts/egolife_retrieval.py

부재 판정(㉗)의 앞단이 증거 검색이다. EgoLife QA 는 이걸 재기에 좋다 — 문항마다
**`target_time`(정답 근거 시각)** 이 붙어 있고, **`last_time` 플래그**로 "가장 최근
것을 묻는 질문" 이 표시돼 있다.

  **Ⓐ 증거 검색** — 질의로 검색했을 때 근거 시각이 상위에 오는가 (hit@k)
  **Ⓑ 최신 데이터 검색** — `last_time=True` 문항에서 **가장 최근** 근거를 올리는가.
      "지금 어디 있나" 는 과거 관측을 올리면 틀린 답이 된다.
  **Ⓒ 최근성 가중의 효과** — 검색 점수에 **exp(−Δt/τ) 를 곱한다**(운용 구현과 동일,
      `supermem_answer.py:348`). ⚠️ 뺄셈 페널티로 넣으면 안 된다 — CLIP 유사도가
      0.2~0.3 인데 τ=2h·창 2.8h 면 페널티가 1.4까지 가 **유사도를 압도**한다.
      우리 운용 기본값은 **τ=12h**(SuperMemory 136문항 스윕으로 정함)인데,
      EgoLife 는 시간창이 2.8시간이라 그 τ 가 맞는지 확인해야 한다.

⚠️ `target_time` 은 형식이 **두 가지**다: `{"time": "..."}` 464건 ·
`{"time_list": [...]}` 36건. 게다가 근거가 여럿이면 **이어 붙어 있다**
(`"11153417DAY1_11181201"` = 11:15:34.17 과 11:18:12.01). 8자리씩 끊어 모두 쓴다.
"""
import argparse, json, os, re, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_times(t):
    """'11153417DAY1_11181201' → [40534.17, 40692.01]. list 도 받는다."""
    if isinstance(t, (list, tuple)):
        return [x for e in t for x in parse_times(e)]
    out = []
    for m in re.findall(r"\d{8}", str(t)):
        h, mi, s, cs = int(m[:2]), int(m[2:4]), int(m[4:6]), int(m[6:])
        out.append(h * 3600 + mi * 60 + s + cs / 100.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(ROOT, "data/egolife/index_a1_day1.npz"))
    ap.add_argument("--qa", default=os.path.join(
        ROOT, "data/egolife/EgoLifeQA/EgoLifeQA_A1_JAKE.json"))
    ap.add_argument("--tol", type=float, default=60.0, help="정답 허용 오차(초)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--use", default="keywords", choices=["keywords", "question"],
                    help="질의로 무엇을 쓸지. 실사용은 질문 원문이다")
    ap.add_argument("--answerable", action="store_true",
                    help="**시각만으로 답 가능한 문항만.** EgoLife QA 는 사람 이름을"
                         " 요구하거나(need_name) 오디오가 필요한(need_audio) 문항이"
                         " 대부분이다 — 28문항 중 23개가 need_name, 7개가 need_audio 라"
                         " 시각만으로 답할 수 있는 것은 **2문항**뿐이다."
                         " 우리 층(프레임 검색)이 답할 수 없는 것을 섞으면 층의 성능이"
                         " 아니라 데이터 구성을 재게 된다")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    z = np.load(args.index)
    E = z["emb"].astype(np.float32); ts = z["ts"].astype(float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    qa = json.load(open(args.qa))

    # 근거·질의가 모두 시간창 안인 문항만
    Q = []
    for x in qa:
        if x["query_time"]["date"] != "DAY1" or x["target_time"]["date"] != "DAY1":
            continue
        tt_ = x["target_time"].get("time", x["target_time"].get("time_list"))
        tg = [t for t in parse_times(tt_) if ts.min() <= t <= ts.max()]
        qt = parse_times(x["query_time"].get("time",
                                             x["query_time"].get("time_list")))
        if not tg or not qt or not (ts.min() <= qt[0] <= ts.max()):
            continue
        if args.answerable and (x.get("need_audio") or x.get("need_name")):
            continue
        Q.append((x, qt[0], tg))
    print("문항 %d (근거·질의가 모두 시간창 안) · 프레임 %d · 창 %.1f시간"
          % (len(Q), len(ts), (ts.max() - ts.min()) / 3600))
    if not Q:
        return

    from scripts.absence_evidence import clip_text
    texts = [(x.get("keywords") or x["question"]) if args.use == "keywords"
             else x["question"] for x, _, _ in Q]
    T = clip_text(texts, args.device)

    TAUS = [None, 12.0, 2.0, 0.5, 0.2]
    print("\n%-9s %-7s %-7s %-7s %-9s %s"
          % ("최근성", "hit@1", "hit@5", "hit@10", "최신근거", "인과위반"))
    rows = []
    for tau in TAUS:
        h = {1: [], 5: [], 10: []}
        latest, viol = [], []
        for i, (x, qt, tg) in enumerate(Q):
            # **인과** — 질의 시각 이후 프레임은 볼 수 없다
            m = ts <= qt
            if m.sum() < 10:
                continue
            sim = E[m] @ T[i]
            tt = ts[m]
            if tau is not None:
                sim = sim * np.exp(-(qt - tt) / (tau * 3600.0))
            order = np.argsort(-sim)
            good = np.zeros(m.sum(), bool)
            for g in tg:
                good |= np.abs(tt - g) <= args.tol
            if not good.any():
                continue
            for k in h:
                h[k].append(bool(good[order[:k]].any()))
            # 최신 근거 — 1등이 **마지막 근거**에 드는가
            latest.append(bool(abs(tt[order[0]] - max(tg)) <= args.tol))
            viol.append(False)
        if not h[1]:
            continue
        r = dict(tau=tau, n=len(h[1]),
                 h1=float(np.mean(h[1])), h5=float(np.mean(h[5])),
                 h10=float(np.mean(h[10])), latest=float(np.mean(latest)))
        rows.append(r)
        print("%-9s %-7.3f %-7.3f %-7.3f %-9.3f n=%d"
              % ("없음" if tau is None else "τ=%.1fh" % tau,
                 r["h1"], r["h5"], r["h10"], r["latest"], r["n"]))

    # last_time 문항만 따로 — "가장 최근" 을 명시적으로 묻는 것들
    lt = [i for i, (x, _, _) in enumerate(Q) if x["last_time"]]
    print("\n`last_time=True` 문항 %d개 — **가장 최근 것을 묻는 질문**" % len(lt))
    if len(lt) >= 5:
        for tau in TAUS:
            ok = []
            for i in lt:
                x, qt, tg = Q[i]
                m = ts <= qt
                if m.sum() < 10:
                    continue
                sim = E[m] @ T[i]; tt = ts[m]
                if tau is not None:
                    sim = sim * np.exp(-(qt - tt) / (tau * 3600.0))
                if np.abs(tt - max(tg)).min() > args.tol:
                    continue
                ok.append(bool(abs(tt[np.argmax(sim)] - max(tg)) <= args.tol))
            if ok:
                print("  %-9s 최신근거 적중 %.3f (n=%d)"
                      % ("없음" if tau is None else "τ=%.1fh" % tau,
                         float(np.mean(ok)), len(ok)))

    print("\n우연 기준선: hit@k ≈ k×(2·%.0f초) / 창 = %.4f (k=5)"
          % (args.tol, 5 * 2 * args.tol / (ts.max() - ts.min())))
    print("(대조: SuperMemory 물체·위치 hit@5 **0.75** · τ=12h 기본)")
    if args.out:
        json.dump(rows, open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
