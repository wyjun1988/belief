#!/usr/bin/env python3
"""**엔드투엔드** — "내 안경 어디 뒀지?" 를 층을 연결해 한 번에 답한다.

    $P scripts/e2e_supermem.py --sessions s1 s8 s14 s19 s20

지금까지는 층을 **따로** 쟀다(장소식별 0.900 · 검색 0.863 · 마지막목격 0.774 ·
부재 0.714). 그건 각각의 숫자지 **연결한 결과가 아니다.** 층을 이으면 오류가
곱해지고, 어느 층이 진짜 병목인지는 이어봐야 안다.

### 답해야 하는 세 상태

    (a) 있다      장소를 다시 봤고 물체가 그대로 → "거실 탁자에 있다"
    (b) 있을 것이다 장소를 다시 안 봤다        → "거실 탁자에 있을 것이다"
    (c) 없다      다시 봤는데 물체가 없다      → "거기엔 없다" (belief 로 넘김)

### 파이프라인

    ① 검색      물체구 → 기록에서 **마지막 목격** 프레임
    ② 장소식별   그 프레임의 방 (CLIP latent 방 키)
    ③ 재방문     그 시각 **이후** 그 방 프레임이 있는가
    ④ 부재       있다면 거기서 물체가 검출되는가 (OWL)

### 오라클 치환

각 층을 GT 로 하나씩 바꿔 최종 정답이 얼마나 오르는지 본다. **그 차이가 곧 그 층의
병목 크기**다. 층별 정확도만 보면 오류가 어디서 증폭되는지 알 수 없다.

### ⚠️ 3상태 정답률을 그대로 읽으면 안 된다

3세션 첫 실측에서 GT 분포가 **(a)40 · (c)4 · (b)0** 이었다. "항상 (a)" 만 해도
0.909 라 우리 0.864 보다 **높다.** 방이 4개뿐이고 세션이 42일에 걸쳐 있어 마지막
목격 방을 거의 항상 다시 보게 되고((b)가 0), 집에서 물건은 대개 안 움직인다((c)가 희소).

그래서 **질의 시점을 세션 끝마다** 두어 표본을 늘리고, **다수결 기준선을 항상 같이**
찍는다. 그리고 판정을 이렇게 정리한다 — 세 상태는 사실 **존재 여부 + 헤지**다:

    (a)/(b) = "그 방에 있다"(확인함 / 못 확인함)      (c) = "그 방에 없다"
    정답 = 이 판정이 시각 T 의 **실제 위치**와 맞는가

움직인 물체와 안 움직인 물체를 **갈라서** 본다 — 안 움직인 쪽만 보면 항상 (a)가
이기므로 층의 성능이 아니라 데이터 편향을 재게 된다(SceneDiff 의 Removed/Moved 와 같은 이유).

### GT

`object_location_memory` 206문항에서 물체구를 뽑고(㉜), 방은 `answer_evidence[].room`
(사람이 단 것). 물체의 **전 세션 마지막 근거**가 진짜 최종 위치다.

    (a) 최종 방 == 마지막 목격 방 이고 재방문 있음
    (c) 최종 방 != 마지막 목격 방 이고 재방문 있음
    (b) 재방문 없음

⚠️ 물체를 **질문 키워드로 묶으면 안 된다**(㉜ 에서 물린 여섯 번째 오류).
⚠️ 물체 간 **절대 검출점수를 비교하지 않는다** — 각 물체의 자기 기준으로만 본다.
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
from scripts.absence_evidence import clip_text                   # noqa: E402
from scripts.supermem_absence_gt import obj_of, norm_room        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["s1", "s8", "s14", "s19", "s20"])
    ap.add_argument("--topk", type=int, default=5, help="검색 상위 k 중 가장 늦은 것")
    ap.add_argument("--tau-h", type=float, default=0.0,
                    help="최근성 τ(시간). 0이면 끔. 기록 길이의 1/10 이 경험칙")
    ap.add_argument("--cond2", type=float, default=0.10,
                    help="조건② 절대문턱(--calib 이면 안 씀)")
    ap.add_argument("--calib", action="store_true",
                    help="**물체마다 자기 분포로 문턱을 잡는다.** ⚠️ 절대문턱 0.10 은 "
                         "잡음 바닥보다 낮았다 — IT3DEgo GT bbox 대조 실측: 물체가 "
                         "**안 보이는** 프레임의 검출도 중앙이 0.152 이고 그 61%가 "
                         "0.10 을 넘는다(보일 때는 0.380). 그래서 '없다' 를 말할 수가 "
                         "없었다. 검출기 자체는 멀쩡하다(AUC 0.812). "
                         "대신 그 물체의 **전 기록 분위수**를 기준으로 잡는다 — "
                         "물체는 대개 소수 프레임에만 보이므로 중앙값이 곧 잡음 바닥이다.")
    ap.add_argument("--calib-hi", type=float, default=0.90)
    ap.add_argument("--min-age-h", type=float, default=0.0,
                    help="**부재 층을 언제 부를지 정하는 문지기.** 마지막 목격이 "
                         "이보다 최근이면 부재 검사를 건너뛰고 '있다' 로 답한다. "
                         "⚠️ 실측 근거: 새 위치가 관측된 질의(n=132)에서는 부재 층이 "
                         "오히려 해로웠다(정밀도 0.591 vs 다수결 0.636) — 방금 본 것을 "
                         "'없다' 로 뒤집는 오경보 때문이다. 반대로 미관측 질의에서는 "
                         "0.929 vs 0.737 로 이득이 컸다. 시스템은 '관측됐는지' 를 직접 "
                         "알 수 없지만 **목격이 얼마나 오래됐는지** 는 안다.")
    ap.add_argument("--ratio", type=float, default=0.6,
                    help="재방문 검출도가 목격 때의 이 비율 미만이면 '없다'")
    ap.add_argument("--evidence", choices=["clip", "owl", "both"], default="both")
    ap.add_argument("--oracle", nargs="*", default=[],
                    choices=["evidence", "room"])
    ap.add_argument("--multi-t", action="store_true",
                    help="세션 끝마다 질의한다 — 표본을 늘리고 (b) 상태를 만든다")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    qs = json.load(open(os.path.join(D, "qa_person_1.json")))
    r3 = json.load(open(os.path.join(D, "rooms3d.json")))
    starts = {int(re.search(r"session_(\d+)_", k).group(1)): v
              for k, v in json.load(open(os.path.join(D, "session_starts.json"))).items()}
    vocab = json.load(open(os.path.join(D, "owl_vocab.json")))
    vs = set(vocab); vi = {w: i for i, w in enumerate(vocab)}

    # ── 기록 적재 — 5세션을 **전역 시간축**으로 잇는다
    F = []                                   # (전역초, 세션번호, 세션내초, 프레임idx)
    E, DET, GEO = [], [], []
    for sd in args.sessions:
        n = int(sd[1:])
        z = np.load(os.path.join(D, sd, "index.npz"))
        e = z["emb"].astype(np.float32); e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        det = json.load(open(os.path.join(D, "owl_sm_%s.json" % sd)))
        ks = sorted(det)
        lab = np.array(r3[sd]["frame_room"]) if sd in r3 else None
        m = min(len(e), len(ks), len(lab) if lab is not None else len(e))
        M = np.zeros((m, len(vocab)), np.float32)
        for i in range(m):
            for w, v in det[ks[i]].items():
                j = vi.get(w)
                if j is not None:
                    M[i, j] = v
        for i in range(m):
            F.append((starts[n] + float(z["ts"][i]), n, float(z["ts"][i])))
        E.append(e[:m]); DET.append(M)
        GEO.append(lab[:m] if lab is not None else np.full(m, -1))
    E = np.concatenate(E); DET = np.concatenate(DET); GEO = np.concatenate(GEO)
    gt_ = np.array([f[0] for f in F]); sess_ = np.array([f[1] for f in F])
    print("기록 프레임 %d · 세션 %s · 기간 %.1f일"
          % (len(E), ",".join(args.sessions), (gt_.max() - gt_.min()) / 86400))

    # ── 물체 단위 GT 이력 (전 20세션)
    hist = defaultdict(list)
    ansable = {}
    for x in qs:
        if x["metadata"].get("skill") != "object_location_memory":
            continue
        o = obj_of(x["question"])
        if not o:
            continue
        ansable.setdefault(o, []).append(bool(x["is_answerable"]))
        for ev in ((x.get("answer_evidence") or {}).get("evidence_list") or []):
            rm = norm_room(ev.get("room"))
            t = (ev.get("time_span") or {}).get("start_time")
            sn = re.search(r"session_(\d+)_", ev.get("video_id", ""))
            if rm and t is not None and sn:
                n = int(sn.group(1))
                hist[o].append((starts[n] + float(t), n, float(t), rm))
    hist = {o: sorted(set(v)) for o, v in hist.items()}
    print("물체 %d종 (근거 있는 것)" % len(hist))

    # ── 방 키: 라벨 있는 세션(geo)에서 만들고, 없는 세션은 최근접 키로 배정
    #    ⚠️ 군집 번호는 세션마다 다르다 → QA 근거로 **이름**에 대응시킨다
    lab_names = np.full(len(E), None, object)
    for sd in args.sessions:
        if sd not in r3:
            continue
        n = int(sd[1:])
        sel = np.nonzero(sess_ == n)[0]
        c2n = {}
        for L in sorted(set(GEO[sel].tolist())):
            near = [rm for v in hist.values() for gt, sn, t, rm in v
                    if sn == n and len(np.nonzero((GEO[sel] == L)
                                                  & (np.abs(gt_[sel] - gt) <= 5))[0])]
            if near:
                c2n[L] = Counter(near).most_common(1)[0][0]
        for i in sel:
            lab_names[i] = c2n.get(int(GEO[i]))
    known = np.array([i for i in range(len(E)) if lab_names[i]])
    names_all = sorted({lab_names[i] for i in known})
    K = np.stack([E[[i for i in known if lab_names[i] == r]].mean(0) for r in names_all])
    K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-9
    pred_room = np.array([names_all[int(np.argmax(K @ E[i]))] for i in range(len(E))])
    acc = float(np.mean([pred_room[i] == lab_names[i] for i in known]))
    maj = Counter([lab_names[i] for i in known]).most_common(1)[0][1] / len(known)
    print("방 키 %d개 %s · 라벨 프레임 %d · 방 식별 %.3f (최빈 %.3f)"
          % (len(names_all), names_all, len(known), acc, maj))

    # ── 물체구 → OWL 어휘 (토큰 경계 최장일치)
    def match(o):
        tk = o.split()
        c = [" ".join(tk[i:i + n]) for n in range(len(tk), 0, -1)
             for i in range(len(tk) - n + 1)]
        c = [x for x in c if x in vs]
        if not c:
            return []
        mx = max(len(x.split()) for x in c)
        return [vi[x] for x in c if len(x.split()) == mx]

    objs = [o for o in sorted(hist) if match(o)]
    Q = clip_text(objs, args.device)
    NOW = gt_.max()
    tau = args.tau_h * 3600 if args.tau_h > 0 else None

    # 질의 시점 — 각 세션의 마지막 프레임(그 시점까지가 기록)
    TS = sorted({float(gt_[sess_ == n].max()) for n in set(sess_.tolist())}) \
        if args.multi_t else [float(gt_.max())]

    rows = []
    for T in TS:
        rec = np.nonzero(gt_ <= T)[0]
        if len(rec) < 100:
            continue
        for qi, o in enumerate(objs):
            H = [h for h in hist[o] if h[0] <= T]
            inrec = [h for h in H if h[1] in [int(x[1:]) for x in args.sessions]]
            if len(inrec) < 1:
                continue
            gt_last = inrec[-1]                    # 기록 안 마지막 목격 (GT)
            r_true = H[-1][3]                      # **시각 T 의 실제 위치**
            # ⚠️ 부재 층이 필요한 질의와 아닌 질의를 갈라야 한다. 최종 근거가
            # **기록 안 세션**에 있으면 새 위치가 관측된 것이라 검색만으로 답이 맞는다.
            # 우리 시나리오("안 보는 사이에 옮겨졌다")는 최종 근거가 **기록 밖**인 쪽이다.
            obs_new = (H[-1][1] in [int(x[1:]) for x in args.sessions])
            moved = r_true != gt_last[3]

            # ① 검색 — 마지막 목격 프레임
            if "evidence" in args.oracle:
                li = int(rec[np.argmin(np.abs(gt_[rec] - gt_last[0]))])
            else:
                sc = np.zeros(len(rec), np.float32)
                if args.evidence in ("clip", "both"):
                    sc += E[rec] @ Q[qi]
                if args.evidence in ("owl", "both"):
                    sc += DET[np.ix_(rec, match(o))].max(1)
                if tau:
                    sc *= np.exp(-(T - gt_[rec]) / tau)
                top = rec[np.argsort(-sc)[:args.topk]]
                li = int(top[np.argmax(gt_[top])])

            # ② 장소 식별
            r_pred = (lab_names[li] or pred_room[li]) if "room" in args.oracle \
                else pred_room[li]

            # ③ 재방문 — 그 방을 그 시각 이후 T 까지 다시 봤는가
            rm_of = lab_names if "room" in args.oracle else pred_room
            after = np.array([i for i in rec if gt_[i] > gt_[li] and rm_of[i] == r_pred])

            # ④ 부재 — ⚠️ **고정 문턱에 최댓값을 쓰면 안 된다.** 재방문 프레임이
            # 수백 장이면 max 는 오검출로 늘 문턱을 넘어 첫 실측에서 214/218 이
            # "있다" 로 붙었다(다수결과 무구별). 검증된 형태로 간다 —
            # **물체 자기 기준의 전/후 하락**(㉜)과 **조건②**(㉓·㉔·㉞).
            mi = match(o)
            # 물체 자기 분포 — 중앙 = 잡음 바닥, 상위분위 = 보일 때 수준
            allsc = DET[np.ix_(rec, mi)].max(1)
            nfloor = float(np.median(allsc))
            nhi = float(np.quantile(allsc, args.calib_hi))
            befw = np.array([i for i in rec if rm_of[i] == r_pred
                             and gt_[li] - 900 <= gt_[i] <= gt_[li]])
            s_bef = float(np.median(DET[np.ix_(befw, mi)].max(1))) if len(befw) else 0.0
            # ⚠️ `detect` 오라클은 만들지 않는다 — 부재 판정이 **곧 답**이라
            # GT 를 넣으면 구성상 1.000 이 나오는 순환이다(실측으로 확인).
            age_h = (T - gt_[li]) / 3600.0
            if len(after) == 0:
                state = "b"
            elif age_h < args.min_age_h:
                state = "a"                       # 방금 봤다 — 부재 검사를 안 부른다
            elif args.calib:
                # 조건② — 그 물체가 기록 안에서 잡음 바닥 위로 올라오는 일이 있나
                if nhi - nfloor < 0.05:
                    state = "u"
                else:
                    s_aft = float(np.median(DET[np.ix_(after, mi)].max(1)))
                    # 목격 창이 잡음 바닥에 붙어 있으면 그 창 자체가 못 믿을 것
                    if s_bef < nfloor + 0.05:
                        state = "u"
                    else:
                        state = "c" if s_aft < nfloor + args.ratio * (s_bef - nfloor) else "a"
            elif s_bef < args.cond2:
                state = "u"                       # 지각이 못 본다 → 기권
            else:
                s_aft = float(np.median(DET[np.ix_(after, mi)].max(1)))
                state = "c" if s_aft < args.ratio * s_bef else "a"

            # ── 후보 **순위** — "어디를 먼저 가볼까"
            # ⚠️ 이진 판정(있다/없다)은 오경보를 과대평가한다. 실제로는 사용자가
            # 그 자리에서 못 찾고 묻는 경우가 많고, 시스템도 한 곳만 말할 필요가 없다.
            # 2순위는 **그 물체가 기록에서 두 번째로 많이 잡힌 방**(시스템이 계산 가능).
            byroom = {}
            for rn in names_all:
                ix = [i for i in rec if rm_of[i] == rn]
                byroom[rn] = float(np.median(DET[np.ix_(ix, mi)].max(1))) if ix else 0.0
            alt = sorted((v, k) for k, v in byroom.items() if k != r_pred)
            second = alt[-1][1] if alt else r_pred
            rank_off = [r_pred, second]                       # 부재 층 끔
            rank_on = ([second, r_pred] if state == "c" else [r_pred, second])
            says_here = state in ("a", "b")
            truly_here = (r_true == r_pred)
            rows.append(dict(T=T, obj=o, r_pred=r_pred, r_gt=gt_last[3], r_true=r_true,
                             obs_new=bool(obs_new),
                             state=state, moved=bool(moved), n_after=int(len(after)),
                             says_here=says_here, truly_here=bool(truly_here),
                             ok=bool(says_here == truly_here),
                             top1_off=bool(rank_off[0] == r_true),
                             top2_off=bool(r_true in rank_off),
                             top1_on=bool(rank_on[0] == r_true),
                             top2_on=bool(r_true in rank_on),
                             t_err_h=abs(gt_[li] - gt_last[0]) / 3600.0))

    n = len(rows)
    if not n:
        print("질의 없음"); return
    room_ok = sum(r["r_pred"] == r["r_gt"] for r in rows)
    ans = [r for r in rows if r["state"] in ("a", "b")]      # 위치를 답한 것
    absn = [r for r in rows if r["state"] == "c"]            # 없다고 넘긴 것
    ab = [r for r in rows if r["state"] == "u"]              # 기권
    # ⚠️ 판정을 **정밀도·재현율**로 가른다. 종전처럼 "says_here == truly_here" 로
    # 한 덩어리 정확도를 내면, **방을 틀렸는데 '없다'고 해서 맞은** 경우까지
    # 정답으로 세어 성능이 부풀려진다.
    prec = sum(r["truly_here"] for r in ans) / len(ans) if ans else float("nan")
    truly_gone = [r for r in rows if not r["truly_here"] and r["state"] != "u"]
    rec = sum(1 for r in truly_gone if r["state"] == "c") / len(truly_gone) \
        if truly_gone else float("nan")
    # 다수결: 항상 "있다" 라고 답할 때의 정밀도
    scored = [r for r in rows if r["state"] != "u"]
    base = sum(r["truly_here"] for r in scored) / len(scored) if scored else float("nan")
    print("\n질의 %d (시점 %d개) · 오라클 %s"
          % (n, len(TS), ",".join(args.oracle) or "없음"))
    print("  ① 시각 오차     중앙 %.1f시간" % np.median([r["t_err_h"] for r in rows]))
    print("  ② 방 식별       %.3f (%d/%d)" % (room_ok / n, room_ok, n))
    print("  ④ 기권(u)       %d/%d (%.0f%%) — 목격 때조차 검출 %.2f 미만"
          % (len(ab), n, 100.0 * len(ab) / n, args.cond2))
    print("  **위치를 답한 %d건의 정밀도 %.3f  (다수결 %.3f)**" % (len(ans), prec, base))
    print("  **실제로 없는 %d건 중 '없다' 로 넘긴 비율(재현) %.3f**" % (len(truly_gone), rec))
    print("  상태 분포 %s" % dict(Counter(r["state"] for r in rows)))
    # ── "어디를 먼저 가볼까" 순위로 보면
    sc3 = [r for r in rows if r["state"] != "u"]
    if sc3:
        print("\n  ── 순위로 보면 (기권 제외 %d건)" % len(sc3))
        print("    %-16s %8s %8s" % ("", "1순위", "2순위 안"))
        for tag, a, b in (("부재 층 끔", "top1_off", "top2_off"),
                          ("**부재 층 켬**", "top1_on", "top2_on")):
            print("    %-16s %8.3f %8.3f"
                  % (tag, np.mean([r[a] for r in sc3]), np.mean([r[b] for r in sc3])))
    print("\n  ── 새 위치가 기록에 관측됐는가로 가르면 (부재 층이 필요한 쪽은 '미관측')")
    for tag, sub in (("관측됨", [r for r in rows if r["obs_new"]]),
                     ("**미관측**", [r for r in rows if not r["obs_new"]])):
        a = [r for r in sub if r["state"] in ("a", "b")]
        sc2 = [r for r in sub if r["state"] != "u"]
        g = [r for r in sc2 if not r["truly_here"]]
        if not a or not sc2:
            print("    %-10s 질의 %d — 판정 표본 부족" % (tag, len(sub))); continue
        print("    %-10s 질의 %3d · 정밀도 %.3f (다수결 %.3f) · 부재재현 %s"
              % (tag, len(sub), sum(r["truly_here"] for r in a) / len(a),
                 sum(r["truly_here"] for r in sc2) / len(sc2),
                 ("%.3f" % (sum(1 for r in g if r["state"] == "c") / len(g))) if g else "—"))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
