#!/usr/bin/env python3
"""**엔드투엔드 · 3RScan** — RGB 만으로 세 상태를 답하고 **belief 까지 채점**한다.

    $P scripts/r3scan_e2e.py --root … --cache … --out …

### 장소를 무엇으로 답하나

3RScan 은 스캔당 대개 방 하나라 "어느 방" 이 퇴화한다. 대신 **앵커 물체 기준**으로
답하게 한다 — "안경은 **침대 옆**에 있다". 이건 RGB 만으로 구할 수 있고(㊷ 의 동반
물체 기계), GT 는 `semseg.v2.json` 의 `obb.centroid` 최근접 앵커로 만든다.

    앵커 = 큰 가구 (벽·바닥·천장 제외). 물체의 GT 자리 = 가장 가까운 앵커.

### 세 상태 (참조 스캔 R = 기록, 재방문 S = 다시 봄)

    (a) 있다        S 에서 원래 앵커를 봤고 물체도 여전히 검출됨
    (b) 있을 것이다  S 에서 **원래 앵커 자체를 못 봄** → 확인 불가, 마지막 위치로 답
    (c) 없다        원래 앵커는 봤는데 물체가 없음 → **belief 로 넘긴다**

### belief — 이번에 처음으로 실제로 채점한다

㉟~㊷ 까지 belief 는 **한 번도 평가에 안 들어갔다**(주석에 "belief 로 넘김" 이라고만
적혀 있었다). 여기서는 (c) 일 때 **어디로 갔는지**까지 예측하고 채점한다.

    사전확률 P(앵커라벨 | 물체라벨) 을 **다른 씬들에서** 집계 (leave-one-scene-out)
    예측 = 그 물체가 보통 놓이는 앵커 상위 k
    GT   = 재방문 스캔에서 그 인스턴스의 실제 최근접 앵커 (removed 면 "씬에 없음")

### 채점

  ① 상태 3분류 정답률 (다수결 기준선 병기)
  ② (a)/(b) 일 때 **앵커 정답률** — "어디 있다" 가 맞았나
  ③ (c) 일 때 **belief top-1 / top-3** — "어디로 갔나" 가 맞았나
  ④ 전체 — 질문에 제대로 답한 비율

⚠️ 이번 세션에서 배운 통제를 전부 적용한다: 조건①(라벨 유일 + `ambiguity`),
물체 자기 기준(절대 검출점수 물체 간 비교 금지), 창 대표는 상위 분위수(㊴),
동반 물체 유효성(㊷).
"""
import argparse, json, os
from collections import Counter, defaultdict

import numpy as np

# 앵커에서 뺄 것 — 어디에나 있어 자리를 특정 못 한다
NOT_ANCHOR = {"wall", "floor", "ceiling", "door", "window", "doorframe", "curtain",
              "carpet", "rug", "item", "object", "stuff", "light", "lamp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--wq", type=float, default=0.90)
    ap.add_argument("--anchor-vol", type=float, default=0.15, help="앵커 최소 부피 m^3")
    ap.add_argument("--anchor-max-d", type=float, default=2.5, help="앵커로 인정할 최대 거리 m")
    ap.add_argument("--topf", type=int, default=6, help="물체가 가장 잘 잡힌 프레임 수")
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--topk", type=int, default=3, help="belief top-k")
    ap.add_argument("--anchor-score", default="lift", choices=["mean", "lift"],
                    help="앵커 고르는 법. ⚠️ `mean`(물체가 잘 잡힌 프레임에서 평균 검출)은 "
                         "**가장 큰 가구가 항상 이긴다** — 방 전경에 다 들어오므로 근접성이 "
                         "아니라 검출 용이성을 잰다(첫 실측 앵커 정답 0.223). "
                         "`lift`= 그 물체의 프레임에서의 검출 − 전체 프레임 평균. "
                         "'이 물체와 **특별히** 같이 나오는 앵커' 를 고른다.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vocab = json.load(open(os.path.join(args.root, "vocab.json")))
    vi = {w: i for i, w in enumerate(vocab)}
    meta = json.load(open(os.path.join(args.root, "3RScan.json")))
    sd = os.path.join(args.root, "scans")

    def load(sid):
        p = os.path.join(sd, sid, "semseg.v2.json")
        c = os.path.join(args.cache, sid + ".npz")
        if not (os.path.exists(p) and os.path.exists(c)):
            return None
        try:
            d = json.load(open(p))
        except Exception:
            return None
        ins = {}
        for g in d["segGroups"]:
            ob = g.get("obb") or {}
            ce = ob.get("centroid"); ax = ob.get("axesLengths")
            if not ce or not ax:
                continue
            ins[int(g.get("objectId", g["id"]))] = dict(
                label=g["label"], c=np.array(ce, float), vol=float(np.prod(ax)))
        return dict(ins=ins, owl=np.load(c, allow_pickle=True)["owl"])

    def anchors_of(ins):
        return {i: v for i, v in ins.items()
                if v["vol"] >= args.anchor_vol and v["label"] not in NOT_ANCHOR}

    def gt_anchor(ins, anc, iid):
        """물체의 GT 자리 = 가장 가까운 앵커 라벨."""
        if iid not in ins:
            return None
        c = ins[iid]["c"]
        best, bd = None, 1e9
        for j, v in anc.items():
            if j == iid:
                continue
            d = float(np.linalg.norm(v["c"] - c))
            if d < bd:
                best, bd = v["label"], d
        return best if bd <= args.anchor_max_d else None

    # ── 1차 통과: GT 수집 + belief 사전확률 재료
    pairs = []
    prior_raw = defaultdict(Counter)                 # 물체라벨 → 앵커라벨 빈도 (씬별로 기록)
    scene_of = {}
    for x in meta:
        R = x["reference"]; dr = load(R)
        if dr is None:
            continue
        anc_r = anchors_of(dr["ins"])
        cnt = Counter(v["label"] for v in dr["ins"].values())
        amb = set()
        for grp in (x.get("ambiguity") or []):
            for e in grp:
                amb.add(int(e.get("instance_source", -1)))
                amb.add(int(e.get("instance_target", -1)))
        # belief 사전확률 재료 — 참조 스캔의 (물체 → 앵커) 쌍
        for iid, v in dr["ins"].items():
            a = gt_anchor(dr["ins"], anc_r, iid)
            if a:
                prior_raw[v["label"]][a] += 1
                scene_of.setdefault((v["label"], a), set()).add(R)
        for s in x.get("scans", []):
            S = s["reference"]; ds = load(S)
            if ds is None:
                continue
            def ids(key):
                out = set()
                for v in (s.get(key) or []):
                    try:
                        out.add(int(v["instance_reference"]) if isinstance(v, dict) else int(v))
                    except Exception:
                        pass
                return out
            pairs.append((x, R, dr, anc_r, cnt, amb, S, ds,
                          ids("removed"), ids("rigid"), ids("nonrigid")))
    print("쌍 %d · belief 사전 어휘 %d" % (len(pairs), len(prior_raw)))

    def belief(label, exclude_scene):
        """다른 씬들에서 집계한 P(앵커|물체) 상위 k. ⚠️ leave-one-scene-out."""
        c = Counter()
        for a, n in prior_raw.get(label, {}).items():
            sc = scene_of.get((label, a), set())
            m = n - (1 if exclude_scene in sc else 0)
            if m > 0:
                c[a] = m
        return [a for a, _ in c.most_common(args.topk)]

    rows = []
    for x, R, dr, anc_r, cnt, amb, S, ds, removed, moved, nonrig in pairs:
        anc_s = anchors_of(ds["ins"])
        oR, oS = dr["owl"], ds["owl"]
        for iid, v in dr["ins"].items():
            lb = v["label"]
            if lb not in vi or cnt[lb] > 1 or iid in amb or lb in NOT_ANCHOR:
                continue
            a_gt = gt_anchor(dr["ins"], anc_r, iid)
            if a_gt is None:
                continue
            j = vi[lb]
            sb = float(np.quantile(oR[:, j], args.wq))
            # ── 시스템의 위치 답: 물체가 가장 잘 잡힌 프레임들에서 가장 센 앵커
            top = np.argsort(-oR[:, j])[:args.topf]
            cand = {a["label"] for a in anc_r.values()} & set(vi)
            if not cand:
                continue
            if args.anchor_score == "mean":
                a_pred = max(cand, key=lambda w: float(np.mean(oR[top, vi[w]])))
            else:
                a_pred = max(cand, key=lambda w: float(np.mean(oR[top, vi[w]]))
                             - float(np.mean(oR[:, vi[w]])))
            # ── 재방문에서: 원래 앵커를 봤는가 · 물체가 있는가
            aj = vi[a_pred]
            seen_anchor = float(np.quantile(oS[:, aj], args.wq)) >= \
                0.5 * float(np.quantile(oR[:, aj], args.wq))
            sa = float(np.quantile(oS[:, j], args.wq))
            if not seen_anchor:
                state = "b"
            else:
                state = "c" if sa < args.ratio * sb else "a"
            # ── GT 상태
            kind = ("removed" if iid in removed else "moved" if iid in moved
                    else "nonrigid" if iid in nonrig else "static")
            a_gt_s = gt_anchor(ds["ins"], anc_s, iid) if iid not in removed else None
            gt_state = "c" if (kind == "removed" or (kind == "moved" and a_gt_s != a_gt)) else "a"
            bl = belief(lb, R) if state == "c" else []
            rows.append(dict(ref=R, rescan=S, iid=iid, label=lb, kind=kind,
                             a_gt=a_gt, a_pred=a_pred, a_gt_s=a_gt_s,
                             state=state, gt_state=gt_state,
                             s_before=sb, s_after=sa, belief=bl))

    if not rows:
        print("표본 없음"); return
    n = len(rows)
    print("\n사건 %d · 쌍 %d" % (n, len({(r["ref"], r["rescan"]) for r in rows})))
    print("  GT 종류 %s" % dict(Counter(r["kind"] for r in rows)))
    print("  예측 상태 %s · GT 상태 %s"
          % (dict(Counter(r["state"] for r in rows)), dict(Counter(r["gt_state"] for r in rows))))

    # ① 상태 (b 는 '확인 못함' 이므로 a 로 취급해 채점)
    def coll(s):
        return "c" if s == "c" else "a"
    ok = sum(1 for r in rows if coll(r["state"]) == r["gt_state"])
    base = max(Counter(r["gt_state"] for r in rows).values()) / n
    print("\n① 상태 판정   %.3f (%d/%d) — 다수결 %.3f" % (ok / n, ok, n, base))
    # ② 위치 답 (a/b) 의 앵커 정답률
    ab = [r for r in rows if r["state"] in ("a", "b")]
    if ab:
        hit = sum(1 for r in ab if r["a_pred"] == r["a_gt"])
        # 기준선 — 늘 최빈 앵커라고 답할 때
        mb = Counter(r["a_gt"] for r in ab).most_common(1)[0][1] / len(ab)
        print("② 위치 답 %d건 · 앵커 정답 %.3f — 최빈 앵커 기준선 %.3f · GT 앵커 종류 %d"
              % (len(ab), hit / len(ab), mb, len({r["a_gt"] for r in ab})))
    # ③ belief
    cc = [r for r in rows if r["state"] == "c" and r["a_gt_s"]]
    if cc:
        t1 = sum(1 for r in cc if r["belief"][:1] == [r["a_gt_s"]])
        tk = sum(1 for r in cc if r["a_gt_s"] in r["belief"])
        print("③ belief %d건 · top-1 %.3f · top-%d %.3f" % (len(cc), t1 / len(cc), args.topk, tk / len(cc)))
    # ④ 전체
    full = sum(1 for r in rows
               if (r["gt_state"] == "a" and coll(r["state"]) == "a" and r["a_pred"] == r["a_gt"])
               or (r["gt_state"] == "c" and r["state"] == "c"
                   and (r["a_gt_s"] is None or r["a_gt_s"] in r["belief"])))
    print("④ **전체 정답 %.3f (%d/%d)**" % (full / n, full, n))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
