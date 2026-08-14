#!/usr/bin/env python3
"""씬그래프 채점 — GT 타임라인(`gt/objects.json`) 대비.

    $P scripts/eval_scene_graph.py --seq <name> --graph graph_t23.json

재는 것 네 가지:

  1. **위치 오차**  각 배치(placement)의 추정 위치 vs 그 구간 GT 위치 (관측된 물체 전부)
  2. **변화 감지 P/R**  GT 이동 이벤트를 잡았는가 / 없는 이동을 지어냈는가
  3. **감지 지연**  GT 이동이 끝난 뒤 몇 프레임 만에 그래프가 알아챘는가
     — 관측이 끊긴 사이의 이동은 재관측 전까지 알 수 없으므로, 이 값이 곧
        "이 시스템이 belief 를 갱신하는 데 걸리는 시간"이다
  4. **정적 물체 오탐율**  움직이지 않은 물체가 움직였다고 잡히는 비율
     — 뎁스 드리프트가 그래프로 새는지를 직접 재는 지표
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIG = 0.5          # m. 이보다 큰 변화만 '유의미한 이동'으로 채점한다
MATCH_TOL = 0.6    # m. 도착 위치가 이 안에 들어오면 같은 이동으로 본다
LATENCY_MAX = 600  # 프레임. 이보다 늦게 잡히면 미검출


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_t23.json")
    ap.add_argument("--sig", type=float, default=SIG)
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(seq_dir, args.graph)))
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    objs = g["objects"]

    # --- 1. 위치 오차 --------------------------------------------------------
    errs, err_by_kind = [], {"static": [], "dynamic": []}
    for iid, o in objs.items():
        rec = gt.get(iid)
        if rec is None:
            continue
        P = np.array(rec["positions"])
        for pl in o["placements"]:
            mid = (pl["start_frame"] + pl["end_frame"]) // 2
            if mid >= len(P):
                continue
            e = float(np.linalg.norm(np.array(pl["position"]) - P[mid]))
            errs.append(e)
            err_by_kind.setdefault(rec["motion_type"], []).append(e)

    # --- 2/3. 변화 감지 ------------------------------------------------------
    # ⚠️ 원시 `changes` 로 세면 안 된다. 사람이 물건을 들고 걸어가는 동안 그래프는
    # 매 관측마다 새 위치를 보고 변화를 쌓는데, GT 의 이동은 짧은 정지를 병합한
    # **한 건**이다. 세는 단위가 다르면 precision 이 구조적으로 무너진다.
    # 그래서 **정지 배치(stable placement) 사이의 전이**를 그래프의 '이동'으로 센다.
    def graph_moves(o):
        if o is None:
            return []
        st = [p for p in o["placements"] if p.get("stable")]
        if len(st) < 2:
            return []
        out = []
        for a, b in zip(st, st[1:]):
            d = float(np.linalg.norm(np.array(b["position"]) - np.array(a["position"])))
            out.append({"detected_at_frame": b["start_frame"], "from": a["position"],
                        "to": b["position"], "distance_m": d})
        return out

    tp, fn, matched_lat, missed = 0, 0, [], []
    fp, fp_list = 0, []
    for iid, rec in gt.items():
        gmoves = [m for m in rec["moves"] if m["displacement_m"] >= args.sig]
        o = objs.get(iid)
        chs = [c for c in graph_moves(o) if c["distance_m"] >= args.sig]
        used = set()
        for m in gmoves:
            best = None
            for k, c in enumerate(chs):
                if k in used:
                    continue
                if not (m["start_idx"] <= c["detected_at_frame"] <= m["end_idx"] + LATENCY_MAX):
                    continue
                d = np.linalg.norm(np.array(c["to"]) - np.array(m["to"]))
                if d <= MATCH_TOL and (best is None or d < best[1]):
                    best = (k, d)
            if best is None:
                fn += 1
                missed.append((rec["name"], round(m["displacement_m"], 2),
                               o is not None, len(chs)))
            else:
                used.add(best[0])
                tp += 1
                matched_lat.append(chs[best[0]]["detected_at_frame"] - m["end_idx"])
        for k, c in enumerate(chs):
            if k not in used:
                fp += 1
                fp_list.append((rec["name"], rec["motion_type"], round(c["distance_m"], 2)))

    # --- 4. 정적 물체 오탐 ---------------------------------------------------
    stat_ids = [i for i, r in gt.items() if r["motion_type"] == "static" and not r["moves"]]
    obs_stat = [i for i in stat_ids if i in objs]
    def _gm(o):
        st = [p for p in o["placements"] if p.get("stable")]
        return [float(np.linalg.norm(np.array(b["position"]) - np.array(a["position"])))
                for a, b in zip(st, st[1:])]

    bad_stat = [i for i in obs_stat if any(d >= args.sig for d in _gm(objs[i]))]

    # --- 5. belief 정확도 (주 지표) ------------------------------------------
    # "지금 이 물체가 어디 있다고 믿는가" — 이벤트 매칭이 아니라 시간축 전체의 상태를 본다.
    # 그래프의 믿음 = t 를 포함하는 placement, 없으면 **마지막으로 본 자리**(last-known).
    # home-jepa 의 문제 정의와 같고, last-known 자체가 그쪽의 기준 베이스라인이다.
    bel_all, bel_moved, base_moved = [], [], []
    for iid, rec in gt.items():
        o = objs.get(iid)
        if o is None or not rec["moves"]:
            continue
        P = np.array(rec["positions"])
        pls = o["placements"]
        first_move = min(m["start_idx"] for m in rec["moves"])
        for t in range(o["first_frame"], min(len(P), g["n_frames"]), 5):
            bel = None
            for pl in pls:
                if pl["start_frame"] <= t <= pl["end_frame"]:
                    bel = pl["position"]
                    break
                if pl["end_frame"] < t:
                    bel = pl["position"]          # 마지막으로 본 자리로 유지
            if bel is None:
                continue
            e = float(np.linalg.norm(np.array(bel) - P[t]))
            bel_all.append(e)
            if t >= first_move:
                bel_moved.append(e)
                base_moved.append(float(np.linalg.norm(np.array(pls[0]["position"]) - P[t])))

    def _q(v):
        return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                "within_0.5m": float(np.mean(np.array(v) < 0.5)), "n": len(v)} if v else None

    lat = np.array(matched_lat) if matched_lat else np.array([])
    out = {
        "sequence": g["sequence"], "graph": args.graph, "depth_dir": g.get("depth_dir"),
        "n_objects": len(objs), "n_agents": len(g.get("agents", {})),
        "position_err_m": {"median": float(np.median(errs)) if errs else None,
                           "p90": float(np.percentile(errs, 90)) if errs else None,
                           "n": len(errs),
                           "static_median": float(np.median(err_by_kind["static"]))
                           if err_by_kind["static"] else None,
                           "dynamic_median": float(np.median(err_by_kind["dynamic"]))
                           if err_by_kind["dynamic"] else None},
        "change_detection": {"tp": tp, "fn": fn, "fp": fp,
                             "recall": tp / max(tp + fn, 1),
                             "precision": tp / max(tp + fp, 1)},
        "latency_frames": {"median": float(np.median(lat)) if len(lat) else None,
                           "p90": float(np.percentile(lat, 90)) if len(lat) else None,
                           "median_s": float(np.median(lat)) / 10.0 if len(lat) else None},
        "static_false_move": {"observed_static": len(obs_stat), "flagged": len(bad_stat),
                              "rate": len(bad_stat) / max(len(obs_stat), 1)},
        "belief": {"all_frames": _q(bel_all), "after_first_move": _q(bel_moved),
                   "baseline_initial_pos": _q(base_moved)},
        "sig_threshold_m": args.sig,
    }

    print("== %s  (%s)" % (out["sequence"], out["depth_dir"]))
    print("  물체 노드 %d  에이전트 %d" % (out["n_objects"], out["n_agents"]))
    pe = out["position_err_m"]
    print("  위치오차 중앙 %.3f m  p90 %.3f m  (정적 %.3f / 동적 %.3f)"
          % (pe["median"], pe["p90"], pe["static_median"] or -1, pe["dynamic_median"] or -1))
    cd = out["change_detection"]
    print("  변화감지  TP %d  FN %d  FP %d   recall %.2f  precision %.2f"
          % (cd["tp"], cd["fn"], cd["fp"], cd["recall"], cd["precision"]))
    if len(lat):
        print("  감지지연  중앙 %.0f 프레임 (%.1f초)  p90 %.0f" %
              (out["latency_frames"]["median"], out["latency_frames"]["median_s"],
               out["latency_frames"]["p90"]))
    sf = out["static_false_move"]
    print("  정적 오탐  %d/%d = %.3f" % (sf["flagged"], sf["observed_static"], sf["rate"]))
    b = out["belief"]
    if b["after_first_move"]:
        bm, bb = b["after_first_move"], b["baseline_initial_pos"]
        print("  ** belief (이동 이후 프레임, 이동 물체만) **")
        print("     그래프    오차중앙 %.3f m   0.5m 이내 %.3f   (n=%d)"
              % (bm["median"], bm["within_0.5m"], bm["n"]))
        print("     초기위치  오차중앙 %.3f m   0.5m 이내 %.3f   ← 베이스라인"
              % (bb["median"], bb["within_0.5m"]))
    if missed:
        print("  놓친 이동:")
        for n, d, seen, nc in missed:
            print("     %-28s %.2fm  관측됨=%s  그래프변화수=%d" % (n, d, seen, nc))
    if fp_list:
        top = sorted(fp_list, key=lambda x: -x[2])[:8]
        print("  거짓 변화 상위: %s" % ", ".join("%s(%s,%.1fm)" % t for t in top))

    p = os.path.join(seq_dir, args.graph.replace(".json", "_eval.json"))
    json.dump(out, open(p, "w"), indent=1)
    print("→ %s" % p)


if __name__ == "__main__":
    main()
