"""home-jepa 의 belief 질의를 우리 씬그래프로 답한다.

home-jepa 의 ADT 변환(`homejepa/adt.py`)은 물체 위치를 **GT 좌표 → 가장 가까운 가구
(receptacle, 1.6m 이내, 히스테리시스 0.10m)** 로 이산화해서 "지금 이 물체가 어느 표면
위에 있나"를 묻는다. 지금까지 그 입력은 전부 모캡 GT 였다.

여기서 묻는 것은 하나다: **GT 대신 우리 지각 파이프라인을 넣어도 답이 유지되는가.**
그래서 같은 규칙을 두 번 적용한다 —

    A) GT 물체 위치 + GT 가구 위치        (home-jepa 의 현행 입력)
    B) 그래프 belief 위치 + 그래프 가구 위치  (우리 지각)

그리고 A 를 정답으로 B 를 채점한다. 덤으로 '초기 위치 고정' 베이스라인도 같이 낸다
(움직임을 전혀 모르는 시스템의 성능 = 이 문제의 바닥).

주의: 우리 그래프는 **예측하지 않는다**. 안 보이는 동안의 이동은 재관측 전까지 모르고,
그때까지 마지막으로 본 자리를 유지한다. 그 '모르는 구간'을 메우는 것이 home-jepa 모델의
일이고, 이 모듈은 그 모델이 딛고 설 지각 층이 쓸 만한지를 잰다.
"""
import numpy as np

NEAR_MAX = 1.6      # m — homejepa/adt.py 와 같은 값
HYST = 0.10
FURN_CATS = {       # homejepa/adt.py 의 FURN_CATS 를 그대로 옮김
    "couch": "sofa", "armchair": "sofa", "dining table": "dining_table",
    "table": "dining_table", "coffee table": "coffee_table",
    "console table": "console", "side table": "nightstand",
    "tv stand": "tv_stand", "bed frame": "bed", "dressing table": "desk",
    "cabinet": "cabinet_top", "cabinets and shelves": "cabinet_top",
    "shelf": "shelf", "bar stool": "shelf", "chair": "shelf",
    "dining chair": "shelf",
}


def receptacles_from(objects, gt, use_gt_pos, cap=28, furn_pos="position"):
    """(ids, positions, names) — 후보 가구 목록.

    home-jepa 는 '동적 물체를 자주 얹은 순'으로 28개를 고르지만, 여기서는 시퀀스
    하나를 보는 것이라 **크기 순**으로 고른다(같은 종류의 안정적 표면 집합이면 된다).
    """
    cand = []
    for iid, o in objects.items():
        cat = (o.get("category") or "").lower()
        if cat not in FURN_CATS:
            continue
        rec = gt.get(iid)
        if rec is None or (rec.get("motion_type") != "static"):
            continue
        if use_gt_pos:
            pos = np.array(rec["positions"][0])
        else:
            pl = o["placements"][0]
            pos = np.array(pl.get(furn_pos) or pl["position"])
        ext = max(rec["extent_m"]) if rec.get("extent_m") else 0.0
        cand.append((ext, iid, pos, o.get("name")))
    cand.sort(key=lambda c: -c[0])
    cand = cand[:cap]
    return ([c[1] for c in cand],
            np.array([c[2] for c in cand]) if cand else np.zeros((0, 3)),
            [c[3] for c in cand])


def nearest_recept(p, R, prev=None):
    """1.6m 이내 최근접 가구 인덱스(+히스테리시스). 없으면 None(=floor)."""
    if len(R) == 0:
        return None
    d = np.linalg.norm(R - np.asarray(p), axis=1)
    j = int(np.argmin(d))
    if d[j] > NEAR_MAX:
        return None
    if prev is not None and prev != j and d[prev] < d[j] + HYST:
        return prev
    return j


def belief_position(o, t):
    """그래프가 t 시점에 믿는 위치 — t 를 덮는 배치, 없으면 마지막으로 본 자리."""
    bel = None
    for pl in o["placements"]:
        if pl["start_frame"] <= t <= pl["end_frame"]:
            return pl["position"], True
        if pl["end_frame"] < t:
            bel = pl["position"]
    return bel, False


def run(graph, gt, tick_frames=50, furn_pos="position"):
    """A(GT) vs B(그래프) 수용체 답 비교. tick_frames=50 → 5초 (home-jepa 의 틱)."""
    objs = graph["objects"]
    n_frames = graph["n_frames"]

    ids_gt, R_gt, names_gt = receptacles_from(objs, gt, use_gt_pos=True)
    ids_pr, R_pr, names_pr = receptacles_from(objs, gt, use_gt_pos=False, furn_pos=furn_pos)
    # 같은 물체 집합을 쓰되 위치만 다르게 — 답을 직접 비교할 수 있어야 한다
    common = [i for i in ids_gt if i in ids_pr]
    R_gt = np.array([R_gt[ids_gt.index(i)] for i in common])
    R_pr = np.array([R_pr[ids_pr.index(i)] for i in common])
    names = [names_gt[ids_gt.index(i)] for i in common]

    dyn = [i for i, r in gt.items() if r["motion_type"] == "dynamic" and r["moves"] and i in objs]
    rows = []
    for iid in dyn:
        rec, o = gt[iid], objs[iid]
        P = np.array(rec["positions"])
        prev_g = prev_b = prev_m = prev_l = None
        init = np.array(o["placements"][0]["position"])
        for t in range(0, min(n_frames, len(P)), tick_frames):
            if t < o["first_frame"]:
                continue
            g_ans = nearest_recept(P[t], R_gt, prev_g)
            bpos, observed = belief_position(o, t)
            if bpos is None:
                continue
            b_ans = nearest_recept(bpos, R_pr, prev_b)
            # 우리 물체 위치 + **GT 가구 위치** — 실패 원인이 물체 쪽인지 가구 쪽인지 가른다
            m_ans = nearest_recept(bpos, R_gt, prev_m)
            l_ans = nearest_recept(init, R_pr, prev_l)
            prev_g, prev_b, prev_m, prev_l = g_ans, b_ans, m_ans, l_ans
            rows.append({"obj": rec["name"], "t": t, "observed": observed,
                         "gt": g_ans, "graph": b_ans, "graph_gtfurn": m_ans, "initial": l_ans,
                         "err_m": float(np.linalg.norm(np.array(bpos) - P[t]))})

    def acc(key, sub=None):
        r = [x for x in rows if sub is None or sub(x)]
        return (float(np.mean([x[key] == x["gt"] for x in r])), len(r)) if r else (None, 0)

    return {
        "n_receptacles": len(common), "receptacles": names, "furn_pos": furn_pos,
        "n_queries": len(rows), "n_objects": len(dyn),
        "tick_frames": tick_frames,
        "graph_acc": acc("graph")[0],
        "graph_acc_observed": acc("graph", lambda x: x["observed"])[0],
        "graph_acc_unobserved": acc("graph", lambda x: not x["observed"])[0],
        "graph_gtfurn_acc": acc("graph_gtfurn")[0],
        "initial_acc": acc("initial")[0],
        "n_observed": acc("graph", lambda x: x["observed"])[1],
        "n_unobserved": acc("graph", lambda x: not x["observed"])[1],
        "belief_err_median_m": float(np.median([x["err_m"] for x in rows])) if rows else None,
        "rows": rows,
    }
