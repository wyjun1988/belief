"""조각을 **합치지 않고** belief 질의에 답한다.

SAM+BotSort 는 한 물체를 여러 트랙으로 쪼갠다(액자 하나가 트랙 3개). 그걸 하나로 잇는
재연결을 시도했지만 precision 0.387 에 그쳤다 — 실내 가구 크롭의 CLIP 유사도가 전부
0.97+ 로 몰려 변별이 안 되기 때문이다.

그런데 **질문에 답하는 데 합칠 필요가 없다.** "액자 어디 있어?" 는 조각 A 와 B 가 같은
물체인지 묻지 않는다. 둘 다 "액자처럼 생긴 것"이고 어느 쪽이 더 최근인지만 알면 된다.

    후보 = 질의와 맞는(카테고리 또는 외형) 모든 노드
    답   = 그중 **가장 최근에 관측된** 배치의 구역
    단,  부재 증거로 "떠났다"고 판정된 배치는 후보에서 내린다

애매함이 생기면(같은 종류가 둘) 그건 질문 자체의 애매함이지 지어내서 없앨 것이 아니다 —
그래서 후보 수와 2순위 답도 함께 돌려준다.
"""
import numpy as np


def candidates(graph, by="category", key=None, emb=None, ref_emb=None, sim_min=0.9):
    """질의에 맞는 노드 목록."""
    out = []
    for iid, o in graph["objects"].items():
        if by == "category":
            if key is not None and (o.get("category") or "").lower() != str(key).lower():
                continue
        elif by == "appearance":
            e = None if emb is None else emb.get(int(o.get("local_id", -1)))
            if e is None or ref_emb is None or float(np.dot(e, ref_emb)) < sim_min:
                continue
        elif by == "gt":
            if int(o.get("gt_instance", -1)) != int(key):
                continue
        out.append((iid, o))
    return out


def answer(graph, cands, t, assign_zone, respect_absence=True):
    """t 시점 질의 → (zone, position, 근거). 후보 중 **가장 최근 배치**를 고른다."""
    best = None
    for iid, o in cands:
        dep = o.get("departure") if respect_absence else None
        for pl in o["placements"]:
            if pl["start_frame"] > t:
                continue
            # 그 자리를 다시 봤는데 없었다면(부재 증거) 이 배치는 이미 무효다
            if dep and dep.get("departed_at") is not None and dep["departed_at"] <= t \
                    and pl is o["placements"][-1]:
                continue
            end = min(pl["end_frame"], t)
            if best is None or end > best[0]:
                best = (end, iid, pl)
    if best is None:
        return None
    end, iid, pl = best
    return {"zone": assign_zone(pl["position"]), "position": pl["position"],
            "node": iid, "last_seen": int(end), "n_candidates": len(cands),
            "n_obs": pl["n_obs"]}


def _ref_embedding(graph, emb, gt_iid):
    """질의 대상을 한 번 '가리킨' 셈 — 그 인스턴스를 가장 많이 본 트랙의 임베딩을 쓴다.

    ⚠️ GT 는 **질의를 지목하는 데만** 쓴다(사용자가 물건을 한 번 보여주는 상황).
    답을 고를 때는 임베딩 유사도만 본다 — 카테고리도 GT id 도 보지 않는다.
    """
    best = None
    for _, o in graph["objects"].items():
        if int(o.get("gt_instance", -1)) != int(gt_iid):
            continue
        if best is None or o["n_obs"] > best["n_obs"]:
            best = o
    if best is None or emb is None:
        return None
    return emb.get(int(best.get("local_id", -1)))


def run(graph, gt, ref_reg, assign_fn, tick=50, by="category", emb=None, sim_min=0.9):
    """동적 물체별로 매 틱 질의하고 GT 구역과 대조."""
    rows = []
    for iid, rec in gt.items():
        if rec["motion_type"] != "dynamic" or not rec["moves"]:
            continue
        cat = rec.get("category")
        if by == "appearance":
            ref = _ref_embedding(graph, emb, iid)
            if ref is None:
                continue
            cands = candidates(graph, by=by, emb=emb, ref_emb=ref, sim_min=sim_min)
        else:
            cands = candidates(graph, by=by, key=(cat if by == "category" else int(iid)), emb=emb)
        if not cands:
            continue
        P = np.array(rec["positions"])
        first_move = min(m["start_idx"] for m in rec["moves"])
        t0 = min(o["first_frame"] for _, o in cands)
        for t in range(t0, min(len(P), graph["n_frames"]), tick):
            a = answer(graph, cands, t, assign_fn)
            if a is None:
                continue
            rows.append({"obj": rec["name"], "category": cat, "t": t,
                         "gt": assign_fn(P[t]), "ans": a["zone"],
                         "n_candidates": a["n_candidates"],
                         "after_move": t >= first_move,
                         "err_m": float(np.linalg.norm(np.array(a["position"]) - P[t]))})

    def acc(sub=None):
        r = [x for x in rows if (sub is None or sub(x)) and x["gt"] is not None]
        return (float(np.mean([x["ans"] == x["gt"] for x in r])), len(r)) if r else (None, 0)

    a_all, n_all = acc()
    a_mv, n_mv = acc(lambda x: x["after_move"])
    return {"n_queries": len(rows), "accuracy": a_all,
            "accuracy_after_move": a_mv, "n_after_move": n_mv,
            "median_candidates": float(np.median([r["n_candidates"] for r in rows])) if rows else None,
            "median_err_m": float(np.median([r["err_m"] for r in rows])) if rows else None,
            "rows": rows}
