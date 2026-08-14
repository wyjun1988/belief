"""belief 평가 — **"지금 어느 방에 있나"**.

이전 판은 home-jepa 의 ADT 규칙을 그대로 따라 "어느 가구(receptacle) 위에 있나"를
물었다. 그런데 기기 없는 조건(DA3 포즈)에서 물체 위치오차가 1.5m 라 가구 단위 질의는
원리적으로 답할 수 없다 — 가구 간격이 0.5~1.5m 이기 때문이다. 방은 3~5m 라 견딘다.

그래서 질의 단위를 **구역(방)** 으로 내린다. 실측상 이게 기기 없는 조건에서 유지되는
가장 세밀한 층이다(방 0.904 / 가구 실패).

두 가지를 따로 잰다 — 무엇이 틀렸는지 갈라 보려면 필요하다:

    end_to_end   우리 답 = **우리 그래프의 구역지도**로 매긴 구역
                 → 위치 오차 + 구역 분할 오차를 함께 본다 (실제 성능)
    localization 우리 답 = **기준 구역지도**로 매긴 구역
                 → 위치 오차만 본다 (구역 분할을 정답으로 고정)

정답은 언제나 기준 구역지도(GT 포즈·GT 뎁스로 만든 것)에 GT 좌표를 넣어 얻는다.
"""
import numpy as np

from kx.graph.frames import floor_basis
from kx.graph.regions import assign


class _Grid:
    def __init__(self, lo, res, shape):
        self.lo, self.res, self.shape = lo, res, shape

    def idx(self, uv):
        return np.clip(((np.atleast_2d(uv) - self.lo) / self.res).astype(int),
                       0, np.array(self.shape) - 1)


def load_regions(npz, zone_names, up):
    """`regions_*.npz` + 그래프 메타 → `kx.graph.regions.assign` 이 받는 형태."""
    return {"zones": npz["zones"], "rooms": npz["rooms"], "reach": npz["reach"],
            "res": float(npz["res"]), "zone_names": zone_names,
            "grid": _Grid(npz["lo"], float(npz["res"]), npz["zones"].shape),
            "basis": floor_basis(np.asarray(up, float))}


def belief_position(o, t):
    """t 시점의 믿음 — t 를 덮는 배치, 없으면 마지막으로 본 자리. (pos, 관측중 여부)"""
    bel = None
    for pl in o["placements"]:
        if pl["start_frame"] <= t <= pl["end_frame"]:
            return pl["position"], True
        if pl["end_frame"] < t:
            bel = pl["position"]
    return bel, False


def run(graph, gt, ref_reg, own_reg=None, tick=50, dynamic_only=True):
    """{obj, t} 단위 구역 정답률. own_reg 가 없으면 end_to_end 는 생략."""
    rows = []
    for iid, rec in gt.items():
        if dynamic_only and (rec["motion_type"] != "dynamic" or not rec["moves"]):
            continue
        o = graph["objects"].get(iid)
        if o is None:
            continue
        P = np.array(rec["positions"])
        first_move = min((m["start_idx"] for m in rec["moves"]), default=10 ** 9)
        init = o["placements"][0]["position"]
        for t in range(o["first_frame"], min(len(P), graph["n_frames"]), tick):
            bel, seen = belief_position(o, t)
            if bel is None:
                continue
            rows.append({
                "obj": rec["name"], "t": t, "observed": seen,
                "after_move": t >= first_move,
                "gt": assign(ref_reg, P[t])[1],
                "loc": assign(ref_reg, bel)[1],
                "e2e": assign(own_reg, bel)[1] if own_reg else None,
                "init": assign(ref_reg, init)[1],
                "room_changed": None,      # 아래에서 채운다
                "err_m": float(np.linalg.norm(np.array(bel) - P[t])),
            })

    # ⚠️ **층화가 없으면 이 지표는 무의미하다.** decoration 은 GT 방 전환이 0건이라
    # "절대 갱신 안 하는" 베이스라인이 0.918 을 받는다. 진짜로 재야 하는 것은
    # **방이 바뀐 질의**에서 그래프가 따라갔는가다.
    for r in rows:
        r["room_changed"] = (r["gt"] is not None and r["init"] is not None
                             and r["gt"] != r["init"])

    def acc(key, sub=None):
        r = [x for x in rows if (sub is None or sub(x)) and x["gt"] is not None]
        return (float(np.mean([x[key] == x["gt"] for x in r])), len(r)) if r else (None, 0)

    out = {
        "n_queries": len(rows),
        "n_objects": len({r["obj"] for r in rows}),
        "tick": tick,
        "end_to_end": acc("e2e")[0] if own_reg else None,
        "localization": acc("loc")[0],
        "localization_after_move": acc("loc", lambda x: x["after_move"])[0],
        "localization_observed": acc("loc", lambda x: x["observed"])[0],
        "localization_unobserved": acc("loc", lambda x: not x["observed"])[0],
        "baseline_initial": acc("init")[0],
        # 방이 실제로 바뀐 질의만 — 여기가 유일하게 판별력 있는 층
        "changed_end_to_end": (acc("e2e", lambda x: x["room_changed"])[0] if own_reg else None),
        "changed_localization": acc("loc", lambda x: x["room_changed"])[0],
        "changed_baseline": acc("init", lambda x: x["room_changed"])[0],
        "n_changed": acc("loc", lambda x: x["room_changed"])[1],
        "n_after_move": acc("loc", lambda x: x["after_move"])[1],
        "belief_err_median_m": float(np.median([r["err_m"] for r in rows])) if rows else None,
        "rows": rows,
    }
    return out
