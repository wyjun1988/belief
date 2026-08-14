"""이동 이벤트 로그 — "언제 · 어떤 물체가 · 어느 방에서 어느 방으로 · 누가 옮겼나".

씬그래프가 정적 지도면 반쪽이다. Khronos/DAAAM 이 내는 DSG 는 물체 노드의 *현재*
상태를 담지만, 우리가 답해야 하는 질문("액자 어디 있어?", "누가 옮겼어?")은 **이력**을
요구한다. 그래서 배치(placement) 이력을 home-jepa 시뮬레이터가 쓰는 것과 **같은 스키마**의
이벤트로 환원한다(`homejepa/sim.py:_move`):

    {t, obj, src, dst, src_room, dst_room, mover, reason}

같은 스키마로 내는 이유는 하나다 — home-jepa 는 지금까지 시뮬레이터가 만든 이벤트로만
학습·평가했다. 실센서 지각이 같은 모양의 이벤트를 내면 그 모델을 그대로 태울 수 있다.

## `mover`(누가 옮겼나)는 **기본에서 뺐다** — 채점할 수 없기 때문이다.

    ADT 에는 "누가 옮겼는지" 라벨이 자체가 없다(home-jepa 의 ADT 변환도 같은 이유로
    "ADT does not label movers" 라 적고 전부 동거인으로 뭉갰다). 라벨이 없으면 정확도를
    낼 수 없고, 낼 수 없는 값을 그래프에 박아 두면 하류가 그걸 사실로 믿는다.

    시도했던 것과 실패한 이유를 남긴다:
      1. GT 스켈레톤으로 "Skeleton_T 가 옮겼다" → 모캡 없이는 불가능한 정보.
      2. "카메라가 팔 길이 안 → 내가 옮김" → 1인 시퀀스는 맞았지만 2인 파티에서 9건 중
         8건이 self 로 뒤집혔다. 식탁 옆에서 구경만 해도 카메라는 1m 안이다.
      3. "다른 사람이 나보다 가까우면 other" → 파티가 8/9 other 로 바뀌었지만, 이게
         맞았는지 **확인할 방법이 없다**. GT 스켈레톤은 사람이 *어디 있었는지*만 말한다.

    `enable_mover=True` 로 켤 수는 있다(연구용). 켜면 근거 거리도 함께 남는다.

## `move_observed`(이동을 봤나)는 남긴다 — 채점은 되지만 **이 데이터로는 판별력이 없다.**

    GT 마스크 가시율로 채점하면 두 파일럿 시퀀스에서 13/13 일치인데, 13건이 전부
    "봤다"라서 "항상 True" 예측기도 13/13 이다. ADT 파티·데코레이션은 관찰자가 내내
    현장에 있어 **미관측 이동이 구조적으로 없다**. 이 축을 시험하려면 관찰자가 자리를
    비우는 시퀀스가 필요하다.
"""
import numpy as np

ARM_REACH = 1.0         # m. 착용자 카메라(머리)에서 손이 닿는 거리 — 넉넉하게
MOVER_RADIUS = 1.2      # m. (GT 스켈레톤 모드에서만) 사람이 이 안에 있어야 후보
SELF, OTHER = "self", "other"


def _agent_positions(graph, t):
    """t 시점 근처 에이전트 위치 {id: xyz}. 궤적에서 최근접 샘플."""
    out = {}
    for aid, a in (graph.get("agents") or {}).items():
        tr = a.get("trajectory")
        if not tr:
            continue
        T = np.array([r[0] for r in tr])
        j = int(np.argmin(np.abs(T - t)))
        if abs(T[j] - t) <= 30:                 # 3초 이내 관측만
            out[a.get("name") or aid] = np.array(tr[j][1:4], float)
    return out


def _transit(all_placements, t0, t1):
    """두 정지 배치 사이의 **운반 중 관측** 목록. 여기에 이동의 증거가 다 있다."""
    return [p for p in all_placements if t0 < p["start_frame"] <= t1
            and p["end_frame"] <= t1 and not p.get("stable")]


def attribute_mover(graph, poses, transit, t0, t1, use_gt_skeletons=False):
    """이 이동을 누가 했는가. (mover, min_distance).

    ⚠️ 정지 배치 사이의 간격만 보면 안 된다. 처음엔 그렇게 했다가 거의 전부 '미상'이
    나왔는데, 실제로는 그 사이에 운반 중 관측이 촘촘히 있었다 — 우리는 물건이 옮겨지는
    것을 **보고 있었다**. 그 운반 관측들에서 물체와 카메라의 최소 거리를 본다.

        self   최소 거리가 팔 길이 안  → 내가 옮겼다 (확신)
        other  운반은 봤는데 카메라는 멀다 → 누군가 옮겼다 (신원 모름)
        None   운반 관측 자체가 없다 → 안 보는 사이에 움직였다 (모름)
    """
    if not transit:
        return None, None
    dmin, dother, other_name = np.inf, np.inf, None
    for p in transit:
        q = np.array(p["position"])
        for t in range(p["start_frame"], p["end_frame"] + 1):
            if poses is not None and 0 <= t < len(poses):
                dmin = min(dmin, float(np.linalg.norm(poses[t][:3, 3] - q)))
        # 검출된 다른 사람 — GT 스켈레톤 관절이 아니라 **우리가 세그먼트한 사람 마스크**의
        # 3D 위치다(agents 층). 1차의 GT-세그 가정과 같은 층위이고, SAM 으로 교체 가능하다.
        for name, ap in _agent_positions(graph, p["start_frame"]).items():
            dd = float(np.linalg.norm(ap - q))
            if dd < dother:
                dother, other_name = dd, name

    # ⚠️ "카메라가 팔 길이 안 → 내가 옮김"만 쓰면 사람 많은 곳에서 무너진다. 실측:
    # 1인 decoration 은 4/4 정확했는데, 2인 파티에서 8/9 가 self 로 잘못 나왔다 —
    # 착용자가 식탁 옆에서 구경하는 동안 다른 사람이 옮겨도 카메라는 1m 안이다.
    # 그래서 **다른 사람이 나보다 가까우면 그 사람**으로 준다.
    if other_name is not None and dother < min(dmin, MOVER_RADIUS):
        return (other_name if use_gt_skeletons else OTHER), round(dother, 2)
    if dmin < ARM_REACH:
        return SELF, round(dmin, 2)
    return OTHER, (round(dmin, 2) if np.isfinite(dmin) else None)


def move_events(graph, poses=None, stable_only=True, min_distance=0.3,
                enable_mover=False, use_gt_skeletons=False, observed_gap=20):
    """배치 이력 → 이동 이벤트 목록 (home-jepa `gt_moves` 스키마 + 확장 필드).

    `observed_gap`: 앞 배치가 끝난 뒤 이만큼(프레임) 안에 새 자리를 봤으면
    "이동을 보고 있었다"로 친다. 그보다 길면 미관측 이동 — 옮긴이는 알 수 없다.
    """
    events = []
    for iid, o in graph["objects"].items():
        allp = o["placements"]
        pls = [p for p in allp if p.get("stable")] if stable_only else allp
        if len(pls) < 2:
            continue
        for a, b in zip(pls, pls[1:]):
            d = float(np.linalg.norm(np.array(b["position"]) - np.array(a["position"])))
            if d < min_distance:
                continue
            t = int(b["start_frame"])
            gap = int(b["start_frame"] - a["end_frame"])
            transit = _transit(allp, a["end_frame"], b["start_frame"])
            n_seen = sum(p["n_obs"] for p in transit)
            observed = bool(transit) or gap <= observed_gap
            mover, mdist = (attribute_mover(graph, poses, transit, a["end_frame"],
                                            b["start_frame"], use_gt_skeletons)
                            if enable_mover else (None, None))
            events.append({
                # --- home-jepa 스키마 ---
                "t": t,                                  # 그래프가 알아챈 시점
                "obj": o["name"],
                "src": a.get("support"),
                "dst": b.get("support"),
                "src_room": a.get("zone"),
                "dst_room": b.get("zone"),
                "mover": mover,          # 기본 None — ADT 에 라벨이 없어 채점 불가
                "reason": "observed",
                # --- 확장 (실센서에서만 의미 있는 것들) ---
                "instance_id": o["instance_id"],
                "category": o["category"],
                "t_last_seen_before": int(a["end_frame"]),
                "unobserved_gap_frames": gap,
                "move_observed": observed,
                "transit_observations": n_seen,
                "distance_m": round(d, 3),
                "changed_room": bool(a.get("zone") != b.get("zone")),
                "src_pos": a["position"], "dst_pos": b["position"],
                "mover_dist_m": mdist,
            })
    events.sort(key=lambda e: e["t"])
    return events


def summarize(events):
    from collections import Counter
    return {
        "n_events": len(events),
        "n_room_changes": sum(1 for e in events if e["changed_room"]),
        "movers": dict(Counter(e["mover"] or "unknown" for e in events)),
        "room_transitions": dict(Counter(
            "%s→%s" % (e["src_room"], e["dst_room"]) for e in events if e["changed_room"])),
        "median_unobserved_gap_frames": (
            float(np.median([e["unobserved_gap_frames"] for e in events])) if events else None),
    }
