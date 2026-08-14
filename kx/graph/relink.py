"""트랙 파편을 **belief 층에서** 잇는다 — "여기서 사라지고 저기서 나타났다".

**왜 지각 층에서 안 푸는가.** SAM+BotSort 는 짧은 트랙을 많이 만든다(실측: decoration
918프레임에서 고유 id 1,051개, 수명 중앙 11프레임). 액자 하나가 트랙 3개로 쪼개졌다.
시야에서 놓쳤다 다시 잡을 때마다 새 id 가 발급되기 때문이다. 이걸 외형 ReID 로 이으려면
긴 시간·시점 변화를 건너뛰는 재식별이 필요한데, 그건 어렵고 틀리면 조용히 오염된다.

**대신 추론으로 잇는다.** 물체가 옮겨졌다는 사건은 두 관측으로 남는다:

    ① A 자리를 **다시 봤는데 없더라**  (부재 증거)
    ② 비슷한 것이 B 자리에 **새로 나타났다** (출현 증거)

①이 핵심이다. 지금까지 우리 그래프에는 "그 자리를 봤는데 없었다"가 아예 기록되지
않았다 — 관측된 것만 쌓았다. 부재는 카메라 기하로 판정할 수 있다: 그 좌표가 화각 안에
있었고, 뎁스가 그보다 멀리 있었다면(가려지지 않았다면) 우리는 그것을 **볼 수 있었다**.
볼 수 있었는데 없었다면 떠난 것이다.

그 다음 ①과 ②를 잇는 것은 가설이고, 외형(CLIP)·크기·카테고리·시간 순서로 점수를 매겨
Hungarian 으로 푼다. 못 이으면 그냥 새 물체로 둔다 — **지어내지 않는다.**
"""
import numpy as np

Z_MIN, Z_MAX = 0.3, 8.0      # m. 관측 가능 거리
OCCL_TOL = 0.35              # m. 뎁스가 이보다 앞이면 가려진 것으로 본다
MIN_ABSENT = 5               # 이만큼의 프레임에서 "보일 수 있었는데 없었다"면 떠났다고 본다
MAX_GAP_S = 60.0             # 초. 이보다 오래 지나면 잇지 않는다
W_APP, W_SIZE, W_TIME = 0.6, 0.25, 0.15
LINK_MIN = 0.55              # 이 점수 아래는 잇지 않는다


def observable(p_world, T_wc, K, W, H, depth=None):
    """그 좌표를 이 프레임에서 **볼 수 있었는가** (화각 안 + 가려지지 않음)."""
    R, t = T_wc[:3, :3], T_wc[:3, 3]
    pc = R.T @ (np.asarray(p_world, float) - t)
    z = pc[2]
    if not (Z_MIN < z < Z_MAX):
        return False
    u = K[0, 0] * pc[0] / z + K[0, 2]
    v = K[1, 1] * pc[1] / z + K[1, 2]
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        return False
    if depth is not None:
        d = depth[vi, ui]
        if d > 0 and d < z - OCCL_TOL:
            return False                      # 앞에 뭔가 있다 — 못 봤을 수 있다
    return True


def departure_times(graph, poses, K, W, H, seg_reader, depth_reader=None,
                    every=4, min_absent=MIN_ABSENT):
    """물체별 '떠난 시점' — 마지막 관측 이후 **보일 수 있었는데 없었던** 첫 프레임.

    seg_reader(i) -> (H,W) 라벨맵,  depth_reader(i) -> (H,W) m  (없으면 가림 판정 생략)
    """
    out = {}
    last = {int(o.get("local_id", -1)): (o["last_frame"], o["placements"][-1]["position"], iid)
            for iid, o in graph["objects"].items()}
    n = graph["n_frames"]
    pend = {lid: [0, None] for lid in last}          # [부재 카운트, 첫 부재 프레임]
    for i in range(0, n, every):
        active = [lid for lid, (lf, _, _) in last.items() if lf < i and pend[lid][0] < min_absent]
        if not active:
            continue
        seg = seg_reader(i)
        if seg is None:
            continue
        dep = depth_reader(i) if depth_reader else None
        present = set(np.unique(seg).tolist())
        for lid in active:
            _, p, iid = last[lid]
            if not observable(p, poses[i], K, W, H, dep):
                continue
            if lid in present:
                pend[lid] = [0, None]                # 다시 보였다 — 초기화
                continue
            if pend[lid][1] is None:
                pend[lid][1] = i
            pend[lid][0] += 1
            if pend[lid][0] >= min_absent:
                out[last[lid][2]] = {"departed_at": int(pend[lid][1]),
                                     "absent_observations": int(pend[lid][0]),
                                     "from": p}
    return out


def _appearance(graph_obj, emb):
    e = emb.get(int(graph_obj.get("local_id", -1)))
    return None if e is None else np.asarray(e, float)


def link_fragments(graph, departures, emb=None, fps=10.0,
                   max_gap_s=MAX_GAP_S, link_min=LINK_MIN):
    """떠난 조각 ↔ 새로 나타난 조각을 잇는다. [{from, to, score, ...}]"""
    objs = graph["objects"]
    gone = [(iid, d) for iid, d in departures.items() if iid in objs]
    born = [(iid, o) for iid, o in objs.items()
            if o["first_frame"] > 0 and iid not in departures]
    if not gone or not born:
        return []

    cand = []
    for gi, (a_id, d) in enumerate(gone):
        A = objs[a_id]
        ea = _appearance(A, emb) if emb else None
        sa = np.array(A["placements"][-1].get("vox_extent") or [0, 0, 0], float)
        for bj, (b_id, B) in enumerate(born):
            if b_id == a_id or B["first_frame"] <= d["departed_at"] - 20:
                continue
            gap = (B["first_frame"] - d["departed_at"]) / fps
            if gap < 0 or gap > max_gap_s:
                continue
            eb = _appearance(B, emb) if emb else None
            app = float(np.dot(ea, eb)) if (ea is not None and eb is not None) else 0.5
            sb = np.array(B["placements"][0].get("vox_extent") or [0, 0, 0], float)
            if sa.any() and sb.any():
                r = np.sort(sa)[::-1] / np.maximum(np.sort(sb)[::-1], 1e-3)
                size = float(np.exp(-np.abs(np.log(np.clip(r, 1e-3, 1e3))).mean()))
            else:
                size = 0.5
            tscore = float(np.exp(-gap / max_gap_s))
            s = W_APP * app + W_SIZE * size + W_TIME * tscore
            if s >= link_min:
                cand.append((s, gi, bj, {"from": a_id, "to": b_id, "score": round(s, 3),
                                         "appearance": round(app, 3), "size": round(size, 3),
                                         "gap_s": round(gap, 1),
                                         "departed_at": d["departed_at"],
                                         "appeared_at": B["first_frame"],
                                         "from_pos": d["from"],
                                         "to_pos": B["placements"][0]["position"]}))
    # 탐욕적 1:1 매칭 (점수 높은 것부터) — 가설이 겹치면 강한 쪽만 남긴다
    cand.sort(key=lambda c: -c[0])
    used_g, used_b, links = set(), set(), []
    for s, gi, bj, rec in cand:
        if gi in used_g or bj in used_b:
            continue
        used_g.add(gi)
        used_b.add(bj)
        links.append(rec)
    return links


def merge(graph, links):
    """연결된 조각들을 하나의 물체 노드로 합친다(배치 이력을 시간순으로 잇는다)."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    for L in links:
        a, b = find(L["from"]), find(L["to"])
        if a != b:
            parent[b] = a
    groups = {}
    for iid in graph["objects"]:
        groups.setdefault(find(iid), []).append(iid)

    merged, mapping = {}, {}
    for root, members in groups.items():
        members.sort(key=lambda m: graph["objects"][m]["first_frame"])
        base = dict(graph["objects"][members[0]])
        pls, chs = [], []
        for m in members:
            o = graph["objects"][m]
            pls += o.get("placements", [])
            chs += o.get("changes", [])
        pls.sort(key=lambda p: p["start_frame"])
        base.update({"placements": pls, "changes": chs,
                     "merged_from": members, "n_fragments": len(members),
                     "first_frame": pls[0]["start_frame"] if pls else base["first_frame"],
                     "last_frame": pls[-1]["end_frame"] if pls else base["last_frame"],
                     "n_obs": sum(graph["objects"][m]["n_obs"] for m in members)})
        merged[root] = base
        for m in members:
            mapping[m] = root
    g = dict(graph)
    g["objects"] = merged
    g["relink"] = {"n_links": len(links), "n_before": len(graph["objects"]),
                   "n_after": len(merged), "links": links}
    return g, mapping
