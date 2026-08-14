"""방·구역 분할 — 벽으로 갈리는 방과, 벽 없이 이어진 기능적 구역을 함께 만든다.

**문제.** decoration 시퀀스의 아파트는 침실 하나만 벽과 문으로 닫혀 있고 거실·부엌·
다이닝은 한 공간으로 이어져 있다. 순수 기하(자유공간 연결성)로는 뒤의 셋을 절대 못
가른다 — 실제로 가르는 것은 **가구의 기능**이지 벽이 아니다.

**해법: 두 층을 따로 만든다.**

  rooms  자유공간을 침식(erosion)해 문 폭(≈0.8m)에서 끊고 연결성분을 잡는다.
         → 벽과 문이 만드는 위상적 방. 침실이 여기서 갈린다.
  zones  대형 가구를 **의미 시드**로 삼아 자유공간 위에서 **측지 보로노이**를 친다.
         냉장고→kitchen, 식탁→dining, 소파/TV→living, 침대→bedroom.
         → 벽 없는 개방공간도 기능 단위로 갈린다.

측지(geodesic) 거리를 쓰는 이유: 유클리드로 하면 벽 너머의 시드가 이길 수 있다.
자유공간 위 BFS 로 재면 문을 통과하는 실제 도달 거리가 되어, rooms 층과 zones 층이
저절로 정합한다(침실 안은 언제나 bedroom 시드가 이긴다).

`rooms` 는 순수 기하라 알고리즘적이고, `zones` 는 시맨틱 후처리다 — 사용자 요청대로
"알고리즘으로 안 되면 후처리로"를 층으로 분리해 무엇이 무엇 덕인지 드러나게 했다.
"""
from collections import deque

import numpy as np
from scipy import ndimage

from kx.graph.frames import floor_basis, to_floor

RES = 0.05               # m/cell
WALL_SLAB = (1.7, 2.4)   # 바닥 기준 높이(m). 가구는 없고 벽만 남는 띠
DOOR_ERODE = 0.45        # m. 문 폭 절반보다 조금 크게 — 이만큼 침식하면 출입구가 끊긴다
MIN_ROOM_M2 = 1.5

# 큰 정적 가구만 시드로 쓴다. 컵·책 같은 소품은 방을 정의하지 않는다.
SEED_CATEGORIES = {
    "refrigerator": "kitchen", "oven": "kitchen", "stove": "kitchen",
    "dishwasher": "kitchen", "microwave": "kitchen", "kitchen sink": "kitchen",
    "dining table": "dining", "dining chair": "dining",
    "couch": "living", "television": "living", "tv stand": "living",
    "coffee table": "living", "armchair": "living",
    "bed frame": "bedroom", "mattress": "bedroom", "nightstand": "bedroom",
    "dressing table": "bedroom", "wardrobe": "bedroom",
    "toilet": "bathroom", "bathtub": "bathroom",
    "desk": "office", "office chair": "office",
}
SEED_MIN_EXTENT = 0.5


class Grid:
    def __init__(self, uv, res=RES, margin=1.0):
        self.res = res
        self.lo = uv.min(axis=0) - margin
        self.shape = tuple(np.ceil((uv.max(axis=0) + margin - self.lo) / res).astype(int) + 1)

    def idx(self, uv):
        return np.clip(((np.atleast_2d(uv) - self.lo) / self.res).astype(int),
                       0, np.array(self.shape) - 1)

    def world(self, ij):
        return np.asarray(ij) * self.res + self.lo


def _geodesic_voronoi(free, seeds):
    """자유공간 위 다중소스 BFS. seeds = [(i, j, label_id)] → 라벨 격자."""
    lab = np.full(free.shape, -1, np.int32)
    dq = deque()
    for i, j, k in seeds:
        if free[i, j] and lab[i, j] < 0:
            lab[i, j] = k
            dq.append((i, j))
    H, W = free.shape
    while dq:
        i, j = dq.popleft()
        k = lab[i, j]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < H and 0 <= b < W and free[a, b] and lab[a, b] < 0:
                lab[a, b] = k
                dq.append((a, b))
    return lab


def _nearest_free(free, ij, radius=40):
    """시드가 가구 위(점유)에 떨어졌을 때 가장 가까운 자유 셀로 옮긴다."""
    i, j = ij
    if free[i, j]:
        return i, j
    H, W = free.shape
    for r in range(1, radius):
        i0, i1 = max(i - r, 0), min(i + r + 1, H)
        j0, j1 = max(j - r, 0), min(j + r + 1, W)
        sub = free[i0:i1, j0:j1]
        if sub.any():
            ii, jj = np.nonzero(sub)
            d = (ii + i0 - i) ** 2 + (jj + j0 - j) ** 2
            m = int(np.argmin(d))
            return int(ii[m] + i0), int(jj[m] + j0)
    return None


def build_regions(points_world, poses, up, seed_objects, res=RES):
    """방·구역 분할.

    points_world : (N,3) 정적 전역 포인트 (MPS 반-조밀)
    poses        : (F,4,4) T_world_camera — 자유공간의 씨앗이자 도달성 판정
    seed_objects : [{"name","category","position"(world 3), "extent_m"}]
    """
    basis = floor_basis(up)
    fl = to_floor(points_world, basis)
    traj = to_floor(poses[:, :3, 3], basis)

    floor_h = np.percentile(fl[:, 2], 1.0)
    grid = Grid(np.vstack([fl[:, :2], traj[:, :2]]), res=res)

    # --- 벽 점유: 가구가 닿지 않는 높은 띠에서만 본다 ---------------------------
    slab = fl[(fl[:, 2] - floor_h > WALL_SLAB[0]) & (fl[:, 2] - floor_h < WALL_SLAB[1])]
    occ = np.zeros(grid.shape, bool)
    if len(slab):
        ij = grid.idx(slab[:, :2])
        cnt = np.zeros(grid.shape, np.int32)
        np.add.at(cnt, (ij[:, 0], ij[:, 1]), 1)
        occ = cnt >= 3                                   # 점 3개 이상이면 벽으로 인정
    occ = ndimage.binary_closing(occ, np.ones((3, 3), bool))

    # ⚠️ **관측된 영역으로 먼저 가둔다.** 처음엔 free = ~wall 로 두고 궤적에서 flood
    # fill 했는데, 포인트가 하나도 없는 셀(집 바깥·미탐색)은 벽이 아니라는 이유로 전부
    # 자유공간이 되어 넘쳐 흘렀다 — 구역 면적 합이 233m² 로 나왔다. 반-조밀 포인트가
    # 하나라도 있는 셀(=우리가 본 곳) 주변으로만 자유공간을 인정한다.
    support = np.zeros(grid.shape, bool)
    sij = grid.idx(fl[:, :2])
    support[sij[:, 0], sij[:, 1]] = True
    known = ndimage.binary_dilation(support, np.ones((7, 7), bool))   # ≈0.3m 여유

    free = known & ~ndimage.binary_dilation(occ, np.ones((3, 3), bool))

    # 관찰자가 실제로 지나간 곳에서 flood fill — 도달 못한 방·가구 속을 잘라낸다
    seedmask = np.zeros(grid.shape, bool)
    tij = grid.idx(traj[:, :2])
    seedmask[tij[:, 0], tij[:, 1]] = True
    seedmask &= free
    if not seedmask.any():                       # 궤적이 전부 점유 셀 위 — 완화
        seedmask[tij[:, 0], tij[:, 1]] = True
        free = free | seedmask
    reach = ndimage.binary_propagation(seedmask, mask=free)

    # --- rooms: 문 폭에서 끊고 연결성분 ------------------------------------------
    r = max(int(round(DOOR_ERODE / res)), 1)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy ** 2 + xx ** 2) <= r * r
    core = ndimage.binary_erosion(reach, disk)
    lab, n = ndimage.label(core)
    if n:                                   # 너무 작은 조각은 배경으로
        for k in range(1, n + 1):
            if (lab == k).sum() * res * res < MIN_ROOM_M2:
                lab[lab == k] = 0
        lab, n = ndimage.label(lab > 0)
    rooms = _geodesic_voronoi(reach, [(i, j, int(lab[i, j]))
                                      for i, j in zip(*np.nonzero(lab > 0))])

    # --- zones: 의미 시드 측지 보로노이 ------------------------------------------
    seeds, names = [], []
    for o in seed_objects:
        z = SEED_CATEGORIES.get((o.get("category") or "").lower())
        if z is None:
            continue
        if o.get("extent_m") and max(o["extent_m"]) < SEED_MIN_EXTENT:
            continue
        f = to_floor(np.asarray(o["position"])[None], basis)[0]
        cell = _nearest_free(reach, tuple(grid.idx(f[:2])[0]))
        if cell is None:
            continue
        if z not in names:
            names.append(z)
        seeds.append((cell[0], cell[1], names.index(z)))
    # ⚠️ **구역은 방 경계를 넘으면 안 된다.** 측지 보로노이를 자유공간 전체에 한 번
    # 돌렸더니 문 바로 안쪽에서 거실 시드가 이겨, 침실(room 2)의 507셀이 living 으로
    # 새어 들어갔다. 실제로 그 자리에 액자 시작 위치(FloatingShelf04_1)가 있어서
    # "방에서 거실로 나왔다"는 사건이 통째로 사라졌다.
    # → 각 방 안에서 **그 방에 있는 시드끼리만** 경쟁시킨다. 시드가 없는 방은
    #   전역 보로노이 결과를 물려받는다(문 밖 시드가 대표하게 둔다).
    if seeds:
        zones = np.full(reach.shape, -1, np.int32)
        seed_room = {}
        for (i, j, k) in seeds:
            seed_room.setdefault(int(rooms[i, j]), []).append((i, j, k))
        global_z = _geodesic_voronoi(reach, seeds)
        for r in range(1, int(rooms.max()) + 1) if rooms.size else []:
            mask = (rooms == r) & reach
            if not mask.any():
                continue
            local = seed_room.get(r)
            if local:
                zones[mask] = _geodesic_voronoi(mask, local)[mask]
            else:
                zones[mask] = global_z[mask]
        rest = reach & (rooms <= 0)
        zones[rest] = global_z[rest]
    else:
        zones = np.full(reach.shape, -1, np.int32)

    return {
        "grid": grid, "basis": basis, "reach": reach, "occ": occ,
        "rooms": rooms, "n_rooms": int(rooms.max()) if rooms.size else 0,
        "zones": zones, "zone_names": names,
        "floor_h": float(floor_h), "res": res,
        "n_seeds": len(seeds),
    }


def assign(reg, position_world, require_free=False):
    """world 좌표 → (room_id, zone_name).

    `require_free=True` 면 그 셀이 자유공간이 아닐 때 최근접 스냅을 **하지 않고**
    (None, None) 을 준다 — 벽걸이 물체처럼 스냅 방향이 방을 가르는 경우, 호출측이
    '어느 방에서 보았는가' 같은 더 나은 근거를 쓰게 하기 위해서다.
    """
    f = to_floor(np.asarray(position_world)[None], reg["basis"])[0]
    i, j = reg["grid"].idx(f[:2])[0]
    zones, rooms = reg["zones"], reg["rooms"]
    if zones[i, j] < 0:
        if require_free:
            return None, None
        cell = _nearest_free(reg["reach"], (i, j))
        if cell is None:
            return None, None
        i, j = cell
    z = int(zones[i, j])
    return (int(rooms[i, j]) if rooms[i, j] > 0 else None,
            reg["zone_names"][z] if 0 <= z < len(reg["zone_names"]) else None)


def summary(reg):
    """방×구역 교차표 — 어떤 구역이 어떤 방 안에 있는지(= 벽이 가른 것 vs 시맨틱이 가른 것)."""
    out = {}
    for zi, zn in enumerate(reg["zone_names"]):
        m = reg["zones"] == zi
        area = float(m.sum()) * reg["res"] ** 2
        rr = reg["rooms"][m]
        rr = rr[rr > 0]
        vals, cnts = np.unique(rr, return_counts=True)
        out[zn] = {"area_m2": round(area, 2),
                   "rooms": {int(v): int(c) for v, c in zip(vals, cnts)},
                   "dominant_room": int(vals[np.argmax(cnts)]) if len(vals) else None}
    return out
