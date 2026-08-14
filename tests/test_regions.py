"""방/구역 분할 — 벽이 가르는 것과 의미가 가르는 것.

계약 두 개:
  · 벽으로 갈린 방은 침식으로 분리된다
  · **구역은 방 경계를 넘지 않는다** — 문 바로 안쪽에서 바깥 시드가 이기면
    "방에서 거실로 나왔다"는 사건이 통째로 사라진다(실제로 507셀이 샜었다)
"""
import numpy as np

from kx.graph.regions import assign, build_regions

RNG = np.random.default_rng(0)


def two_room_scene(door=(2.2, 2.8)):
    """10×5 m 공간, x=5 에 벽. `door` 구간만 열려 있다. (y-up)"""
    pts = [[RNG.uniform(0, 10), 0.0, RNG.uniform(0, 5)] for _ in range(12000)]
    for _ in range(4000):                                  # 내벽 (문 제외)
        z = RNG.uniform(0, 5)
        if not (door[0] < z < door[1]):
            pts.append([5.0, RNG.uniform(1.7, 2.4), z])
    for _ in range(3000):                                  # 외벽
        t, h = RNG.uniform(0, 1), RNG.uniform(1.7, 2.4)
        pts += [[t * 10, h, 0], [t * 10, h, 5], [0, h, t * 5], [10, h, t * 5]]
    traj = np.array([[x, 1.5, 2.5] for x in np.linspace(1, 9, 60)])
    poses = np.tile(np.eye(4), (60, 1, 1))
    poses[:, :3, 3] = traj
    return np.asarray(pts), poses


SEEDS = [{"name": "couch", "category": "couch", "extent_m": [1, 1, 1], "position": [2, .4, 2.5]},
         {"name": "bed", "category": "bed frame", "extent_m": [1, 1, 1], "position": [8, .4, 2.5]}]


def test_wall_splits_rooms():
    reg = build_regions(*two_room_scene(), up=np.array([0, 1, 0.]), seed_objects=SEEDS)
    assert reg["n_rooms"] == 2, "문 폭 침식이 방을 못 갈랐다"


def test_zones_respect_room_boundary():
    """문 바로 안쪽(x=4.9)은 자기 방 구역이어야 한다 — 옆방 시드가 새면 안 된다."""
    reg = build_regions(*two_room_scene(), up=np.array([0, 1, 0.]), seed_objects=SEEDS)
    assert assign(reg, [2, .5, 2.5])[1] == "living"
    assert assign(reg, [8, .5, 2.5])[1] == "bedroom"
    assert assign(reg, [4.9, .5, 2.5])[1] == "living"
    assert assign(reg, [5.3, .5, 2.5])[1] == "bedroom"


def test_free_space_is_bounded_by_observation():
    """관측된 곳 주변만 자유공간이다 — 안 그러면 집 밖으로 넘쳐 면적이 폭발한다."""
    reg = build_regions(*two_room_scene(), up=np.array([0, 1, 0.]), seed_objects=SEEDS)
    area = reg["reach"].sum() * reg["res"] ** 2
    assert 20 < area < 70, "자유공간 %.0f m² — 실제 바닥 50 m² 대비 폭주/과소" % area


def test_require_free_refuses_to_snap():
    """벽걸이 물체처럼 스냅 방향이 방을 가르는 경우, 호출측이 더 나은 근거를 쓰게 한다."""
    reg = build_regions(*two_room_scene(), up=np.array([0, 1, 0.]), seed_objects=SEEDS)
    assert assign(reg, [5.0, 2.0, 1.0], require_free=True) == (None, None)
