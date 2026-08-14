"""GT 이동 검출 — **채점 기준의 기준**이라 여기가 틀리면 모든 지표가 같이 틀린다.

속도 기반으로 잡는 이유: 사람이 물건을 들었다 제자리에 놓는 구간이 많아서
"첫↔마지막 변위"만 보면 이동을 통째로 놓친다.
"""
import numpy as np

from kx.adt.gt_timeline import _detect_moves, _runs

DT = 0.1


def test_runs_boundaries():
    assert _runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    assert _runs(np.array([1, 1, 0], bool)) == [(0, 1)]
    assert _runs(np.zeros(5, bool)) == []


def test_single_move():
    pos = np.concatenate([np.zeros((20, 3)),
                          np.linspace([0, 0, 0], [2, 0, 0], 20),
                          np.tile([2, 0, 0], (20, 1))])
    mv = _detect_moves(pos, DT)
    assert len(mv) == 1
    assert abs(mv[0]["displacement_m"] - 2.0) < 0.05
    assert 15 <= mv[0]["start_idx"] <= 22 and 36 <= mv[0]["end_idx"] <= 42


def test_pick_up_and_put_back_is_not_a_move():
    """순변위가 D_MIN 미만이면 이동이 아니다 — 들었다 제자리."""
    pos = np.concatenate([np.zeros((20, 3)),
                          np.linspace([0, 0, 0], [.3, 0, 0], 5),
                          np.linspace([.3, 0, 0], [0, 0, 0], 5),
                          np.zeros((20, 3))])
    assert _detect_moves(pos, DT) == []


def test_brief_pause_merges_into_one_move():
    """0.5초 멈칫은 GAP_S 안이라 한 건으로 이어야 한다 — 안 그러면 precision 이 무너진다."""
    pos = np.concatenate([np.zeros((10, 3)),
                          np.linspace([0, 0, 0], [1, 0, 0], 10), np.tile([1, 0, 0], (5, 1)),
                          np.linspace([1, 0, 0], [2, 0, 0], 10), np.tile([2, 0, 0], (10, 1))])
    assert len(_detect_moves(pos, DT)) == 1


def test_long_pause_splits():
    """3초 정지는 GAP_S(1초)를 넘으므로 두 건으로 갈려야 한다."""
    pos = np.concatenate([np.zeros((10, 3)),
                          np.linspace([0, 0, 0], [1, 0, 0], 10), np.tile([1, 0, 0], (30, 1)),
                          np.linspace([1, 0, 0], [2, 0, 0], 10), np.tile([2, 0, 0], (10, 1))])
    assert len(_detect_moves(pos, DT)) == 2
