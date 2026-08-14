"""배치 추적기 — 파이프라인의 심장.

여기서 지키는 계약은 하나다: **우리가 움직인 것과 물체가 움직인 것을 구별한다.**
시점이 바뀌면 보이는 표면이 달라져 관측 중심이 물체 위를 미끄러지는데(소파 1.28m,
침대 1.44m 까지 관측된 적 있다), 그걸 이동으로 읽으면 안 된다.
"""
import numpy as np

from kx.graph.build import _segment_track

RNG = np.random.default_rng(0)


def face(center, right_side, n=200, half=0.4):
    """정육면체의 한 면만 관측한 점구름. 면을 바꾸면 중심이 반대편으로 미끄러진다."""
    t = RNG.uniform(-half, half, (n, 2))
    P = np.zeros((n, 3))
    P[:, 0] = center[0] + (0.5 if right_side else -0.5)
    P[:, 1] = center[1] + t[:, 0]
    P[:, 2] = center[2] + t[:, 1]
    return P


def test_static_object_under_viewpoint_change():
    """가만히 있는 물체를 앞뒤로 번갈아 보아도 배치는 하나여야 한다."""
    frames = np.arange(0, 40, 2)
    pts = [face([0, 0, 0], i % 2 == 0) for i in range(len(frames))]
    pl, ch = _segment_track(frames, pts, thresh=0.6)
    assert len(pl) == 1, "시점 변화를 이동으로 오인했다"
    assert ch == []


def test_real_move_is_detected():
    pts = [face([0, 0, 0], True)] * 10 + [face([2, 0, 0], True)] * 10
    pl, ch = _segment_track(np.arange(20) * 2, pts, thresh=0.3)
    assert len(pl) == 2 and len(ch) == 1
    assert 1.5 < ch[0]["distance_m"] < 2.5
    # 감지는 **처음 어긋난 관측** 시점이어야 한다 (확정 시점이 아니라)
    assert ch[0]["detected_at_frame"] == 20


def test_single_frame_leak_is_ignored():
    """한 프레임짜리 마스크 누출은 CONFIRM 이 막는다."""
    pts = [face([0, 0, 0], True)] * 10 + [face([5, 0, 0], True)] + [face([0, 0, 0], True)] * 10
    pl, ch = _segment_track(np.arange(21) * 2, pts, thresh=0.3)
    assert len(pl) == 1 and ch == []


def test_round_trip_gives_two_changes():
    """갔다가 제자리로 오면 배치 3개·변화 2건 — 상태를 되돌리지 않는다."""
    pts = ([face([0, 0, 0], True)] * 8 + [face([2, 0, 0], True)] * 8
           + [face([0, 0, 0], True)] * 8)
    pl, ch = _segment_track(np.arange(24) * 2, pts, thresh=0.3)
    assert len(pl) == 3 and len(ch) == 2


def test_position_is_median_not_voxel_centroid():
    """위치 추정량은 **프레임 중심의 중앙값**이다. 복셀 중심으로 바꾸면 회귀다.

    복셀은 관측 횟수와 무관한 집합이라, 9번 본 면과 1번 본 면이 같은 무게를 갖는다.
    중앙값은 관측 분포를 따라가고 복셀 중심은 두 면 사이로 끌린다 —
    실측에서 복셀 중심으로 바꿨을 때 belief 수용체 정확도가 0.900 → 0.700 이었다.
    """
    # 두 면이 0.3m 떨어져 있고 thresh 0.5 라 이동으로 잡히지 않는다(한 배치로 병합)
    pts = [face([0, 0, 0], True)] * 9 + [face([-0.3, 0, 0], True)]
    pl, _ = _segment_track(np.arange(10) * 2, pts, thresh=0.5)
    assert len(pl) == 1
    assert pl[0]["position"][0] > 0.45, "중앙값이면 다수 관측(+0.5) 쪽에 남는다"
    assert pl[0]["vox_centroid"][0] < 0.45, "복셀 중심은 소수 관측 쪽으로 끌린다"


def test_trailing_unconfirmed_observations_are_dropped():
    """⚠️ 확정(CONFIRM)에 못 미친 채 트랙이 끝나면 그 관측은 **버려진다**.

    설계상 의도된 보수적 동작이다 — 노이즈인지 진짜 이동인지 모르는 관측으로
    안정 배치를 오염시키지 않는다. 대신 마지막 1~2 관측이 배치에서 빠지므로
    end_frame·n_obs 가 그만큼 짧게 잡힌다. 바꾸려면 공표된 수치가 전부 움직인다.
    """
    pts = [face([0, 0, 0], True)] * 9 + [face([0, 0, 0], False)]     # 마지막에 반대면 1장
    pl, ch = _segment_track(np.arange(10) * 2, pts, thresh=0.3)
    assert len(pl) == 1 and ch == []
    assert pl[0]["n_obs"] == 9, "미확정 관측이 배치에 섞였다"
    assert pl[0]["end_frame"] == 16, "end_frame 이 마지막 관측(18)까지 가면 안 된다"
