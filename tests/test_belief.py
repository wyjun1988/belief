"""belief 응답과 에피소드 내보내기 — "지금 어디 있나"의 시간 규약.

이 문제의 본질은 **관측이 끊긴 사이**다. 그래서 계약도 시간에 관한 것이다:
관측 중에는 본 자리, 끊긴 뒤에는 마지막으로 본 자리(last-known), 그 자리를 다시
봤는데 없었다면 답을 유보한다.
"""
import numpy as np

from kx.eval.homejepa_export import build_episode
from kx.eval.node_belief import answer, candidates

ZONE = lambda p: "A" if p[0] < 2 else "B"          # noqa: E731


def obj(placements, **kw):
    return dict(placements=[dict(start_frame=s, end_frame=e, position=p, n_obs=50)
                            for s, e, p in placements], **kw)


def test_last_known_persists_between_observations():
    o = obj([(0, 100, [0, 0, 0]), (200, 300, [5, 0, 0])])
    for t, want in [(50, "A"), (150, "A"), (250, "B"), (400, "B")]:
        assert answer({}, [("x", o)], t, ZONE)["zone"] == want, "t=%d" % t


def test_absence_evidence_withdraws_the_answer():
    """그 자리를 다시 봤는데 없었다면 마지막 배치는 이미 무효다."""
    o = obj([(0, 100, [0, 0, 0])], departure={"departed_at": 150})
    assert answer({}, [("x", o)], 200, ZONE) is None
    assert answer({}, [("x", o)], 50, ZONE)["zone"] == "A"      # 부재 이전은 유효


def test_answer_picks_the_freshest_candidate():
    """조각이 여럿이면 **가장 최근에 본** 조각이 이긴다 — 재식별의 핵심 규칙."""
    stale = obj([(0, 100, [0, 0, 0])])
    fresh = obj([(0, 300, [5, 0, 0])])
    assert answer({}, [("a", stale), ("b", fresh)], 350, ZONE)["zone"] == "B"


def test_candidates_by_category():
    g = {"objects": {"1": {"category": "cup"}, "2": {"category": "book"}}}
    assert len(candidates(g, by="category", key="cup")) == 1
    assert len(candidates(g, by="gt", key=7)) == 0


def _tiny_graph():
    mk = lambda s, e, p: dict(start_frame=s, end_frame=e, position=p,      # noqa: E731
                              zone="living", n_obs=20)
    return {"n_frames": 500, "objects": {
        "10": dict(name="TableA", category="coffee table", gt_motion_type="STATIC",
                   n_obs=99, instance_id=10, placements=[mk(0, 499, [0, 0, 0])]),
        "11": dict(name="ShelfB", category="shelf", gt_motion_type="STATIC",
                   n_obs=90, instance_id=11, placements=[mk(0, 499, [4, 0, 0])]),
        "20": dict(name="Cup", category="cup", gt_motion_type="DYNAMIC", n_obs=40,
                   instance_id=20, gt_instance=20,
                   placements=[mk(0, 240, [0, .3, 0]), mk(300, 480, [4, .3, 0])]),
    }}


def test_episode_last_known_tracks_observations():
    """⚠️ 관측 시점은 이벤트의 **시작 틱**이다. 종료 틱을 쓰면 아직 진행 중인 관측이
    영원히 '과거'가 되지 않아 last-known 이 한 자리 뒤처진다(실제로 그랬다)."""
    P = np.concatenate([np.tile([0, .3, 0], (250, 1)), np.tile([4, .3, 0], (250, 1))])
    gt = {"20": dict(motion_type="dynamic", positions=P.tolist(),
                     moves=[dict(start_idx=245, end_idx=255)])}
    qs = build_episode(_tiny_graph(), gt, lambda p: "living", poses=None)["queries"]
    assert qs, "질의가 하나도 안 생겼다"
    by_t = {q["qt"]: q for q in qs}
    assert by_t[2]["last_recept"] == by_t[2]["gt_recept"], "관측 중인데 어긋났다"
    # 이동(프레임 250) 뒤 재관측(프레임 300 = 틱 6)이 늦어도 다음 틱에는 반영된다
    assert by_t[7]["last_recept"] == by_t[7]["gt_recept"]
    assert by_t[7]["moved"] == 0
    # 그 사이 구간은 moved 로 표시되어 채점 시 층화된다
    assert by_t[5]["moved"] == 1


def test_episode_marks_dt_since_last_observation():
    P = np.tile([0, .3, 0], (500, 1))
    gt = {"20": dict(motion_type="dynamic", positions=P.tolist(),
                     moves=[dict(start_idx=245, end_idx=255)])}
    qs = build_episode(_tiny_graph(), gt, lambda p: "living", poses=None)["queries"]
    assert all(q["dt"] == (q["qt"] - q["last_t"]) * 5 for q in qs)
