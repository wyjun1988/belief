#!/usr/bin/env python3
"""분할이 아니라 **명명**이 병목인지 가른다 (SuperMemory).

    $P scripts/room_naming_probe.py

⑯ 에서 SuperMemory 세 세션 모두 무작위 분할보다 유의하게 갈린다는 것을 확인했다
(z −4 ~ −32). 그런데 방 belief 는 최빈방(0.69)에 한참 못 미친다(0.41).

분할이 되는데 belief 가 안 된다면 남는 후보는 **군집에 이름을 붙이는 단계**다.
현재는 CLIP 이 군집 안 프레임의 방 점수를 합해 4개 이름 중 하나를 고른다.
실제로 예측 방 분포는 고른데(kitchen 3576 / living 3442 / entrance 2287 /
bedroom 1586) **GT 는 69% 가 kitchen** 이다.

여기서는 근거 프레임의 방 라벨을 GT 와 직접 대조해 **명명 정확도**만 잰다.
belief 를 거치지 않으므로 어느 단계가 깎아먹는지 분리된다.
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
ROOMS = ["kitchen", "living_room", "bedroom", "entrance"]
NORM = {"kitchen": "kitchen", "apartment kitchen": "kitchen",
        "living room": "living_room", "apartment living area": "living_room",
        "an apartment living area": "living_room", "living area": "living_room",
        "bedroom": "bedroom", "hallway": "entrance", "entrance": "entrance"}
SESS5 = {"Person_1_session_1_01312026_glasses_1266": "s1",
         "Person_1_session_8_03102026_glasses_1264": "s8",
         "Person_1_session_14_03152026_glasses_1266": "s14",
         "Person_1_session_19_03292026_glasses_1266sm": "s19",
         "Person_1_session_20_03292026_glasses_1284": "s20"}


def norm_room(r):
    if not r:
        return None
    r = str(r).strip().lower()
    return NORM.get(r) or next((v for k, v in NORM.items() if k in r), None)


def build_rooms(k):
    """세션별로 직접 군집한다(공유 rooms3d.json 미사용 — 경합 방지)."""
    from scripts.supermem_rooms3d import load_traj, gravity, floor_basis, cluster_rooms
    out = {}
    for sd in ["s1", "s8", "s14", "s19", "s20"]:
        f = os.path.join(D, sd, "closed_loop_trajectory.csv")
        if not os.path.exists(f):
            continue
        sec, P = load_traj(sd)
        g = gravity(P); e1, e2 = floor_basis(g)
        uv = np.stack([P @ e1, P @ e2], 1)
        cen, _ = cluster_rooms(uv, k)
        fr = np.arange(0, int(sec.max()) + 1)
        fu = np.stack([np.interp(fr, sec, uv[:, 0]), np.interp(fr, sec, uv[:, 1])], 1)
        out[sd] = dict(frame_room=np.argmin(
            np.linalg.norm(fu[:, None] - cen[None], axis=2), 1).tolist())
    return out


def main():
    qs = json.load(open(os.path.join(D, "qa_person_1.json")))

    # 근거: (세션, 초) → GT 방
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="방 군집 수(⑯ 기준 선택값)")
    ap.add_argument("--modality", default=None,
                    help="근거 modality 제한. 오디오 근거의 방 라벨은 **말한 사람이"
                         " 있던 방**이라 카메라 위치와 다를 수 있다")
    a = ap.parse_args()
    r3 = build_rooms(a.k)
    ev = []
    mods = Counter()
    for x in qs:
        for e in ((x.get("answer_evidence") or {}).get("evidence_list") or []):
            g = norm_room(e.get("room"))
            # 근거마다 자기 video_id 와 세션 내 초가 붙어 있다 — 문항의 video_ids
            # 첫 항목을 쓰면 **전혀 다른 세션**을 가리킨다(20개가 다 들어있다).
            sd = SESS5.get(e.get("video_id"))
            t = (e.get("time_span") or {}).get("start_time")
            m = tuple(sorted(e.get("modalities") or []))
            if g and sd in r3 and t is not None:
                mods[m] += 1
                if a.modality and a.modality not in (e.get("modalities") or []):
                    continue
                ev.append((sd, int(t), g))
    print("modality 분포 %s" % dict(mods.most_common()))
    print("근거 %d건 · 세션 %s" % (len(ev), sorted(set(s for s, _, _ in ev))))
    print("GT 방 분포 %s" % dict(Counter(g for _, _, g in ev)))

    # 군집 → GT 최빈방 (군집이 실제로 방을 가르는지의 상한)
    cl = defaultdict(Counter)
    for sd, t, g in ev:
        lab = r3[sd]["frame_room"]
        if 0 <= t < len(lab):
            cl[(sd, int(lab[t]))][g] += 1
    n = sum(sum(c.values()) for c in cl.values())
    if not n:
        print("근거가 군집 범위 밖 — 판정 불가")
        return
    hit = sum(c.most_common(1)[0][1] for c in cl.values())
    maj = Counter(g for _, _, g in ev).most_common(1)[0]
    print("\n군집 %d개 · 근거 %d건" % (len(cl), n))
    print("  최빈방 기준선(%s)          %.3f" % (maj[0], maj[1] / len(ev)))
    print("  **군집→GT최빈방 상한**      %.3f" % (hit / n))
    print("\n군집별 GT 방 구성:")
    for (sd, c), v in sorted(cl.items()):
        tot = sum(v.values())
        pur = v.most_common(1)[0][1] / tot
        print("  %-4s 군집%d  n=%-4d 순도 %.2f  %s"
              % (sd, c, tot, pur, dict(v.most_common(3))))
    # ── 군집 선택과 무관하게: **위치가 방을 담고 있는가**
    # 근거 위치 → GT 방 을 1-NN(leave-one-out)으로 직접 푼다. 군집 개수·명명이
    # 개입하지 않으므로 "분할 방법이 나쁜가" 와 "위치에 방 정보가 없는가" 가 갈린다.
    from scripts.supermem_rooms3d import load_traj, gravity, floor_basis
    print("\n위치→방 1-NN (leave-one-out) — 군집·명명 무관:")
    for sd in sorted(set(s_ for s_, _, _ in ev)):
        pts = [(t, g) for s_, t, g in ev if s_ == sd]
        if len(pts) < 10:
            print("  %-4s 근거 %d건 — 표본 부족" % (sd, len(pts)))
            continue
        sec, P = load_traj(sd)
        gg = gravity(P); e1, e2 = floor_basis(gg)
        uv = np.stack([P @ e1, P @ e2], 1)
        X = np.array([[np.interp(t, sec, uv[:, 0]), np.interp(t, sec, uv[:, 1])]
                      for t, _ in pts])
        y = [g for _, g in pts]
        Dm = np.linalg.norm(X[:, None] - X[None], axis=2)
        T = np.array([t for t, _ in pts])
        mj = Counter(y).most_common(1)[0]
        # **시간 인접 누수 차단.** 근거는 시간적으로 뭉쳐 있다 — 몇 초 떨어진 두
        # 점은 위치도 방도 같으므로 leave-one-out 1-NN 이 사실상 이웃 시각을
        # 베낀다. 최근접을 시간으로 T초 이상 떨어진 것만 허용해 다시 잰다.
        for gap in (0, 30, 300, 1800):
            M = Dm.copy()
            np.fill_diagonal(M, np.inf)
            M[np.abs(T[:, None] - T[None]) < gap] = np.inf
            ok = np.isfinite(M).any(1)
            if ok.sum() < 10:
                print("  %-4s 간격 %-5ds — 남는 표본 부족(%d)" % (sd, gap, ok.sum()))
                continue
            pr = [y[int(np.argmin(M[i]))] for i in np.nonzero(ok)[0]]
            gt = [y[i] for i in np.nonzero(ok)[0]]
            acc = float(np.mean([a == b for a, b in zip(pr, gt)]))
            base = Counter(gt).most_common(1)[0][1] / len(gt)
            print("  %-4s 간격 %-5ds n=%-4d 1-NN %.3f · 최빈 %.3f · %s"
                  % (sd, gap, len(gt), acc, base,
                     "**위치가 방을 담는다**" if acc > base + 0.05 else "차이 없음"))

    print("\n→ 상한이 최빈방보다 **높으면 명명이 병목**(군집은 방을 가르는데 이름이 틀림).")
    print("  상한이 최빈방과 비슷하면 **군집이 GT 방 경계와 무관**하다는 뜻이다.")


if __name__ == "__main__":
    main()
