#!/usr/bin/env python3
"""s1 이 왜 안 갈리는지 — **궤적 품질**인지 **공간 구조**인지 가른다.

    $P scripts/traj_quality.py

⑳ 에서 s8 은 체류 분할로 belief 0.48→0.86 이 됐다. s1 은 어떤 분할기로도 최빈방을
못 넘는다. 두 가설:

  (가) 궤적 품질 — MPS closed-loop 궤적이 드리프트해 방 경계가 흐려졌다
  (나) 공간 구조 — s1 의 방들이 실제로 겹쳐 있어 위치로 못 가른다

**같은 GT 방 라벨을 가진 근거끼리의 거리**(방 내 산포)와 **다른 방 근거와의 거리**
(방 간 거리)를 비교하면 갈린다. 위치가 방을 담고 있으면 방 내 < 방 간 이어야 한다.
드리프트가 있으면 **같은 방인데도 시간이 멀면 멀어진다** — 시간차에 따른 방 내
거리 증가로 드러난다.
"""
import json, os, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.supermem_rooms3d import D, load_traj, gravity, floor_basis   # noqa
from scripts.room_naming_probe import norm_room, SESS5                    # noqa


def main():
    qs = json.load(open(os.path.join(D, "qa_person_1.json")))
    ev = defaultdict(list)
    for x in qs:
        for e in ((x.get("answer_evidence") or {}).get("evidence_list") or []):
            g = norm_room(e.get("room"))
            sd = SESS5.get(e.get("video_id"))
            t = (e.get("time_span") or {}).get("start_time")
            if g and sd and t is not None and "Video" in (e.get("modalities") or []):
                ev[sd].append((float(t), g))

    for sd in ("s1", "s8"):
        pts = ev.get(sd, [])
        if len(pts) < 20:
            continue
        sec, P = load_traj(sd)
        gv = gravity(P); e1, e2 = floor_basis(gv)
        uv = np.stack([P @ e1, P @ e2], 1)
        X = np.array([[np.interp(t, sec, uv[:, 0]), np.interp(t, sec, uv[:, 1])]
                      for t, _ in pts])
        y = np.array([g for _, g in pts])
        T = np.array([t for t, _ in pts])
        Dm = np.linalg.norm(X[:, None] - X[None], axis=2)
        same = y[:, None] == y[None]
        np.fill_diagonal(same, False)
        diff = (y[:, None] != y[None])
        din, dout = Dm[same], Dm[diff]
        print("\n[%s] 근거 %d · 방 %s" % (sd, len(pts), dict(Counter(y))))
        print("  방 내 거리 중앙 %.2f m · 방 간 거리 중앙 %.2f m · 비 %.2f"
              % (np.median(din), np.median(dout), np.median(din) / np.median(dout)))
        # 드리프트: 같은 방 쌍을 시간차 구간별로
        dt = np.abs(T[:, None] - T[None])
        print("  같은 방인데 시간차가 벌어질 때 거리:")
        for lo, hi in ((0, 60), (60, 600), (600, 3600), (3600, 1e9)):
            m = same & (dt >= lo) & (dt < hi)
            if m.sum() < 20:
                continue
            print("    %-12s n=%-6d 중앙 %.2f m"
                  % ("%d~%ds" % (lo, hi) if hi < 1e9 else ">%ds" % lo,
                     int(m.sum()), float(np.median(Dm[m]))))
    print("\n→ 방 내/방 간 비가 1 에 가까우면 **위치로 방이 안 갈린다**(공간 구조).")
    print("  같은 방인데 시간차에 따라 거리가 계속 커지면 **드리프트**(궤적 품질).")


if __name__ == "__main__":
    main()
