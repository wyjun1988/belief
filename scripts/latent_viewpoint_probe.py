#!/usr/bin/env python3
"""latent 장소 찾기가 **시점 변화**에 견디는가 — 세 거리를 한 자로 잰다.

    $P scripts/latent_viewpoint_probe.py --cache <cache.npz> [--status <rows.json>]

CLIP 프레임 임베딩은 구도에 민감하다. 같은 장소도 각도가 바뀌면 임베딩이 움직인다.
그러면 "장소가 달라서 안 닮은 것" 과 "각도가 달라서 안 닮은 것" 이 섞여, latent 로
장소를 찾는 것이 원리적으로 위태로워진다. 세 가지를 같은 자로 재서 가른다:

  ① **같은 클립 안**   같은 장소 · 다른 시점(몇 초 사이 시선 이동)  ← 시점만의 효과
  ② **같은 장소 v1↔v2** 같은 장소 · 다른 스캔(시점도 내용도 다름)
  ③ **다른 장소**      가장 닮은 남의 장소                        ← 구별해야 할 상대

판정: ② > ③ 이어야 장소를 찾을 수 있다. 그리고 ①이 ③에 가깝게 낮으면
**시점 변화만으로도 남의 장소만큼 멀어진다**는 뜻이라 latent 단독으로는 못 쓴다.
"""
import argparse, json
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--status", default=None,
                    help="쌍→객체 상태 JSON. 주면 Removed/Moved 별로 나눠 본다")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    E = {}
    for k in z.files:
        if k.startswith("E|"):
            nm, wv = k[2:].rsplit("|", 1)
            e = z[k].astype(np.float32)
            E[(nm, wv)] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    if not E:
        print("캐시에 임베딩이 없다")
        return
    names = sorted({n for n, _ in E})
    st = {}
    if args.status:
        kinds = defaultdict(set)
        for r in json.load(open(args.status)):
            kinds[r["pair"]].add(r["status"])
        st = {k: ("Removed" if "Removed" in v else "Moved") for k, v in kinds.items()}

    rows = []
    for n in names:
        if (n, "v1") not in E or (n, "v2") not in E:
            continue
        A, B = E[(n, "v1")], E[(n, "v2")]
        # ① 같은 클립 안 — 프레임 쌍 유사도(대각 제외)
        S = A @ A.T
        intra = float(np.median(S[np.triu_indices(len(A), 1)])) if len(A) > 1 else np.nan
        a = A.mean(0); a /= np.linalg.norm(a) + 1e-9
        # ② 같은 장소 다른 스캔
        cross = float(np.median(B @ a))
        # ③ 다른 장소 중 가장 닮은 것
        other = max(float(np.median(E[(m, wv)] @ a))
                    for m in names if m != n for wv in ("v1", "v2") if (m, wv) in E)
        rows.append((n, st.get(n, "?"), intra, cross, other))

    def rep(sel, tag):
        if len(sel) < 3:
            print("  %-10s 표본 부족(%d)" % (tag, len(sel)))
            return
        i = np.array([x[2] for x in sel]); c = np.array([x[3] for x in sel])
        o = np.array([x[4] for x in sel])
        print("  %-10s n=%-3d ①같은클립 **%.3f** · ②같은장소 v1↔v2 **%.3f** · "
              "③다른장소 최고 %.3f" % (tag, len(sel), np.median(i), np.median(c), np.median(o)))
        print("  %-10s   여백 ②−③ 중앙 **%+.3f** (양수 %.0f%%) · "
              "시점여백 ①−③ 중앙 %+.3f" % ("", np.median(c - o), 100 * (c > o).mean(),
                                          np.median(i - o)))

    print("장소 %d곳\n" % len(rows))
    rep(rows, "전체")
    if st:
        rep([r for r in rows if r[1] == "Removed"], "Removed")
        rep([r for r in rows if r[1] == "Moved"], "Moved")
    print("\n→ ②>③ 이면 장소를 찾을 수 있다. ①이 ③ 수준으로 낮으면 **시점 변화만으로도**")
    print("  남의 장소만큼 멀어진다는 뜻이라 latent 단독으로는 위험하다.")
    print("  Removed 의 ②가 Moved 보다 낮으면 그 차이가 **부재를 가리는 편향**이다.")


if __name__ == "__main__":
    main()
