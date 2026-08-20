#!/usr/bin/env python3
"""latent 로 장소를 찾으면 **부재를 가리는가** — 편향을 직접 잰다.

    $P scripts/latent_bias_probe.py --cache <sd_cache.npz>

원래 설계가 **질의에서 키워드를 뺀** 이유가 있다: 키워드로 검색하면 물건이 보이는
프레임만 올라와(생존 편향) **부재가 원리적으로 관측 불가**하다. latent 유사도로
장소를 찾으면 그 편향이 **암묵적으로** 되살아난다 — 물건이 사라지면 그 프레임의
임베딩도 달라지므로, 앵커(물건이 있던 장면)와 덜 닮게 된다.

두 가지를 가른다:

  ① **편향의 크기** — 같은 장소인데 물건이 사라졌을 때 유사도가 얼마나 떨어지나.
     Removed 와 Moved 를 비교한다(Moved 는 씬에 남아 있다).
  ② **그래도 장소를 찾는가** — 여백 = (같은 장소 후) − (가장 닮은 다른 장소).
     이 값이 양수로 남으면 유사도가 떨어져도 **장소 식별에는 지장이 없다.**

②가 양수면 latent 장소 찾기는 쓸 수 있다. 음수로 뒤집히면 부재가 큰 물체일수록
장소를 놓치게 되므로 **마스크로 물체를 가린 앵커**가 필요하다.
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--status", required=True,
                    help="쌍→객체 상태 JSON (scenediff_absence --out 산출물)")
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    E = {}
    for k in z.files:
        if k.startswith("E|"):
            nm, wv = k[2:].rsplit("|", 1)
            E[(nm, wv)] = z[k]
    if not E:
        print("캐시에 임베딩이 없다 — 먼저 timeline 스크립트로 캐시를 구워야 한다")
        return
    rows = json.load(open(args.status))
    st = defaultdict(set)
    for r in rows:
        st[r["pair"]].add(r["status"])

    names = sorted({n for n, _ in E})
    out = []
    for n in names:
        if (n, "v1") not in E or (n, "v2") not in E:
            continue
        a = E[(n, "v1")].mean(0); a /= np.linalg.norm(a) + 1e-9
        same = float(np.median(E[(n, "v2")] @ a))          # 같은 장소, 변화 후
        self_ = float(np.median(E[(n, "v1")] @ a))         # 같은 장소, 변화 전
        best_other = -9.0
        for m in names:
            if m == n:
                continue
            for wv in ("v1", "v2"):
                if (m, wv) in E:
                    best_other = max(best_other, float(np.median(E[(m, wv)] @ a)))
        kinds = st.get(n, set())
        tag = ("Removed" if "Removed" in kinds else "Moved") if kinds else "?"
        out.append((n, tag, self_, same, best_other, same - best_other))

    def rep(sel, tag):
        if len(sel) < 3:
            print("  %-10s 표본 부족(%d)" % (tag, len(sel)))
            return
        s = np.array([x[3] for x in sel]); o = np.array([x[4] for x in sel])
        m = np.array([x[5] for x in sel]); v0 = np.array([x[2] for x in sel])
        print("  %-10s n=%-3d 변화전 %.3f → 변화후 **%.3f** (하락 %.3f) · "
              "다른 장소 최고 %.3f · **여백 중앙 %+.3f** · 여백>0 %.0f%%"
              % (tag, len(sel), np.median(v0), np.median(s), np.median(v0 - s),
                 np.median(o), np.median(m), 100 * (m > 0).mean()))

    print("장소 %d곳 · 앵커 = 변화 전(v1) 프레임 평균 임베딩\n" % len(out))
    rep(out, "전체")
    rep([x for x in out if x[1] == "Removed"], "Removed")
    rep([x for x in out if x[1] == "Moved"], "Moved")
    print("\n→ 여백 = (같은 장소 변화후) − (가장 닮은 다른 장소).")
    print("  양수면 물건이 사라져도 **장소 식별은 된다** = latent 장소 찾기 사용 가능.")
    print("  Removed 의 하락이 Moved 보다 크면 그만큼이 **부재를 가리는 편향**이다.")


if __name__ == "__main__":
    main()
