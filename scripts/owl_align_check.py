#!/usr/bin/env python3
"""OWLv2 프레임 키가 CLIP 인덱스와 같은 프레임을 가리키는지 검증.

    $P scripts/owl_align_check.py

SuperMemory 프레임은 비디오에서 index.npz 의 ts 로 되찾아 뽑았다. 이 대응이 어긋나면
CLIP/OWLv2 비교 전체가 무의미해진다. 공통 단어에서 두 지각층의 프레임별 z 가
**양의 상관**이어야 한다 — 정렬이 깨지면 0 근처로 무너진다.

대조군으로 프레임을 500칸 굴린 것도 같이 재서, 상관이 정렬 덕분인지 확인한다.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")

from scripts.absence_evidence import clip_text                    # noqa: E402
from scripts.owl_presence import load_owl, owl_matrix, zscore     # noqa: E402


def main():
    sess = [s for s in ("s8", "s14", "s1", "s19", "s20")
            if os.path.exists(os.path.join(D, "owl_sm_%s.json" % s))]
    if not sess:
        sys.exit("OWLv2 검출 파일 없음")
    print("검증 세션: %s" % ", ".join(sess))
    for sd in sess:
        z = np.load(os.path.join(D, sd, "index.npz"))
        E = z["emb"].astype(np.float32)
        owl = load_owl({sd: os.path.join(D, "owl_sm_%s.json" % sd)})
        det = owl[sd]
        words = [w for w, _ in __import__("collections").Counter(
            w for d in det.values() for w in d).most_common(25)]
        order = [(sd, i) for i in range(len(E))]
        O = zscore(owl_matrix(owl, order, words))
        V = clip_text(["a photo of a " + w for w in words], "mps")
        C = zscore(V @ E.T)
        r = np.mean([np.corrcoef(O[i], C[i])[0, 1] for i in range(len(words))])
        Osh = np.roll(O, 500, axis=1)
        rs = np.mean([np.corrcoef(Osh[i], C[i])[0, 1] for i in range(len(words))])
        flag = "OK" if r > 0.1 and r > rs + 0.05 else "⚠️ 의심"
        print("  %-4s 프레임 %-5d 상관 **%+.3f** (500칸 어긋냄 %+.3f) %s"
              % (sd, len(E), r, rs, flag))


if __name__ == "__main__":
    main()
