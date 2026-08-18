#!/usr/bin/env python3
"""OWLv2 검출을 CLIP presence 와 같은 인터페이스로 바꾼다.

기존 평가는 전부 `Z[word, frame]` (CLIP 텍스트-이미지 유사도의 z점수)를 받는다.
지각층만 갈아끼우고 나머지를 그대로 두려면 OWLv2 검출도 같은 모양이어야 한다.

    Z, src = owl_z(owl, order, vocab, E=E, device="mps")

**어휘에 없는 단어는 CLIP 으로 폴백한다.** OWLv2 는 질의한 어휘만 검출하므로,
질문에서 나온 희귀 단어까지 다 넣으면 느려지고 빼면 행이 통째로 0 이 된다.
어느 쪽이 쓰였는지 `src` 로 돌려주니 결과 해석에서 반드시 같이 봐야 한다.
"""
import json
import os

import numpy as np


def load_owl(paths):
    """{세션: {프레임인덱스: {단어: 점수}}} — owl_dir.py 출력 여러 개를 합친다."""
    out = {}
    for sess, p in paths.items():
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        out[sess] = {int(os.path.splitext(k)[0]): v for k, v in d.items()}
    return out


def owl_matrix(owl, order, vocab, thr=0.0):
    """(단어 × 프레임) 밀집 점수. order = [(세션, 프레임인덱스), ...] = E 의 행 순서.

    검출 안 된 칸은 0 이다. CLIP 유사도가 항상 0.2~0.3 근처에 몰려 있는 것과 달리
    OWLv2 점수는 **희소**하다 — 이 희소성 자체가 신호다.

    thr 은 근접 게이트다. Nymeria 실측: 약한 검출은 문 너머·먼 거리 관측이라
    물체를 엉뚱한 장소에 넣는다 — 고정 가전의 세션 간 방 일치도가 문턱 0 에서 0.53,
    0.35 에서 0.93 이었다. 같은 논리가 프레임 단위 존재판정에도 적용된다."""
    vi = {w: i for i, w in enumerate(vocab)}
    S = np.zeros((len(vocab), len(order)), np.float32)
    for j, (sess, idx) in enumerate(order):
        for w, s in owl.get(sess, {}).get(idx, {}).items():
            i = vi.get(w)
            if i is not None and s >= thr:
                S[i, j] = s
    return S


def zscore(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)


def owl_z(owl, order, words, E=None, device="mps", owl_vocab=None, thr=0.0):
    """words 각각에 대해 OWLv2 행(있으면) 또는 CLIP 행(없으면)을 만들어 z점수화.

    돌려주는 src 는 단어별 출처("owl"/"clip") — 폴백 비율을 보고해야 비교가 정직하다.
    """
    seen = owl_vocab
    if seen is None:
        # 질의 어휘 파일이 있으면 그것을 쓴다. 없으면 '실제로 검출된 단어' 로
        # 대신하는데, 그러면 **어휘에 있었지만 한 번도 안 보인 단어가 CLIP 으로
        # 폴백**한다 — "한 번도 못 봤다"는 OWLv2 의 정당한 판정인데 CLIP 신호가
        # 섞여 들어가 비교가 오염된다.
        vp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "supermem", "owl_vocab.json")
        if os.path.exists(vp):
            seen = set(json.load(open(vp)))
        else:
            seen = set()
            for fr in owl.values():
                for d in fr.values():
                    seen.update(d)
    hit = [w for w in words if w in seen]
    miss = [w for w in words if w not in seen]
    Z = np.zeros((len(words), len(order)), np.float32)
    wi = {w: i for i, w in enumerate(words)}
    if hit:
        Zo = zscore(owl_matrix(owl, order, hit, thr))
        for k, w in enumerate(hit):
            Z[wi[w]] = Zo[k]
    if miss:
        if E is None:
            raise ValueError("CLIP 폴백에 E 가 필요하다 (어휘 밖 단어 %d개)" % len(miss))
        from scripts.absence_evidence import clip_text
        V = clip_text(["a photo of a " + w for w in miss], device)
        Zc = zscore(V @ E.T)
        for k, w in enumerate(miss):
            Z[wi[w]] = Zc[k]
    src = {w: ("owl" if w in seen else "clip") for w in words}
    return Z, src


def report_src(src, label=""):
    n_owl = sum(1 for v in src.values() if v == "owl")
    print("   지각층 %s: OWLv2 %d단어 · CLIP 폴백 %d단어 (%.0f%% OWLv2)"
          % (label, n_owl, len(src) - n_owl, 100 * n_owl / max(len(src), 1)))
    return n_owl / max(len(src), 1)
