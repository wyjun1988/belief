#!/usr/bin/env python3
"""임의 어휘 목록에 대한 표면형 메타데이터 — make_synonyms.py 의 범용판.

    $P scripts/make_synonyms_list.py --words a.json --out b.json

ADT 는 GT 카테고리에서 개념을 뽑았지만, Nymeria·SuperMemory 는 우리가 정한
어휘가 곧 개념이다. 충돌 규칙은 동일하다 — 다른 개념의 정식 이름이거나 두 개념이
동시에 주장하는 표면형은 버린다.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.make_synonyms import ask          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", required=True, help="개념 목록 JSON(문자열 배열)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    concepts = sorted(set(json.load(open(args.words))))
    print("개념 %d개" % len(concepts), flush=True)
    raw = {}
    for i, c in enumerate(concepts):
        try:
            raw[c] = ask(c, [o for o in concepts if o != c], args.n)
        except Exception as e:
            print("  [%s] 오류 %s" % (c, e), flush=True); raw[c] = []
        if i % 10 == 0:
            print("  %d/%d" % (i, len(concepts)), flush=True)
    canon = set(concepts)
    claims = {}
    for c, ws in raw.items():
        for w in set(ws):
            claims.setdefault(w, set()).add(c)
    out, n = {}, 0
    for c in concepts:
        keep, drop = [c], []
        for w in dict.fromkeys(raw.get(c, [])):
            if w == c:
                continue
            if w in canon:
                drop.append((w, "정식이름"))
            elif len(claims.get(w, ())) > 1:
                drop.append((w, "중복주장"))
            else:
                keep.append(w)
        out[c] = {"surface": keep, "dropped": drop}; n += len(keep)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("표면형 %d개 (개념당 %.1f) → %s" % (n, n / len(concepts), args.out))


if __name__ == "__main__":
    main()
