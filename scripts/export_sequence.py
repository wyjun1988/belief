#!/usr/bin/env python3
"""ADT 시퀀스를 DAAAM 입력 레이아웃으로 내보낸다.

    P=~/work/stock-v2/.venv-mps/bin/python
    $P scripts/export_sequence.py --seq Apartment_release_decoration_seq137_M1292
    $P scripts/export_sequence.py --preset pilot --limit 20      # 빠른 스모크

내보낸 결과는 data/seq/<seq>/ 에 쌓이고, 검증은 scripts/audit_export.py 가 한다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.adt.export import export_depth, export_sequence   # noqa: E402
from kx.adt.gt_timeline import build_timeline      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADT_GT = os.path.join(ROOT, "data", "adt", "gt")
OUT = os.path.join(ROOT, "data", "seq")

PRESETS = {
    "pilot": [
        "Apartment_release_multiskeleton_party_seq102_M1292",
        "Apartment_release_decoration_seq137_M1292",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default=None)
    ap.add_argument("--preset", default=None, choices=sorted(PRESETS))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=704)
    ap.add_argument("--focal", type=float, default=350.0)
    ap.add_argument("--limit", type=int, default=None, help="앞의 N 프레임만 (스모크)")
    ap.add_argument("--no-depth", action="store_true", help="GT 뎁스 내보내기 생략")
    ap.add_argument("--no-seg", action="store_true")
    ap.add_argument("--tag", default="", help="출력 폴더 접미사 (예: _smoke)")
    ap.add_argument("--depth-only", action="store_true",
                    help="이미 내보낸 시퀀스에 GT 뎁스만 덧붙인다 (프레임 선정 재사용)")
    args = ap.parse_args()

    seqs = [args.seq] if args.seq else PRESETS[args.preset or "pilot"]
    for name in seqs:
        seq_dir = os.path.join(ADT_GT, name)
        if not os.path.isdir(seq_dir):
            sys.exit("시퀀스 폴더 없음: %s  (scripts/fetch_adt.py 먼저)" % seq_dir)
        out_dir = os.path.join(args.out, name + args.tag)
        os.makedirs(out_dir, exist_ok=True)
        print("== %s → %s" % (name, out_dir), flush=True)

        if args.depth_only:
            print("   " + json.dumps(export_depth(seq_dir, out_dir)), flush=True)
            continue

        stats = export_sequence(
            seq_dir, out_dir,
            fps=args.fps, size=args.size, focal=args.focal,
            with_depth=not args.no_depth, with_seg=not args.no_seg,
            limit=args.limit,
        )
        print("   " + json.dumps(stats, ensure_ascii=False), flush=True)

        gt = build_timeline(seq_dir, out_dir)
        print("   GT: " + json.dumps(gt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
