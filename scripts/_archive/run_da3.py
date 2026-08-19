#!/usr/bin/env python3
"""DA3 추론 실행 (RunPod).

    python3 scripts/run_da3.py --seq /workspace/data/seq/<name> --mode window
    python3 scripts/run_da3.py --seq ... --mode mono --out-suffix _mono   # 애블레이션 바닥선

`--limit` 으로 앞 N 프레임만 돌려 VRAM·속도를 먼저 잰다.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kx.depth.da3_runner import run   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--model", default="da3metric-large")
    ap.add_argument("--mode", default="window", choices=["window", "mono", "nopose", "anchor"])
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = args.seq
    t0 = time.time()
    meta = run(args.seq, out, model_name=args.model, mode=args.mode,
               window=args.window, stride=args.stride, process_res=args.process_res,
               limit=args.limit, raw_suffix=args.out_suffix)
    meta["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(meta, indent=1)[:600])
    print("DA3_DONE  %.1f s  peak VRAM %.2f GB" % (meta["wall_s"], meta.get("peak_vram_gb", -1)))


if __name__ == "__main__":
    main()
