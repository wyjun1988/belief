#!/usr/bin/env python3
"""타임라인 판정 재료 묶기 (RTX 로 보내기): prep jsonl 이 참조하는 프레임(기록 스캔 프레임·목격·문맥)만 house/live|map 구조로 복사 → zip.
  python scripts/timeline_pack.py <THOR_ROOT> <prep.jsonl> <out_dir>   # out_dir/<house>/{live,map}/... + out_dir/timeline_prep.jsonl"""
import os, sys, json, shutil
root, prep, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); n = 0
for l in open(prep):
    r = json.loads(l); hn = r["house"]; hdr = os.path.realpath(os.path.join(root, hn))
    need = [("map", "%04d.jpg" % r["record_k"])] if r.get("record_k") is not None else []
    need += [("live", "%06d.jpg" % e["t"]) for e in r["sightings"]] + [("live", "%06d.jpg" % e["t"]) for e in r["context"]]
    for sub, fn in need:
        src = os.path.join(hdr, sub, fn); dst = os.path.join(out, hn, sub, fn)
        if os.path.exists(src) and not os.path.exists(dst): os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst); n += 1
shutil.copy(prep, os.path.join(out, os.path.basename(prep))); print("PACK_DONE %d 장 → %s" % (n, out))
