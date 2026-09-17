#!/usr/bin/env python3
"""여러 LoRA 학습셋을 **집(장면) 단위로 겹치지 않게** 합친다 (2026-09-17).

문제: hssd_v2 와 hssd_c2set 은 같은 HSSD 장면 목록에서 만들어져 house_XXXX 가 **같은 장면**이다
(house_0015 의 gt0 117개 oid 가 완전히 동일). 단순 cat 으로 합치면 c2 train 에 v2 val 집 9채가 들어가
val 이 오염된다(이동 후 AUC 가 과대). 규칙: **어느 셋에서든 val 인 집은 합본에서도 val** — 그 집의 모든 행을 val 로.

  python scripts/lora_merge_sets.py <out> <set1> <set2> ...
"""
import json, os, sys, collections
out, srcs = os.path.expanduser(sys.argv[1]), [os.path.expanduser(s) for s in sys.argv[2:]]
os.makedirs(out + "/images", exist_ok=True)
val_h = set()
for s in srcs:
    for l in open(s + "/val.jsonl"): val_h.add(json.loads(l)["house"])
n = collections.Counter(); moved = 0
with open(out + "/train.jsonl", "w") as ft, open(out + "/val.jsonl", "w") as fv:
    for s in srcs:
        for split in ("train", "val"):
            for l in open(f"{s}/{split}.jsonl"):
                r = json.loads(l); dst = "val" if r["house"] in val_h else "train"
                if split == "train" and dst == "val": moved += 1
                (fv if dst == "val" else ft).write(l if l.endswith("\n") else l + "\n"); n[dst] += 1
        for f in os.listdir(s + "/images"):
            d = out + "/images/" + f
            if not os.path.lexists(d): os.symlink(os.path.abspath(s + "/images/" + f), d)
print(f"MERGE_DONE {os.path.basename(out)}: train {n['train']} · val {n['val']} · val 집 {len(val_h)} · train→val 로 옮긴 행 {moved} (장면 누수 차단)")
