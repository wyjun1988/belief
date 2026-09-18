#!/usr/bin/env python3
"""여러 과제의 학습셋을 **한 어댑터로** 합친다 (2026-09-18).

왜: 판정기가 채택·부재 둘(기록자리는 폐기)인데 ③ 인계 뒤 **belief 답**이 셋째로 필요하다.
  실측 근거 — 도메인 혼합(HSSD+OG)은 두 도메인을 **모두** 올렸다(OG 0.760→0.837 · HSSD 0.908→0.918).
  과제 혼합도 같을지는 미지. **위험**: 채택의 "예"는 *새 자리에서* 그 물건, 부재의 "예"는 *옛 자리에*
  그대로 — 자리에 대해 뜻이 반대다. 그래서 합본은 반드시 **과제별로 따로 채점**해야 한다.

각 행에 `_task` 를 박아 트레이너의 `--task multi` 가 행마다 프롬프트·키를 고르게 한다.
집(장면) 단위 분리는 각 원본 셋이 이미 했으므로 그대로 따른다.

  python scripts/lora_multitask_merge.py ~/khcache/lora_mt adopt=~/khcache/lora_adopt_all presence=~/khcache/lora_presence_mix2 belief=~/khcache/lora_belief
"""
import collections, json, os, sys
out = os.path.expanduser(sys.argv[1]); os.makedirs(out + "/images", exist_ok=True)
n = collections.Counter()
with open(out + "/train.jsonl", "w") as ft, open(out + "/val.jsonl", "w") as fv:
    for spec in sys.argv[2:]:
        task, src = spec.split("=", 1); src = os.path.expanduser(src)
        if not os.path.isdir(src): print("⚠ 없음", src); continue
        for split, fo in (("train", ft), ("val", fv)):
            p = f"{src}/{split}.jsonl"
            if not os.path.exists(p): continue
            for l in p and open(p):
                r = json.loads(l); r["_task"] = task
                fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n[f"{task} {split}"] += 1
        for f in os.listdir(src + "/images"):
            d = out + "/images/" + f
            if not os.path.lexists(d): os.symlink(os.path.abspath(src + "/images/" + f), d)
print("MULTITASK_MERGE_DONE", dict(n), "→", out)
