#!/usr/bin/env python3
"""채택 학습셋에 **어려운 음성**을 더한다 — 같은 타입, 다른 집의 진짜 물체 (2026-09-17).

문제: 지금 음성은 '박스가 딴 데를 가리킴' 뿐이라 "이게 <타입>인가" 만 풀어도 절반은 맞는다.
실측: 참조를 다른 타입으로 바꿔치기해도 AUC 0.917 → 0.777 에 그친다(§166-79) — 참조를 덜 본다.
처방: 참조는 A 집의 머그, 후보는 **B 집의 진짜 머그**. 답은 "no" 다. 타입만 봐서는 못 푼다.
집 안에서는 타입이 유일하므로 같은 집에서는 이런 쌍을 만들 수 없어 **교차 집**으로 만든다.

  python scripts/lora_adopt_hardneg.py ~/khcache/lora_adopt_v2 ~/khcache/lora_adopt_hn [--ratio 0.5]
→ <out>/{train,val}.jsonl + images(심볼릭). 원본을 그대로 두고 어려운 음성만 덧붙인다.
"""
import argparse, json, os, random, collections
ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--ratio", type=float, default=0.5, help="원본 음성 대비 추가할 어려운 음성 비율")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); rng = random.Random(a.seed)
S = os.path.expanduser(a.src); O = os.path.expanduser(a.out)
os.makedirs(O, exist_ok=True)
if not os.path.exists(O + "/images"):
    os.symlink(os.path.abspath(S + "/images"), O + "/images")   # 이미지는 원본을 가리킨다(복사 없음)
stat = collections.Counter()
for split in ("train", "val"):
    rows = [json.loads(l) for l in open(f"{S}/{split}.jsonl")]
    pos = [r for r in rows if r["label"] == "yes"]
    n_neg = sum(1 for r in rows if r["label"] == "no")
    # 타입 → (집 → 양성 후보 크롭들)
    byt = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in pos: byt[r["type"]][r["house"]].append(r)
    refs = {}                                   # (집, oid) → 참조 파일
    for r in rows: refs[(r["house"], r["oid"])] = r["ref"]
    hard = []
    want = int(n_neg * a.ratio)
    tries = 0
    while len(hard) < want and tries < want * 40:
        tries += 1
        t = rng.choice(list(byt))
        hs = [h for h in byt[t] if byt[t][h]]
        if len(hs) < 2: continue
        ha, hb = rng.sample(hs, 2)
        ra = rng.choice(byt[t][ha]); rb = rng.choice(byt[t][hb])
        hard.append(dict(house=ra["house"], oid=ra["oid"], type=t, ref=ra["ref"],
                         cands=rb["cands"], label="no", hard=1,
                         neg_from=rb["house"], neg_oid=rb["oid"]))
    with open(f"{O}/{split}.jsonl", "w") as fo:
        for r in rows: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in hard: fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    stat[split + " 원본"] = len(rows); stat[split + " 어려운음성"] = len(hard)
print("HARDNEG_DONE", dict(stat), "→", O)
