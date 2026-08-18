#!/usr/bin/env python3
"""씬그래프의 방·가구 노드를 **Qwen 이 명명**한다 — 고정 어휘를 없앤다.

    $P scripts/name_rooms_qwen.py --seq <name> --graph graph_gtdepth

왜 최초 1회 명명인가: 집 구조는 거의 안 바뀐다. 그러니 방 타입을 고정 8종 어휘나
CLIP 사상으로 강제할 이유가 없다 — **부트스트랩 때 그 방 안의 물체 조합을 Qwen 에게
주고 이름을 정하게** 하면 어떤 집 구조든 열린 어휘로 받는다(사용자 설계).

두 가지를 낸다:
  ① 방 이름     구역별 물체 목록 → "이 방을 뭐라고 부를까?" (자유 명명, 문법 제약)
  ② 가구 노드   중복 트랙 병합(같은 카테고리 + 3D 근접) 후 대표 이름

⚠️ FastSAM 트랙은 소파 하나를 여러 노드로 쪼갠다(실측: GreySofa ×4). 병합 없이
belief 에 넣으면 후보 목록이 같은 가구로 도배된다 — Gen2 세션 정합에 쓴
카테고리+3D근접 기계를 세션 내부에 적용한다.
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SERVER = os.environ.get("LLAMA_SERVER",
                        "http://192.168.219.123:8080/v1/chat/completions")

GRAMMAR = r'''
root ::= "{" ws "\"room\"" ws ":" ws str ws "," ws "\"confidence\"" ws ":" ws num ws "}"
str ::= "\"" [a-z ]{2,24} "\""
num ::= "0." [0-9] | "1.0"
ws ::= [ \n]?
'''

PROMPT = """You are labeling rooms of a home from a 3D scene graph.
These objects were found together in one region of the home:
%s

What would a resident call this room? Use a natural short name
(e.g. kitchen, living room, bedroom, home office, bathroom, hallway).
Respond with JSON: {"room": "<name>", "confidence": <0.0-1.0>}"""


def pretty(name):
    s = re.sub(r"[_\d]+", " ", str(name))
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    return " ".join(s.split()).lower().strip()


def ask(objs, timeout=180):
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT % ", ".join(objs)}],
        "temperature": 0, "max_tokens": 60, "grammar": GRAMMAR,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(SERVER, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(json.loads(r.read())["choices"][0]["message"]["content"])


def merge_furniture(nodes, dist=0.6):
    """같은 카테고리 + 3D 근접 노드를 하나로. FastSAM 단편화 대응."""
    out, used = [], set()
    for i, a in enumerate(nodes):
        if i in used:
            continue
        grp = [a]
        used.add(i)
        for j in range(i + 1, len(nodes)):
            if j in used:
                continue
            b = nodes[j]
            if a["cat"] != b["cat"]:
                continue
            if np.linalg.norm(np.array(a["pos"]) - np.array(b["pos"])) <= dist:
                grp.append(b)
                used.add(j)
        P = np.array([g["pos"] for g in grp])
        out.append(dict(cat=a["cat"], name=grp[0]["name"], pos=P.mean(0).tolist(),
                        n_merged=len(grp), members=[g["name"] for g in grp]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth")
    ap.add_argument("--ref", default="gtdepth")
    ap.add_argument("--merge-dist", type=float, default=0.6)
    ap.add_argument("--top-objs", type=int, default=14, help="방마다 Qwen 에 줄 물체 수")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from kx.eval.room_belief import load_regions
    from kx.graph.regions import assign
    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    g = json.load(open(os.path.join(sd, args.graph + ".json")))
    meta = json.load(open(os.path.join(sd, "graph_%s.json" % args.ref)))["regions"]
    ref = load_regions(np.load(os.path.join(sd, "regions_%s.npz" % args.ref)),
                       meta["zone_names"], meta["up"])
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]

    # 구역별 물체 수집 (기하는 구역 배정에만, 이름은 그래프에서)
    by_zone = defaultdict(list)
    furn_by_zone = defaultdict(list)
    for iid, o in g["objects"].items():
        if not o.get("placements"):
            continue
        p = np.array(o["placements"][0]["position"], float)
        z = assign(ref, p)[1]
        if not z:
            continue
        nm = pretty(o.get("name") or "")
        rec = gt.get(str(o.get("gt_instance") or iid), {})
        cat = (rec.get("category") or o.get("category") or "").lower()
        ext = rec.get("extent_m") or o.get("extent_m")
        big = bool(ext and max(ext) >= 0.6)
        if nm:
            by_zone[z].append(nm)
        if big and cat:
            furn_by_zone[z].append(dict(cat=cat, name=nm or cat, pos=p.tolist()))

    result = {"zones": {}}
    for z, names in sorted(by_zone.items()):
        top = [n for n, _ in Counter(names).most_common(args.top_objs)]
        try:
            r = ask(top)
        except Exception as e:
            print("구역 %s 오류: %s" % (z, str(e)[:70]))
            continue
        furn = merge_furniture(furn_by_zone.get(z, []), args.merge_dist)
        dup = sum(f["n_merged"] - 1 for f in furn)
        result["zones"][z] = dict(qwen_room=r["room"], confidence=r["confidence"],
                                  objects=top, furniture=furn)
        print("구역 %-9s → Qwen **%-14s** (신뢰 %.1f) · 가구 %d개(중복 %d 병합)"
              % (z, r["room"], r["confidence"], len(furn), dup))
        print("      근거 물체: %s" % ", ".join(top[:8]))
        if furn:
            print("      가구: %s" % ", ".join("%s%s" % (f["name"],
                  "×%d" % f["n_merged"] if f["n_merged"] > 1 else "") for f in furn[:6]))
    if args.out:
        json.dump(result, open(args.out, "w"), ensure_ascii=False, indent=1)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
