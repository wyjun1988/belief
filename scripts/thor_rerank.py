#!/usr/bin/env python3
"""**2단계 검색** — 싼 신호로 후보를 좁히고, 좁혀진 것에만 무거운 모델(VLM)을 돌린다.

    $P scripts/thor_rerank.py --root data/thor2v --cache /tmp/thor2vcache --house house_0001

### 왜

진단(51 후속): 배회 프레임에서 **검색 상위 5장 중 물체가 실제로 있는 비율이 0.136** 이다.
물체가 보이는 프레임은 점수 **상위 13%** 에는 들어가는데, 600장 중 상위 5장(상위 0.8%)은
**음성 570장에서 나온 오검출**이 지배한다. 극단값 문제다.

그래서 후보를 **전역 상위 K** 가 아니라 **방마다** 뽑는다. 각 방의 최선 증거 1장씩만
VLM 에 물어 "이 사진에 X 가 있나" 를 판정하고, 있다고 한 방을 고른다.

    1fps 전 프레임 ─ OWL(가벼움) ─→ 방마다 최선 1장 ─ VLM(무거움) ─→ 방 결정

비용: 물체당 **방 개수만큼**(4~10회). 전 프레임에 VLM 을 돌리는 것의 1/100 이하다.
VLM 지연 실측 9~10초/장(`--image-min-tokens 1024` 서버).
"""
import argparse, base64, glob, json, os, subprocess, time

import numpy as np


def ask(url, path, obj, timeout=120):
    b = base64.b64encode(open(path, "rb").read()).decode()
    body = json.dumps(dict(
        model="q", temperature=0, max_tokens=8,
        chat_template_kwargs=dict(enable_thinking=False),
        messages=[dict(role="user", content=[
            dict(type="text", text="/no_think Is there a %s in this image? "
                                   "Answer only yes or no." % obj),
            dict(type="image_url", image_url=dict(url="data:image/jpeg;base64," + b))])]))
    r = subprocess.run(["curl", "-s", "-m", str(timeout), url,
                        "-H", "Content-Type: application/json", "-d", body],
                       capture_output=True).stdout
    try:
        c = json.loads(r)["choices"][0]["message"].get("content", "")
    except Exception:
        return None
    c = c.strip().lower()
    return True if c.startswith("yes") else (False if c.startswith("no") else None)


def words(t):
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--house", default=None)
    ap.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    ap.add_argument("--per-room", type=int, default=1, help="방당 VLM 에 물을 프레임 수")
    ap.add_argument("--max-obj", type=int, default=20)
    ap.add_argument("--only-answerable", action="store_true",
                    help="정답 방을 실제로 방문한 질의만. ⚠️ 안 걸면 65%%가 "
                         "**관측으로 원리적으로 답할 수 없는** 질의라 비교가 퇴화한다"
                         "(실측: 정답 방이 후보에 있던 비율 0.350).")
    ap.add_argument("--wq", type=float, default=0.90)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hds = sorted(glob.glob(os.path.join(args.root, "house_*")))
    if args.house:
        hds = [h for h in hds if os.path.basename(h) == args.house]
    rows = []
    t0 = time.time()
    for hd in hds:
        cf = os.path.join(args.cache, os.path.basename(hd) + ".npz")
        if not os.path.exists(cf):
            continue
        z = np.load(cf, allow_pickle=True)
        ol, ts = z["ol"], z["ts"]
        vocab = list(z["vocab"]); vi = {w: i for i, w in enumerate(vocab)}
        g = json.load(open(os.path.join(hd, "gt.json")))
        rt = g["room_types"]
        live = {m["t"]: m for m in g["live"]}
        room = np.array([live[int(t)]["room"] if int(t) in live else None for t in ts], object)
        moves = sorted(g["moves"], key=lambda m: m["t"])
        rooms = sorted({r for r in room if r})
        objs = [(oid, v["type"]) for oid, v in g["gt0"].items()
                if v["type"] in vi and v["room"]][:args.max_obj]
        for oid, ot in objs:
            j = vi[ot]
            mv = [m for m in moves if m["oid"] == oid]
            r_true = mv[-1]["to"] if mv else g["gt0"][oid]["room"]
            # ── 1단계(싼 것): 방마다 상위 프레임 · 방별 분위수
            cand, base_sc = {}, {}
            for rr in rooms:
                ix = np.nonzero(room == rr)[0]
                if len(ix) < 3:
                    continue
                base_sc[rr] = float(np.quantile(ol[ix, j], args.wq))
                cand[rr] = ix[np.argsort(-ol[ix, j])[:args.per_room]]
            if len(cand) < 2:
                continue
            if args.only_answerable and r_true not in cand:
                continue          # 정답 방을 아예 안 가봤으면 관측으로 답할 수 없다
            r_cheap = max(base_sc, key=base_sc.get)
            _ = r_cheap
            # ── 2단계(무거운 것): 방마다 최선 1장씩만 VLM
            votes = {}
            for rr, ix in cand.items():
                yes = 0
                for k in ix:
                    p = os.path.join(hd, "live", "%06d.jpg" % int(ts[k]))
                    if not os.path.exists(p):
                        continue
                    a = ask(args.url, p, words(ot))
                    yes += 1 if a else 0
                votes[rr] = yes
            best = max(votes.values()) if votes else 0
            # VLM 이 아무 데서도 못 찾으면 싼 신호를 따른다
            r_vlm = (max([r for r, v in votes.items() if v == best],
                         key=lambda r: base_sc.get(r, 0)) if best > 0 else r_cheap)
            rows.append(dict(house=os.path.basename(hd), oid=oid, otype=ot,
                             r_true=rt.get(r_true), r_cheap=rt.get(r_cheap),
                             r_vlm=rt.get(r_vlm), votes={rt.get(k): v for k, v in votes.items()}))
            if len(rows) % 5 == 0:
                print("  %d건 · %.0f초" % (len(rows), time.time() - t0), flush=True)

    if not rows:
        print("표본 없음"); return
    c1 = np.mean([r["r_cheap"] == r["r_true"] for r in rows])
    c2 = np.mean([r["r_vlm"] == r["r_true"] for r in rows])
    print("\n질의 %d · %.0f분 소요" % (len(rows), (time.time() - t0) / 60))
    print("  1단계만 (OWL 방별 분위수)   **%.3f**" % c1)
    print("  **2단계 (VLM 재판정)        %.3f**" % c2)
    from collections import Counter
    print("  방 개수 %s" % dict(Counter(len(r["votes"]) for r in rows)))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
