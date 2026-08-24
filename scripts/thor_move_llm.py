#!/usr/bin/env python3
"""**움직임** 패턴을 Qwen 3.5 9B 로 다양화한다. 배치(`thor_prior_llm.py`)와 짝이다.

    $P scripts/thor_move_llm.py --root data/thor2z --out data/thor_move.json

⚠️ **왜 필요한가.** `thor_gen2.py` 의 배회·이동이 전부 균등 난수였다.

    cur = rng.choice(rids)                          # 에이전트가 어느 방에 있든 같은 확률
    oid = rng.choice(cands)                         # 어느 물체든 같은 확률로 옮겨짐
    tgt = rng.choice([r for r in rids if r != ...])  # 목적지도 균등

배치만 Qwen 으로 현실화하고 움직임을 균등으로 두면 두 군데가 망가진다:

1. **belief 가 원리적으로 이동 물체를 못 맞힌다.** 실제로는 머그컵이 부엌→거실로
   가지 부엌→화장실로 가지 않는다. 목적지가 균등이면 학습할 구조가 없어
   belief 몫(전체의 38%)이 실제보다 나쁘게 측정된다.
2. **재방문 판정이 왜곡된다.** 실제 가정에서 화장실 체류는 짧고 거실은 길다.
   균등이면 방마다 재방문 확률이 같아져 (b)"있을 것이다" 상태가 비현실적으로 고르게 난다.

세 가지를 묻는다 — 방 체류 가중, 물체별 **이동 성향**, 이동 **목적지 분포**.
목적지는 배치 사전확률과 다르다: 머그컵이 놓이는 곳(부엌)과 옮겨지는 곳(거실)이 다르다.

⚠️ Qwen 3.5 는 thinking 모델이라 `enable_thinking=false` 를 줘야 한다.
"""
import argparse, glob, json, os, re, subprocess

ROOMS = ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]


def ask(url, msg, temp, mx=1400):
    body = json.dumps(dict(model="q", messages=[dict(role="user", content=msg)],
                           temperature=temp, max_tokens=mx,
                           chat_template_kwargs=dict(enable_thinking=False)))
    r = subprocess.run(["curl", "-s", "-m", "300", url, "-H", "Content-Type: application/json",
                        "-d", body], capture_output=True).stdout
    try:
        c = json.loads(r)["choices"][0]["message"].get("content", "")
    except Exception:
        return {}
    m = re.search(r"\{.*\}", c, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    types = set()
    for hd in sorted(glob.glob(os.path.join(args.root, "house_*"))):
        g = json.load(open(os.path.join(hd, "gt.json")))
        types |= {v["type"] for v in g.get("gt0", {}).values()}
    types = sorted(types)
    print("물체 유형 %d" % len(types), flush=True)

    # ── ① 방 체류 가중 (사람이 어느 방에서 시간을 보내나) ──
    dwell = ask(args.url,
                "/no_think Rooms: %s. In a real home over an hour, what fraction of "
                "waking time does a person spend in each room? Bathrooms are brief. "
                'JSON only: {"Kitchen": 0.2, ...}' % ", ".join(ROOMS), args.temp, 300)
    s = sum(float(dwell.get(r, 0)) for r in ROOMS)
    dwell = ({r: float(dwell.get(r, 0)) / s for r in ROOMS} if s > 0
             else {r: 1.0 / len(ROOMS) for r in ROOMS})
    print("  방 체류: " + " ".join("%s %.2f" % (r, dwell[r]) for r in ROOMS), flush=True)

    # ── ② 물체별 이동 성향 · ③ 이동 목적지 분포 ──
    prop, dest = {}, {}
    for i in range(0, len(types), args.batch):
        ch = types[i:i + args.batch]
        d = ask(args.url,
                "/no_think Rooms: %s. For each object give (a) how often a person "
                "picks it up and leaves it in a DIFFERENT room during a normal day, "
                '0..1 ("mobility"), and (b) where it typically ENDS UP when moved — '
                "this differs from where it is stored (a mug is stored in the kitchen "
                "but ends up in the living room). JSON only: "
                '{"Mug": {"mobility": 0.8, "dest": {"Kitchen": 0.3, "LivingRoom": 0.5, '
                '"Bedroom": 0.15, "Bathroom": 0.05}}}\nObjects: %s'
                % (", ".join(ROOMS), ", ".join(ch)), args.temp, 1800)
        for t in ch:
            v = d.get(t)
            if not isinstance(v, dict):
                continue
            try:
                prop[t] = min(max(float(v.get("mobility", 0.5)), 0.0), 1.0)
            except Exception:
                pass
            dd = v.get("dest")
            if isinstance(dd, dict):
                s = sum(float(dd.get(r, 0)) for r in ROOMS)
                if s > 0:
                    dest[t] = {r: float(dd.get(r, 0)) / s for r in ROOMS}
        print("  %d/%d · 성향 %d · 목적지 %d" % (i + len(ch), len(types), len(prop), len(dest)),
              flush=True)
    # ⚠️ 배치 하나가 통째로 실패하는 일이 있다(JSON 파싱 실패). 실측에서 첫 배치
    # 12종이 그렇게 날아갔다. 균등 폴백으로 두면 **그 종만 belief 가 못 맞히는**
    # 편향이 생기므로, 남은 것을 더 작은 배치로 다시 묻는다.
    for rnd in range(3):
        miss = [t for t in types if t not in dest]
        if not miss:
            break
        print("  재시도 %d회차 · 남은 %d종" % (rnd + 1, len(miss)), flush=True)
        for i in range(0, len(miss), 6):
            ch = miss[i:i + 6]
            d = ask(args.url,
                    "/no_think Rooms: %s. For each object give (a) how often a person "
                    "picks it up and leaves it in a DIFFERENT room during a normal day, "
                    '0..1 ("mobility"), and (b) where it typically ENDS UP when moved. '
                    'JSON only: {"Mug": {"mobility": 0.8, "dest": {"Kitchen": 0.3, '
                    '"LivingRoom": 0.5, "Bedroom": 0.15, "Bathroom": 0.05}}}\nObjects: %s'
                    % (", ".join(ROOMS), ", ".join(ch)), args.temp, 1200)
            for t in ch:
                v = d.get(t)
                if not isinstance(v, dict):
                    continue
                try:
                    prop[t] = min(max(float(v.get("mobility", 0.5)), 0.0), 1.0)
                except Exception:
                    pass
                dd = v.get("dest")
                if isinstance(dd, dict):
                    ss = sum(float(dd.get(r, 0)) for r in ROOMS)
                    if ss > 0:
                        dest[t] = {r: float(dd.get(r, 0)) / ss for r in ROOMS}
    nf = [t for t in types if t not in dest]
    if nf:
        print("  ⚠️ 끝내 실패해 균등 폴백 %d종: %s" % (len(nf), ", ".join(nf)), flush=True)
    for t in types:
        prop.setdefault(t, 0.5)
        dest.setdefault(t, {r: 1.0 / len(ROOMS) for r in ROOMS})

    json.dump(dict(dwell=dwell, mobility=prop, dest=dest), open(args.out, "w"), indent=1)
    import numpy as np
    ent = [-sum(p * np.log(p + 1e-9) for p in v.values()) for v in dest.values()]
    print("→ %s · 목적지 엔트로피 중앙 %.3f (균등이면 %.3f) · 이동성향 중앙 %.2f"
          % (args.out, float(np.median(ent)), float(np.log(len(ROOMS))),
             float(np.median(list(prop.values())))))


if __name__ == "__main__":
    main()
