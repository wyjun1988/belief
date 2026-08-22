#!/usr/bin/env python3
"""물체 배치 사전확률을 **Qwen 3.5 9B 로 다양화**한다.

    $P scripts/thor_prior_llm.py --root data/thor2 --out data/thor_prior.json

⚠️ **왜 필요한가.** ProcTHOR 는 물체를 유형 규칙으로 배치한다("머그는 부엌").
그래서 사전확률이 실제보다 강해지고, ㊻ 에서 **belief 단독(0.632)이 전체 시스템
(0.541)을 이기는** 결과가 나왔다. 그 결론의 최대 약점이 이 편향이다.

LLM 에 "실제 가정에서 이 물체가 있을 법한 방 분포" 를 물어 **현실적으로 퍼진**
분포를 얻고, 그것으로 t=0 배치를 다시 한다. `Pillow`(침실 0.90)처럼 확실한 것은
몰아주고 `CellPhone`·`Book` 은 고르게 퍼뜨리는 — 실제와 비슷한 구조가 된다.

⚠️ Qwen 3.5 는 thinking 모델이라 `enable_thinking=false` 를 줘야 한다.
안 그러면 추론에 토큰을 다 쓰고 답이 잘린다.
"""
import argparse, glob, json, os, re, subprocess

ROOMS = ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]


def ask(url, types, temp):
    msg = ("/no_think Rooms: %s. For each object give a realistic probability "
           "distribution over rooms in a real home. Spread it out where the object "
           "could plausibly be in several rooms. JSON only, no explanation.\nObjects: %s"
           % (", ".join(ROOMS), ", ".join(types)))
    body = json.dumps(dict(model="q", messages=[dict(role="user", content=msg)],
                           temperature=temp, max_tokens=1400,
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
        types |= {v["type"] for v in g.get("gt0", g.get("gt1", {})).values()}
    types = sorted(types)
    print("물체 유형 %d" % len(types), flush=True)

    out = {}
    for i in range(0, len(types), args.batch):
        chunk = types[i:i + args.batch]
        d = ask(args.url, chunk, args.temp)
        for t in chunk:
            v = d.get(t)
            if isinstance(v, dict):
                s = sum(float(v.get(r, 0)) for r in ROOMS)
                if s > 0:
                    out[t] = {r: float(v.get(r, 0)) / s for r in ROOMS}
        print("  %d/%d · 누적 %d" % (i + len(chunk), len(types), len(out)), flush=True)
    # 실패분은 균등
    for t in types:
        out.setdefault(t, {r: 1.0 / len(ROOMS) for r in ROOMS})
    json.dump(out, open(args.out, "w"), indent=1)
    import numpy as np
    ent = [-sum(p * np.log(p + 1e-9) for p in v.values()) for v in out.values()]
    print("→ %s · 분포 엔트로피 중앙 %.3f (균등이면 %.3f)"
          % (args.out, float(np.median(ent)), float(np.log(len(ROOMS)))))


if __name__ == "__main__":
    main()
