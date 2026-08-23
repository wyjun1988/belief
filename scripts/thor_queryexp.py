#!/usr/bin/env python3
"""**질의 확장** — 물체마다 여러 표현을 만들어 OWL 앙상블.

    $P scripts/thor_queryexp.py --root data/thor2r --out data/thor_queries.json

지금은 `"a photo of a {물체}"` **하나**만 쓴다. OWL 은 개방어휘라 표현에 민감한데,
특히 **작은 물체**(연필 0.01 · 시계 0.01)는 표현 하나로는 못 잡을 수 있다.

⚠️ 비용이 거의 안 든다 — OWL 은 **이미지 인코더가 프레임당 1회**고 텍스트는 캐시된다.
어휘를 4배로 늘려도 추가 비용은 class_predictor 뿐이다.

Qwen 3.5 9B 로 물체별 표현 3개를 생성한다(색·재질·맥락을 넣은 구체적 묘사).
"""
import argparse, glob, json, os, re, subprocess


def words(t):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--batch", type=int, default=10)
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
        msg = ("/no_think For each object below, give %d short visual descriptions that would "
               "help an open-vocabulary object detector find it in an indoor photo. "
               "Vary wording: bare noun, with typical color/material, with typical context. "
               "Keep each under 6 words. JSON only: {\"Object\": [\"desc1\",\"desc2\",\"desc3\"]}\n"
               "Objects: %s" % (args.n, ", ".join(chunk)))
        body = json.dumps(dict(model="q", messages=[dict(role="user", content=msg)],
                               temperature=0.6, max_tokens=1200,
                               chat_template_kwargs=dict(enable_thinking=False)))
        r = subprocess.run(["curl", "-s", "-m", "300", args.url,
                            "-H", "Content-Type: application/json", "-d", body],
                           capture_output=True).stdout
        try:
            c = json.loads(r)["choices"][0]["message"].get("content", "")
            m = re.search(r"\{.*\}", c, re.S)
            d = json.loads(m.group()) if m else {}
        except Exception:
            d = {}
        for t in chunk:
            v = d.get(t)
            qs = [words(t)]
            if isinstance(v, list):
                qs += [str(x).strip().lower() for x in v if isinstance(x, str) and 2 < len(str(x)) < 60]
            out[t] = list(dict.fromkeys(qs))[:args.n + 1]
        print("  %d/%d" % (i + len(chunk), len(types)), flush=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    import numpy as np
    print("→ %s · 표현 수 중앙 %.1f" % (args.out, np.median([len(v) for v in out.values()])))
    for k in list(out)[:5]:
        print("   %-14s %s" % (k, out[k]))


if __name__ == "__main__":
    main()
