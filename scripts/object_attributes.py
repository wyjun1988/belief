#!/usr/bin/env python3
"""물체 attribute 일괄 생성 — 구축·갱신 시점 LLM **1회**로 여러 속성을 한 번에.

    $P scripts/object_attributes.py --out data/supermem/obj_attrs.json

이웃 선별만 하려고 LLM 을 부르는 것은 낭비다. 같은 호출에 belief·부재 층이
실제로 소비하는 속성을 함께 받는다(한계비용 ≈ 0):

    anchors     관측 후보 중 **안정적인** 것만 (부재 판정 문맥 · belief 수용체)
    mobility    fixed_appliance / furniture / portable / consumable
                → 물체별 근접 게이트 차등, 부재 판정 대상 선별
    home        정위치 수용체 이름 → belief 사전분포 초기값
    size        s / m / l → belief 입력 스키마의 기존 필드(현재 "s" 하드코딩)

⚠️ 원칙: **소비처가 실측으로 확인된 속성만** 넣는다. 동의어 표면형이 반례다 —
그럴듯했지만 켜니 belief 가 0.644→0.53 으로 악화됐다(미채택).

프롬프트는 영어다. 후보 목록 밖의 앵커는 버려 환각을 막는다.
"""
import argparse, json, os, re, sys, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SERVER = os.environ.get("LLAMA_SERVER", "http://192.168.219.123:8080/v1/chat/completions")
MOBILITY = ["fixed_appliance", "furniture", "portable", "consumable"]
SIZES = ["s", "m", "l"]


def ask(obj, cand, places):
    p = ('You annotate objects in a home for a memory assistant.\n\n'
         'Object: "%s"\n'
         'Things observed near it in this home: %s\n'
         'Known places in this home: %s\n\n'
         'Return ONE JSON object with exactly these keys:\n'
         '  "anchors": 2-5 items CHOSEN FROM the observed list — pick only STABLE '
         'landmarks (furniture, appliances, fixed places) useful to locate this object '
         'later. EXCLUDE things that move, get used up, or disappear.\n'
         '  "mobility": one of %s\n'
         '  "home": the single place from the known-places list where this object '
         'normally belongs\n'
         '  "size": one of %s (s=fits in a hand, m=two hands, l=furniture-sized)\n'
         'Output ONLY the JSON object.' % (obj, ", ".join(cand), ", ".join(places),
                                           MOBILITY, SIZES))
    b = json.dumps({"messages": [{"role": "user", "content": p}], "temperature": 0,
                    "max_tokens": 160,
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(SERVER, data=b, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=240) as z:
        txt = json.loads(z.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    d = json.loads(m.group(0))
    cs = set(cand)
    return dict(
        anchors=[w for w in map(str.lower, d.get("anchors") or []) if w in cs][:5],
        mobility=d.get("mobility") if d.get("mobility") in MOBILITY else None,
        home=(d.get("home") or "").lower() or None,
        size=d.get("size") if d.get("size") in SIZES else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "supermem", "obj_attrs.json"))
    ap.add_argument("--topm", type=int, default=12, help="LLM 에 보여줄 관측 후보 수")
    args = ap.parse_args()

    from scripts.absence_evidence import PLACES, pmi_graph, presence
    from scripts.supermem_answer import load_index
    D = os.path.join(ROOT, "data", "supermem")
    kwj = json.load(open(os.path.join(D, "v3_keywords.json")))

    def norm(w):
        t = [x for x in re.findall(r"[a-z]+", w.lower()) if len(x) > 1]
        return " ".join(t[-2:]) if t else w.lower()

    objs = sorted({norm(v["keyword"]) for v in kwj.values()})
    E, ts, sid = load_index()
    vocab = sorted(set(objs) | set(PLACES))
    Z, P = presence(E, vocab, "mps")
    G = pmi_graph(P, vocab)
    vi = {w: i for i, w in enumerate(vocab)}
    print("물체 %d · 어휘 %d (관측 후보는 PMI 상위 %d)" % (len(objs), len(vocab), args.topm))

    out = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for i, o in enumerate(objs):
        if o in out:
            continue
        cand = [vocab[j] for j in np.argsort(-G[vi[o]])[:args.topm] if G[vi[o], j] > 0]
        try:
            out[o] = ask(o, cand, PLACES)
        except Exception as e:
            print("  [%s] 오류 %s" % (o, str(e)[:60]), flush=True)
            out[o] = None
        if i % 10 == 0:
            json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
            print("  %d/%d" % (i, len(objs)), flush=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)

    ok = {k: v for k, v in out.items() if v}
    from collections import Counter
    print("\n생성 %d/%d" % (len(ok), len(objs)))
    print("  이동성 분포: %s" % dict(Counter(v["mobility"] for v in ok.values()).most_common()))
    print("  크기 분포:   %s" % dict(Counter(v["size"] for v in ok.values()).most_common()))
    print("  정위치 상위: %s" % dict(Counter(v["home"] for v in ok.values()).most_common(5)))
    print("\n표본:")
    for k in list(ok)[:6]:
        v = ok[k]
        print("  %-16s mob=%-16s home=%-14s size=%s  anchors=%s"
              % (k, v["mobility"], v["home"], v["size"], ", ".join(v["anchors"])))
    print("→ %s" % args.out)


if __name__ == "__main__":
    main()
