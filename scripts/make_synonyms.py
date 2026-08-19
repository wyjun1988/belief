#!/usr/bin/env python3
"""개념별 표면형(동의어) 메타데이터 생성 — 검출기에 '여러 말로' 물어보기 위해서.

    $P scripts/make_synonyms.py --seqs <a> <b> --out data/adt_owl/synonyms.json

왜 재라벨링이 아니라 질의 확장인가: ADT GT 에는 `picture` 와 `wall artwork`,
`table` 과 `dining table` 과 `side table` 이 **각각 독립 클래스**로 있다. 검출
결과를 `picture → wall artwork` 로 옮기면 한 검출이 두 클래스에 중복 계상되어
재현율만 공짜로 오른다(실측으로 확인 — 실제 개선이 아니었다).

올바른 방법은 **개념마다 여러 표면형으로 질의하고 최댓값을 취하는 것**이다.
오픈보캡 검출기에서 어휘는 곧 질의이므로, 같은 물체를 여러 방식으로 부르는 것은
정당한 사용이고 중복 계상도 없다(점수를 옮기는 게 아니라 더 잘 물어보는 것).

⚠️ **충돌 제거가 핵심이다.** 어떤 표면형이 (a) 다른 개념의 정식 이름이거나
(b) 두 개념이 동시에 주장하면 버린다. 안 버리면 `picture` 를 `wall artwork` 의
표면형으로 넣는 순간 위의 중복 계상이 그대로 재현된다.
"""
import argparse, json, os, re, sys, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SERVER = os.environ.get("LLAMA_SERVER", "http://192.168.219.123:8080/v1/chat/completions")


def ask(concept, others, n=5):
    p = ("You name objects for an open-vocabulary object detector. "
         "Give %d different SHORT noun phrases a detector could use to find this object "
         "in a photo of a home:\n\n  %s\n\n"
         "Rules: each phrase 1-4 words, lowercase, visually specific, no explanations. "
         "Do NOT use any of these words as a whole phrase (they name different objects): %s\n"
         "Output only the phrases, comma-separated."
         % (n, concept, ", ".join(others[:40])))
    b = json.dumps({"messages": [{"role": "user", "content": p}], "temperature": 0,
                    "max_tokens": 120,
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(SERVER, data=b, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as z:
        txt = json.loads(z.read())["choices"][0]["message"]["content"]
    out = []
    for w in re.split(r"[,\n]", txt):
        w = re.sub(r"^[\s\-\*\d\.\)]+", "", w).strip().lower().strip('."\'')
        if w and 1 <= len(w.split()) <= 4 and re.fullmatch(r"[a-z ]+", w):
            out.append(w)
    return out


def gt_categories(seq, root):
    gt = json.load(open(os.path.join(root, seq, "gt", "objects.json")))["instances"]
    return {(r.get("category") or "").strip().lower() for r in gt.values() if r.get("category")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", default=[
        "Apartment_release_decoration_seq137_M1292",
        "Apartment_release_multiskeleton_party_seq102_M1292"])
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "adt_owl", "synonyms.json"))
    args = ap.parse_args()

    concepts = set()
    for s in args.seqs:
        concepts |= gt_categories(s, args.root)
    concepts = sorted(c for c in concepts if len(c) > 2)
    print("개념 %d개" % len(concepts), flush=True)

    raw = {}
    for i, c in enumerate(concepts):
        others = [o for o in concepts if o != c]
        try:
            raw[c] = ask(c, others, args.n)
        except Exception as e:
            print("  [%s] 오류 %s" % (c, e), flush=True)
            raw[c] = []
        if i % 15 == 0:
            print("  %d/%d" % (i, len(concepts)), flush=True)

    # ── 충돌 제거 ────────────────────────────────────────────────────
    canon = set(concepts)
    claims = {}
    for c, ws in raw.items():
        for w in set(ws):
            claims.setdefault(w, set()).add(c)
    out, n_kept, n_drop_canon, n_drop_multi = {}, 0, 0, 0
    for c in concepts:
        keep, dropped = [c], []
        for w in dict.fromkeys(raw.get(c, [])):
            if w == c:
                continue
            if w in canon:                      # 다른 개념의 정식 이름
                dropped.append((w, "정식이름"))
                n_drop_canon += 1
            elif len(claims.get(w, ())) > 1:    # 두 개념이 동시에 주장
                dropped.append((w, "중복주장"))
                n_drop_multi += 1
            else:
                keep.append(w)
        out[c] = {"surface": keep, "dropped": dropped}
        n_kept += len(keep)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("\n표면형 %d개 (개념당 평균 %.1f) · 버림 — 정식이름 충돌 %d · 중복주장 %d"
          % (n_kept, n_kept / len(concepts), n_drop_canon, n_drop_multi))
    for c in concepts[:6]:
        print("  %-22s %s" % (c, ", ".join(out[c]["surface"][1:]) or "(표면형 없음)"))
    print("→ %s" % args.out)


if __name__ == "__main__":
    main()
