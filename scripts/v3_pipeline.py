#!/usr/bin/env python3
"""v3 엔드투엔드 — 자연어 한 줄 → 답.

    $P scripts/v3_pipeline.py --seq <adt-seq> --ask "액자 지금 어디 있어?"
    $P scripts/v3_pipeline.py --seq <adt-seq> --demo        # 대표 질의 몇 개

파이프라인(사용자 확정 설계):

    자연어 질문
      → ① Qwen 키워드 추출(문법 JSON, thinking off) — keyword + 이웃
      → ② 라우팅: 미관측 추론(belief) 인가, 과거 이미지로 답할 QA 인가
      → ③-A belief: home-jepa 분포 × 씬그래프(방·가구) → 확률 순위 + 근거
        ③-B QA   : 확장 질의로 프레임 검색(인과 마스크) → Qwen VLM 답변
      → ④ 자연어 정리

라우팅 규칙은 단순하다 — "지금/어디" 같은 현재 상태 질문은 belief(미관측 추론),
"언제/누가/무엇을 했나" 는 과거 관측 QA. 애매하면 belief 를 먼저 시도한다.
"""
import argparse
import json
import os
import re
import sys
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SERVER = os.environ.get("LLAMA_SERVER",
                        "http://192.168.219.123:8080/v1/chat/completions")

KW_GRAMMAR = r'''
root ::= "{" ws "\"keyword\"" ws ":" ws str ws "," ws "\"neighbors\"" ws ":" ws arr ws "," ws "\"kind\"" ws ":" ws kind ws "}"
arr ::= "[" ws str (ws "," ws str){0,4} ws "]"
kind ::= "\"belief\"" | "\"qa\""
str ::= "\"" [a-z ]{1,40} "\""
ws ::= [ \n]?
'''

KW_PROMPT = """You plan queries for a household memory assistant.
From the user's question extract:
- "keyword": the object being asked about (1-3 lowercase words)
- "neighbors": 2-5 lowercase words for objects/places it is usually near
- "kind": "belief" if the question asks where something IS NOW (current, unobserved
  state, or where it will be), "qa" if it asks about a past observed event.
Question: %s
Respond with JSON only."""


def call(prompt, grammar=None, images=None, max_tokens=200, timeout=300):
    content = [{"type": "text", "text": prompt}]
    for im in images or []:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + im}})
    payload = {"messages": [{"role": "user", "content": content}],
               "temperature": 0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    if grammar:
        payload["grammar"] = grammar
    req = urllib.request.Request(SERVER, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def plan(question):
    """① 키워드 추출 + 라우팅"""
    return json.loads(call(KW_PROMPT % question, KW_GRAMMAR, max_tokens=100))


def run_belief(args, kw):
    """③-A belief — 씬그래프(방·가구) 위에서 확률 순위"""
    import torch
    from scripts.belief_engine import (build_Z, camel_words, load_model,
                                       prepare, rank_for)
    sd = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    ep, E_t, g, gt, unk = prepare(sd, args)
    dev = torch.device(args.device)
    model = load_model(args.model, dev)
    name2oid = {}
    for o in ep["home"]["objects"]:
        src = g["objects"].get(o["src_instance"], {})
        name2oid.setdefault((src.get("name") or "").lower(), []).append(o["id"])
    key = kw["keyword"].replace(" ", "")
    hits = {n: v for n, v in name2oid.items()
            if n and (key in n.replace(" ", "") or any(w in n for w in kw["keyword"].split()))}
    if not hits:
        return None, "씬그래프에 '%s' 가 없다 (있는 것 예: %s)" % (
            kw["keyword"], ", ".join(sorted(x for x in name2oid if x)[:6]))
    oids = {o for v in hits.values() for o in v}
    qs = [qi for qi in range(len(E_t.queries)) if E_t.queries[qi]["meta"]["obj"] in oids]
    if not qs:
        return None, "'%s' 는 관측 이력이 없어 추론할 근거가 없다" % kw["keyword"]
    Z, fidx = build_Z(sd, ep, E_t, camel_words(list(hits)[0]), args.device)
    rows, q = rank_for(ep, E_t, model, dev, oids, qs[-1], Z, fidx,
                       camel_words(list(hits)[0]), args.gamma, args.gate, args.device)
    return rows, None


def summarize(question, rows):
    """④ 확률 순위 → 자연어 (Qwen)"""
    cand = "\n".join("- %s (%s): %.0f%%  [%s]"
                     % (r["recept"], r["room"], 100 * r["p"], r["tag"]) for r in rows[:3])
    p = ("A home assistant estimated where an object is now.\n"
         "Question: %s\nRanked candidates with evidence:\n%s\n"
         "Answer the user in one short sentence, mentioning the most likely place "
         "and how confident you are." % (question, cand))
    return call(p, max_tokens=120).strip()


def answer(args, question):
    kw = plan(question)
    print("① 계획: keyword='%s' · 이웃=%s · 유형=%s"
          % (kw["keyword"], ", ".join(kw["neighbors"]), kw["kind"]))
    if kw["kind"] == "belief" or args.force_belief:
        rows, err = run_belief(args, kw)
        if err:
            print("② belief 불가 — %s" % err)
            return
        print("② belief 후보:")
        for r in rows[:3]:
            print("   %5.1f%%  %-22s (%s)  [%s]"
                  % (100 * r["p"], r["recept"][:22], r["room"], r["tag"]))
        print("③ 답: %s" % summarize(question, rows))
    else:
        print("② 과거 관측 QA 경로 — 확장 질의로 프레임 검색 후 VLM 답변")
        print("   (ADT 는 QA 주석이 없어 이 경로는 SuperMemory 하니스에서 채점됨:")
        print("    strict 66문항 물체·위치 0.62, 검색 hit@5 0.70)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Apartment_release_multiskeleton_party_seq102_M1292")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth")
    ap.add_argument("--ref", default="gtdepth")
    ap.add_argument("--model", default="supervised_two_head_v5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--open-vocab", action="store_true", default=True)
    ap.add_argument("--clip-class", action="store_true")
    ap.add_argument("--unknown-class", action="store_true")
    ap.add_argument("--unknown-room", action="store_true")
    ap.add_argument("--force-belief", action="store_true")
    ap.add_argument("--ask", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    qs = [args.ask] if args.ask else [
        "Where is the ceramic mug now?",
        "Where did I leave the white vase?",
        "Who used the coffee table earlier?",
    ] if args.demo else []
    for q in qs:
        print("\n" + "=" * 72)
        print("질문: %s" % q)
        answer(args, q)


if __name__ == "__main__":
    main()
