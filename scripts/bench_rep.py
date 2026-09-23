#!/usr/bin/env python3
"""벤치 행 파일 요약 한 줄 — 채택을 정답/해로움으로 쪼갠다 (2026-09-23).

종전 스크래치 템플릿의 "거짓채택" 열은 ① 에서 채택(c0)된 **전부**를 셌다. 채택이 틀린 기록을 고친
경우(정답)까지 거짓으로 세서, 기록 교정이 많은 변경일수록 비용이 부풀려 보였다. 이제 ① 채택을
정답·해로움으로 나눠 찍는다. 경우 이름은 eval_online 행 덤프 그대로.
  python scripts/bench_rep.py <tag> <rows.jsonl>
"""
import json, sys
tag, p = sys.argv[1], sys.argv[2]
r = [json.loads(l) for l in open(p)]
one = [x for x in r if x["case"] == "①이동없음"]; two = [x for x in r if x["case"] == "②재촬영"]
th = [x for x in r if x["case"] == "③확인기회O"]
c1 = [x for x in one if x.get("branch") == "c0"]; c2 = [x for x in two if x.get("branch") == "c0"]
f = lambda v: sum(1 for x in v if x["ok"]) / max(len(v), 1)
print("%-14s ① %.3f(n=%d · 채택 %d = 정답 %d + 해로움 %d · 거짓인계 %d) · **② %.3f(n=%d · 채택 %d = 정답 %d + 오답 %d)** · ③인계 %.2f(n=%d)" % (
    tag, f(one), len(one), len(c1), sum(1 for x in c1 if x["ok"]), sum(1 for x in c1 if not x["ok"]),
    sum(1 for x in one if x.get("branch") == "c2"),
    f(two), len(two), len(c2), sum(1 for x in c2 if x["ok"]), sum(1 for x in c2 if not x["ok"]),
    sum(1 for x in th if x.get("branch") != "rec") / max(len(th), 1), len(th)), flush=True)
