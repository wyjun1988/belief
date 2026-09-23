#!/usr/bin/env bash
# 새 시뮬 5차분 평가 (M2) — 검증 점수(MLX) → 거리(DA) → 벤치 (표준 · 기록자리 판정기 없이 · 장면별)
set -u; cd ~/work/khronos; export KMP_DUPLICATE_LIB_OK=TRUE
K=$HOME/kx-venv/bin/python; MLX=$HOME/mlx-venv/bin/python; B=$HOME/khcache/bench-ns5; OUT=data/newsim5
echo "=== 1. 검증 점수 (MLX) $(date +%H:%M) ==="
[ -s $B/scores/t1_floor0.8_d40.jsonl ] || THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ ALL_TARGETS=1 FLOOR=0.8 MAXWALK=40 \
  OUT_JSONL=$B/scores/t1_floor0.8_d40.jsonl $MLX -u scripts/exp_t1_verify_mlx.py > $B/verify.log 2>&1
echo "  점수 $(wc -l < $B/scores/t1_floor0.8_d40.jsonl)줄"
echo "=== 2. 거리 (DA) $(date +%H:%M) ==="
[ -s $B/scores/geo_depth_nogt.jsonl ] || THOR_ROOT=$OUT A3_PREFIX=$B/cache/hs2_a_ QC_PREFIX=$B/cache/hs2_q_ AX_PREFIX=$B/cache/hs2_x_ FRAME_W=1280 \
  SCORES=$B/scores/t1_floor0.8_d40.jsonl OUT_JSONL=$B/scores/geo_depth_nogt.jsonl $K -u scripts/geo_depth.py > $B/geo_depth.log 2>&1
echo "  거리 $(wc -l < $B/scores/geo_depth_nogt.jsonl)줄"
echo "=== 3. 벤치 $(date +%H:%M) ==="
source scripts/champion_bench.sh
run_newsim5 ns5_표준
run_newsim5 ns5_기록자리판정기없이 PLACE_W=0
$K - <<'PY'
import json, os, collections
B=os.path.expanduser("~/khcache/bench-ns5")
R=[json.loads(l) for l in open(B+"/rows_CH_ns5_표준.jsonl")]
sc={}
for hd in sorted(os.listdir("data/newsim5")):
    p=os.path.join("data/newsim5",hd,"gt.json")
    if os.path.exists(p): sc[hd]=(json.load(open(p)).get("scene_meta") or {}).get("scene") or json.load(open(p)).get("scene","?")
by=collections.defaultdict(list)
for r in R: by[sc.get(r["house"],"?")].append(r)
print("  장면별 (표준):")
for s_,rs in sorted(by.items()):
    one=[r for r in rs if r["case"]=="①이동없음"]; two=[r for r in rs if r["case"]=="②재촬영"]; th=[r for r in rs if r["case"]=="③확인기회O"]
    f=lambda v: sum(1 for x in v if x["ok"])/max(len(v),1)
    print("   %-14s ① %.3f(n=%d) · ② %.3f(n=%d) · ③확인기회O %d 인계 %.2f" % (s_, f(one), len(one), f(two), len(two), len(th), sum(1 for x in th if x.get("branch")!="rec")/max(len(th),1)))
PY
echo "NS5_EVAL_DONE $(date +%H:%M)"
