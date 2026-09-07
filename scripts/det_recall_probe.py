#!/usr/bin/env python3
"""스캔 단계 검출기 재현율 비교 — 지도 프레임의 GT 박스(gt.json map[k].box)마다 텍스트 질의 검출이 IoU≥0.3·점수≥τ 로 맞히는가. 박스 크기별(작은<64px·중간<160·큰).
    THOR_ROOT=data/hssd90_c4e2 DET=owlv2-base FRAMES=40 OUT_JSONL=~/khcache/det_recall_owlb.jsonl python scripts/det_recall_probe.py
DET: owlv2-base | owlv2-large | gdino-base (transformers). 결과 jsonl + 요약 표. GPU 면 CUDA, 맥이면 MPS."""
import glob, json, os, time, random, collections, re
import numpy as np, torch
from PIL import Image
ROOT = os.environ.get("THOR_ROOT", "data/hssd90_c4e2"); DET = os.environ.get("DET", "owlv2-base"); FRAMES = int(os.environ.get("FRAMES", "40")); TAU = float(os.environ.get("TAU", "0.1"))
OUTJ = os.path.expanduser(os.environ.get("OUT_JSONL", "~/khcache/det_recall_%s.jsonl" % DET)); MAXH = int(os.environ.get("MAXH", "6"))
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"); random.seed(0)
def words(t): return re.sub(r"\|.*$", "", t).replace("_", " ").lower()
if DET.startswith("owlv2"):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    ck = {"owlv2-base": "google/owlv2-base-patch16-ensemble", "owlv2-large": "google/owlv2-large-patch14-ensemble"}[DET]
    proc = Owlv2Processor.from_pretrained(ck); mdl = Owlv2ForObjectDetection.from_pretrained(ck).to(dev).eval()
    def detect(img, types):
        inp = proc(text=[["a photo of a " + t for t in types]], images=img, return_tensors="pt").to(dev)
        with torch.no_grad(): out = mdl(**inp)
        W, H = img.size; S = max(W, H)                                   # OWLv2 는 정방 패딩 기준 정규화 박스
        res = proc.post_process_object_detection(out, threshold=0.0, target_sizes=torch.tensor([[S, S]]).to(dev))[0]
        return [(int(l), float(s), [float(v) for v in b]) for l, s, b in zip(res["labels"].cpu(), res["scores"].cpu(), res["boxes"].cpu())]
else:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    ck = "IDEA-Research/grounding-dino-base"; proc = AutoProcessor.from_pretrained(ck); mdl = AutoModelForZeroShotObjectDetection.from_pretrained(ck).to(dev).eval()
    def detect(img, types):
        text = ". ".join(types) + "."
        inp = proc(images=img, text=text, return_tensors="pt").to(dev)
        with torch.no_grad(): out = mdl(**inp)
        W, H = img.size
        res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.0, text_threshold=0.0, target_sizes=[(H, W)])[0]
        labs = res.get("text_labels", res.get("labels")); dets = []
        for lab, s, b in zip(labs, res["scores"].cpu(), res["boxes"].cpu()):
            lab = lab if isinstance(lab, str) else str(lab)
            li = next((i for i, t in enumerate(types) if t in lab or lab in t), None)
            if li is not None: dets.append((li, float(s), [float(v) for v in b]))
        return dets
def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3]); inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter; return inter / ua if ua > 0 else 0
rows = []; t_all = []
houses = sorted(glob.glob(ROOT + "/house_*"))[:MAXH]
for hd in houses:
    g = json.load(open(os.path.realpath(hd) + "/gt.json")); typ = {oid: v["type"] for oid, v in g["gt0"].items()}
    ks = [k for k, m in enumerate(g["map"]) if m.get("box")]; random.shuffle(ks); ks = sorted(ks[:FRAMES])
    for k in ks:
        m = g["map"][k]; img = Image.open("%s/map/%04d.jpg" % (hd, k)).convert("RGB"); W, H = img.size
        boxes = {oid: b for oid, b in m["box"].items() if oid in typ and len(b) == 4 and (b[2]-b[0]) >= 8 and (b[3]-b[1]) >= 8}
        if not boxes: continue
        types = sorted({words(typ[oid]) for oid in boxes}); t0 = time.time(); dets = detect(img, types); t_all.append(time.time() - t0)
        for oid, b in boxes.items():
            ti = types.index(words(typ[oid])); cand = [(s, iou(b, bb)) for li, s, bb in dets if li == ti]
            best = max(cand, key=lambda x: (x[1] >= 0.3, x[0])) if cand else (0.0, 0.0)
            hit = any(s >= TAU and io >= 0.3 for s, io in cand); sz = max(b[2]-b[0], b[3]-b[1])
            rows.append(dict(house=os.path.basename(hd), k=k, oid=oid, type=typ[oid], size=sz, hit=hit, best_iou=round(best[1], 3), best_score=round(best[0], 3), maxscore=round(max([s for s, io in cand], default=0.0), 3)))
with open(OUTJ, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("%s · %s · 집 %d · 프레임 %d · GT 박스 %d · 프레임당 %.2fs (%s)" % (DET, ROOT, len(houses), len(t_all), len(rows), np.mean(t_all) if t_all else 0, dev))
for lo, hi, nm in ((0, 64, "작은 <64px"), (64, 160, "중간 64~160"), (160, 1e9, "큰 ≥160")):
    sel = [r for r in rows if lo <= r["size"] < hi]
    if sel: print("  %-12s n=%4d 재현율(IoU≥0.3·s≥%.2f) %.2f · IoU≥0.3 (점수 무관) %.2f · 최고 점수 중앙 %.2f" % (nm, len(sel), TAU, np.mean([r["hit"] for r in sel]), np.mean([r["best_iou"] >= 0.3 for r in sel]), np.median([r["maxscore"] for r in sel])))
bt = collections.defaultdict(list)
for r in rows:
    if r["size"] < 64: bt[r["type"]].append(r["hit"])
print("  작은 물체 타입별 재현율(n≥5):", "; ".join("%s %.2f(%d)" % (t, np.mean(v), len(v)) for t, v in sorted(bt.items(), key=lambda x: -len(x[1])) if len(v) >= 5)[:300])
print("DET_RECALL_PROBE_DONE →", OUTJ)
