#!/usr/bin/env python3
"""OWL 소형 물체 검출 파일럿 — 타일 분할과 더 큰 검출기 (§166-98(c) 4번, 2026-09-21 · OWL_MODEL 추가 2026-09-22).

② 진짜 손실 중 '게이트 통과 ≤1' 인 물체에 대해, 이동 후 GT 가시 앵커 프레임에서
  전체 프레임(768, OWL 이 960 으로 키움) vs 2×2 타일(448, 겹침 128) vs 3×3 타일(384, 겹침 192)
로 OWL 을 다시 돌려 "타입 최고 상자가 점수 ≥0.2 이고 GT 중심 80px 이내" 인 프레임 수를 센다.
입력: bench-v2full 의 rows_DG_진단.jsonl + bench_DG_진단.log (C0_DIAG). 출력: 표.
"""
import json, os, sys, glob, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from owl_alias import alias
V = os.path.expanduser(os.environ.get("BENCH_DIR", "~/khcache/bench-v2full")); ROOT = os.environ.get("THOR_ROOT", "data/hssd_v2")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
from transformers import Owlv2Processor, Owlv2ForObjectDetection
OWL_MODEL = os.environ.get("OWL_MODEL", "google/owlv2-base-patch16-ensemble")   # large: google/owlv2-large-patch14-ensemble (2026-09-22 9-58)
op = Owlv2Processor.from_pretrained(OWL_MODEL)
on = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(DEV).eval()
print("검출기 %s · 장치 %s" % (OWL_MODEL, DEV), flush=True)
def sp(t): return "a photo of a " + "".join(" " + c.lower() if c.isupper() else c for c in alias(t)).strip()
def text_embed(typ):
    ti = op(text=[[sp(typ)]], images=[Image.new("RGB", (256, 256), (128,)*3)], return_tensors="pt").to(DEV)
    with torch.no_grad():
        o = on.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"], pixel_values=ti["pixel_values"], return_dict=True)
    return o.text_embeds, (ti["input_ids"][:, 0] > 0)
def detect(ims, TX, MK):
    """각 이미지에서 단일 쿼리 최고 점수·상자(cx,cy,w,h — 패딩 정방 정규화) 반환"""
    pv = op(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad():
        fm = on.image_embedder(pixel_values=pv)[0]; b, ph, pw, hdim = fm.shape
        lg, _ = on.class_predictor(fm.reshape(b, ph*pw, hdim), TX.unsqueeze(0).expand(b, -1, -1), MK.unsqueeze(0).expand(b, -1))
        pr = torch.sigmoid(lg)[:, :, 0]; am = pr.argmax(1)
        bxs = on.box_predictor(fm.reshape(b, ph*pw, hdim), feature_map=fm)
        sel = bxs[torch.arange(b), am]
    return pr.amax(1).float().cpu().numpy(), sel.float().cpu().numpy()
def tiles(im, n, size, offs):
    out = []
    for oy in offs:
        for ox in offs: out.append((ox, oy, im.crop((ox, oy, ox+size, oy+size))))
    return out
D = {}
for l in open(V + "/bench_DG_진단.log"):
    if l.startswith("C0_DIAG "): d = json.loads(l[8:]); D[(d["house"], d["oid"])] = d
R = {(r["house"], r["oid"]): r for r in (json.loads(l) for l in open(V + "/rows_DG_진단.jsonl"))}
cases = [k for k, r in R.items() if r["case"] == "②재촬영" and r["branch"] != "c0" and not r["ok"] and D.get(k, {}).get("n_pas", 9) <= 1]
if os.environ.get("HOUSES"): cases = [k for k in cases if k[0] in os.environ["HOUSES"].split()]
print("대상 %d건 · 장치 %s" % (len(cases), DEV), flush=True)
tot = {"full": 0, "t2": 0, "t3": 0, "n": 0}; rows = []
for hn, oid in cases:
    typ = R[(hn, oid)]["type"]; TX, MK = text_embed(typ)
    g = json.load(open(f"{ROOT}/{hn}/gt.json")); mv = next(m for m in g["moves"] if m["oid"] == oid); live = {m["t"]: m for m in g["live"]}
    ts = np.load(f"{V}/cache/hs2_a_{hn}.npz", allow_pickle=True)["ts"]
    frames = [int(t) for t in ts if t > mv["t"] and oid in live.get(int(t), {}).get("vis", []) and live[int(t)].get("ctr", {}).get(oid)]
    hit = {"full": 0, "t2": 0, "t3": 0}; best = {"full": [], "t2": [], "t3": []}
    for t in frames:
        im = Image.open(f"{ROOT}/{hn}/live/{t:06d}.jpg").convert("RGB"); W = max(im.size); c = live[t]["ctr"][oid]
        s, b = detect([im], TX, MK); cx, cy = float(b[0][0]) * W, float(b[0][1]) * W
        ok = s[0] >= 0.2 and np.hypot(cx - c[0], cy - c[1]) < 80; hit["full"] += ok; best["full"].append(float(s[0]))
        for key, size, offs in (("t2", 448, (0, 320)), ("t3", 384, (0, 192, 384))):
            tl = tiles(im, None, size, offs); s2, b2 = detect([x[2] for x in tl], TX, MK)
            j = int(np.argmax(s2)); ox, oy = tl[j][0], tl[j][1]
            cx, cy = ox + float(b2[j][0]) * size, oy + float(b2[j][1]) * size
            ok = s2[j] >= 0.2 and np.hypot(cx - c[0], cy - c[1]) < 80; hit[key] += ok; best[key].append(float(s2[j]))
    n = len(frames); tot["n"] += n
    for k in hit: tot[k] += hit[k]
    rows.append((hn, typ, n, hit["full"], hit["t2"], hit["t3"], round(float(np.median(best["full"])) if n else 0, 2), round(float(np.median(best["t2"])) if n else 0, 2), round(float(np.median(best["t3"])) if n else 0, 2)))
    print("  %-11s %-13s 프레임 %2d · 맞음(≥0.2·80px) 전체 %2d / 2×2 %2d / 3×3 %2d · 점수중앙 %.2f / %.2f / %.2f" % (rows[-1][0], rows[-1][1], *rows[-1][2:]), flush=True)
print("합계 프레임 %d · 맞음 전체 %d (%.2f) / 2×2 %d (%.2f) / 3×3 %d (%.2f)" % (tot["n"], tot["full"], tot["full"]/max(tot["n"],1), tot["t2"], tot["t2"]/max(tot["n"],1), tot["t3"], tot["t3"]/max(tot["n"],1)))
print("물체 단위(맞는 프레임 ≥2): 전체 %d / 2×2 %d / 3×3 %d / %d" % (sum(1 for r in rows if r[3] >= 2), sum(1 for r in rows if r[4] >= 2), sum(1 for r in rows if r[5] >= 2), len(rows)))
print("OWL_TILE_PILOT_DONE")
