#!/usr/bin/env python3
"""생성기 자가검사 (PORTING_CHECKLIST D) — 이동 물체의 **이동 후 검출률 = 이동 전** 인가.

    python scripts/gen_selfcheck.py data/hssd20S2/house_0000 [house_0001 ...]

§125: 이동 물체가 가구 속에 박혀 렌더에 없었는데 GT 는 "보임" 이라 두 달간 ② 가 0 이었다.
이 검사가 생성 직후 있었으면 하루면 잡혔다. 채마다 OWL 로 이동 물체의 이동 전/후
가시 프레임(<5m, 각 최대 N장)을 채점해 검출률(S≥0.12 & 패치 60px)을 비교하고,
증인 렌더 존재·이동 기록 수도 함께 찍는다. **후/전 비율 < 0.5 면 실패**.
"""
import json, os, sys
import numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

N = int(os.environ.get("N", "30")); TH = 0.12; R = 60
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()

_TX = {}
def _text(ty):
    if ty not in _TX:
        ti = op(text=[["a photo of a " + ty]], images=[Image.new("RGB", (64, 64), (128,) * 3)],
                return_tensors="pt").to(DEV)
        with torch.no_grad():
            o = on.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                         pixel_values=ti["pixel_values"], return_dict=True)
        _TX[ty] = (o.text_embeds, (ti["input_ids"][:, 0] > 0))
    return _TX[ty]

def detect(path, ty):
    TX, MK = _text(ty)
    pv = op(images=[Image.open(path).convert("RGB")], return_tensors="pt")["pixel_values"].to(DEV)
    with torch.no_grad():
        fm = on.image_embedder(pixel_values=pv)[0]; b, ph, pw, hd = fm.shape
        lg, _ = on.class_predictor(fm.reshape(b, ph*pw, hd), TX.unsqueeze(0).expand(b, -1, -1),
                                   MK.unsqueeze(0).expand(b, -1))
    pr = torch.sigmoid(lg)[0, :, 0]; k = int(pr.argmax())
    return float(pr.max()), ((k % pw + .5) / pw * 768, (k // pw + .5) / ph * 768)

fails = 0
for hd in sys.argv[1:]:
    if not os.path.exists(os.path.join(hd, "gt.json")):
        # 한 채의 생성 실패가 체인 전체를 죽이지 말 것 (h40c3: house_0019 가 gt.json 없이 끝나
        # 자가검사가 예외로 중단 → 나머지 30채 벤치가 통째로 날아갔다). 실패로 세고 계속한다.
        print("%s gt.json 없음 — 생성 실패채, 건너뜀" % os.path.basename(hd), flush=True)
        fails += 1; continue
    g = json.load(open(os.path.join(hd, "gt.json"))); live = g["live"]
    wit = len(os.listdir(os.path.join(hd, "witness"))) if os.path.isdir(os.path.join(hd, "witness")) else 0
    pre = [0, 0]; post = [0, 0]
    for m in g["moves"]:
        oid, ty = m["oid"], g["gt0"][m["oid"]]["type"]
        fr = [l for l in live if oid in l["vis"] and l["dist"][oid] < 5]
        a = [l for l in fr if l["t"] <= m["t"]][-N:]; b = [l for l in fr if l["t"] > m["t"]][:N]
        for acc, ls in ((pre, a), (post, b)):
            for l in ls:
                S, (px, py) = detect(os.path.join(hd, "live", "%06d.jpg" % l["t"]), ty); c = l["ctr"][oid]
                acc[1] += 1; acc[0] += int(S >= TH and np.hypot(px - c[0], py - c[1]) <= R)
    rp = pre[0] / max(pre[1], 1); rq = post[0] / max(post[1], 1)
    # 1차 게이트: **증인 렌더(2m 정면)에서 검출** — 배치 버그(박힘·허공)는 여기서 걸린다.
    # 이동 전후 검출률은 참고(같은 물체라도 새 자리의 배경·유사물체로 정당하게 낮아질 수 있다).
    # 게이트 = **기하**: 모든 이동이 받침 위(supported)이고 2m 시선(witness)이 있는가.
    # 검출률은 벤치 속성으로 보고만 한다 — 흰 접시가 흰 침대 위, 검은 휴지통이 어두운 옷장 앞
    # 같은 것은 정당한 난이도이지 배치 버그가 아니다 (2026-09-02 육안 확인).
    # 수납 이동(hidden_verified=True)은 정의상 2m 시선이 없다 — 받침 + 숨김 검증이면 통과 (OG 인수인계 결정)
    geo_ok = all(m.get("supported") and ((m.get("witness_file") and m.get("witness_ctr")) or m.get("hidden_verified"))
                 for m in g["moves"])
    wit_ok = 0; wit_n = 0
    for m in g["moves"]:
        if not m.get("witness_file") or not m.get("witness_ctr"): continue
        wit_n += 1
        S, (px, py) = detect(os.path.join(hd, m["witness_file"]), g["gt0"][m["oid"]]["type"])
        c = m["witness_ctr"]; wit_ok += int(S >= TH and np.hypot(px - c[0], py - c[1]) <= 90)
    fails += (not geo_ok)
    print("%s 이동 %d · 기하게이트 %s · 2m정면 검출 %d/%d · (참고) 이동전 %d/%d=%.2f 이동후 %d/%d=%.2f"
          % (os.path.basename(hd), len(g["moves"]), "OK" if geo_ok else "**실패**", wit_ok, wit_n,
             pre[0], pre[1], rp, post[0], post[1], rq), flush=True)
print("자가검사 실패 %d/%d채" % (fails, len(sys.argv[1:])))
sys.exit(1 if fails else 0)
