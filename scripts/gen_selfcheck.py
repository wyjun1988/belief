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

def detect(path, ty):
    ti = op(text=[["a photo of a " + ty]], images=[Image.open(path).convert("RGB")], return_tensors="pt").to(DEV)
    with torch.no_grad():
        o = on.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                     pixel_values=ti["pixel_values"], return_dict=True)
        fm = o.image_embeds; b, ph, pw, hd = fm.shape
        lg, _ = on.class_predictor(fm.reshape(b, ph*pw, hd), o.text_embeds.unsqueeze(0).expand(b, -1, -1),
                                   (ti["input_ids"][:, 0] > 0).unsqueeze(0).expand(b, -1))
    pr = torch.sigmoid(lg)[0, :, 0]; k = int(pr.argmax())
    return float(pr.max()), ((k % pw + .5) / pw * 768, (k // pw + .5) / ph * 768)

fails = 0
for hd in sys.argv[1:]:
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
    ok = post[1] == 0 or rq >= 0.5 * rp
    fails += (not ok)
    print("%s 이동 %d · 증인 %d · 검출률 이동전 %d/%d=%.2f  이동후 %d/%d=%.2f  → %s"
          % (os.path.basename(hd), len(g["moves"]), wit, pre[0], pre[1], rp, post[0], post[1], rq,
             "OK" if ok else "**실패(배치 버그 의심)**"), flush=True)
print("자가검사 실패 %d/%d채" % (fails, len(sys.argv[1:])))
sys.exit(1 if fails else 0)
