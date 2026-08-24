# 앵커 국소화를 **실전 검출**로. OWLv2 로 프레임마다 타입별 (최대점수, 패치위치) 를 뽑는다.
# 기존 캐시는 amax 로 위치를 버려서 앵커를 못 고른다 — 여기서는 argmax 패치도 남긴다.
import json, glob, os, sys, numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
DEV = os.environ.get("DEV") or ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
ROOT = os.environ.get("THOR_ROOT", "data/thor3")
OUT = os.environ.get("CACHE_PREFIX", "/tmp/a3_")
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 8
stat = json.load(open("data/thor_static_types.json"))
tg = set()
for hd in sorted(glob.glob(ROOT + "/house_*")):
    g = json.load(open(os.path.join(hd, "gt.json")))
    tg |= {v["type"] for v in g["gt0"].values()}
vocab = sorted(tg) + [s for s in stat if s not in tg]
nT = len(sorted(tg))
def sp(t): return "a photo of a " + "".join(" " + c.lower() if c.isupper() else c for c in t).strip()
op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
on = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEV).eval()
ti = op(text=[[sp(v) for v in vocab]], images=[Image.new("RGB", (256, 256), (128,)*3)],
        return_tensors="pt").to(DEV)
with torch.no_grad():
    o = on.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                 pixel_values=ti["pixel_values"], return_dict=True)
TX, MK = o.text_embeds, (ti["input_ids"][:, 0] > 0)
print("어휘 %d (타겟 %d + 정적 %d) · stride %d" % (len(vocab), nT, len(vocab)-nT, STRIDE), flush=True)
for hd in sorted(glob.glob(ROOT + "/house_*")):
    out = OUT + os.path.basename(os.path.realpath(hd)) + ".npz"
    if os.path.exists(out): continue
    lv = sorted(glob.glob(os.path.join(hd, "live", "*.jpg")))[::STRIDE]
    S = []; P = []
    for i in range(0, len(lv), 4):
        ims = [Image.open(p).convert("RGB") for p in lv[i:i+4]]
        pv = op(images=ims, return_tensors="pt")["pixel_values"].to(DEV)
        with torch.no_grad():
            fm = on.image_embedder(pixel_values=pv)[0]
            b, ph, pw, hdim = fm.shape
            lg, _ = on.class_predictor(fm.reshape(b, ph*pw, hdim),
                                       TX.unsqueeze(0).expand(b, -1, -1),
                                       MK.unsqueeze(0).expand(b, -1))
        pr = torch.sigmoid(lg)                      # (b, patches, types)
        S.append(pr.amax(1).float().cpu().numpy())
        P.append(pr.argmax(1).int().cpu().numpy())  # 패치 인덱스 → 화면 위치
        if i % 200 == 0: print("  %s %d/%d" % (os.path.basename(hd), i, len(lv)), flush=True)
    np.savez_compressed(out, s=np.concatenate(S), p=np.concatenate(P), ph=ph, pw=pw,
                        ts=np.array([int(os.path.basename(p)[:-4]) for p in lv]),
                        vocab=np.array(vocab, object), nT=nT)
    print("  %s 완료 %d장" % (os.path.basename(hd), len(lv)), flush=True)
print("완료")
