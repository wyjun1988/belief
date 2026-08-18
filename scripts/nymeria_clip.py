#!/usr/bin/env python3
"""Nymeria 프레임의 CLIP 존재판정 — OWLv2 와 **같은 프레임**에서 대조군을 만든다.

    $P scripts/nymeria_clip.py

세션 간 belief 표(0.378 → 0.632)는 양쪽 다 OWLv2 였고 차이는 점수 문턱뿐이었다.
지각층 자체를 비교하려면 같은 프레임의 CLIP 판정이 있어야 한다. owl_det.json 의
`sec` 를 그대로 써서 프레임을 뽑고 CLIP z 를 낸다 — 프레임 집합이 일치해야
비교가 성립한다.
"""
import json, os, sys
import cv2, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "nymeria")

from scripts.absence_evidence import clip_text                    # noqa: E402


def main():
    det = json.load(open(os.path.join(D, "owl_det.json")))
    vocab = sorted({w for rows in det.values() for r in rows for w in r["det"]})
    print("어휘 %d (OWLv2 가 실제로 검출한 클래스 전체)" % len(vocab))
    import torch
    from transformers import CLIPModel, CLIPProcessor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    nm = "openai/clip-vit-base-patch16"
    proc = CLIPProcessor.from_pretrained(nm)
    model = CLIPModel.from_pretrained(nm, use_safetensors=True).eval().to(dev)
    V = clip_text(["a photo of a " + w for w in vocab], dev)

    out = {}
    for name, rows in sorted(det.items()):
        v = os.path.join(D, "loc49_rgb", name + ".mp4")
        if not os.path.exists(v):
            print("  %s 영상 없음 — 건너뜀" % name)
            continue
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        embs, secs = [], []
        for r in rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r["sec"] * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            im = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                t = proc(images=im, return_tensors="pt").to(dev)
                e = model.get_image_features(**t)
                embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy()[0])
            secs.append(r["sec"])
        if not embs:
            continue
        E = np.stack(embs)
        S = V @ E.T
        Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
        # OWLv2 와 같은 모양으로 저장 — z 를 점수처럼 쓴다
        out[name] = [{"sec": s, "det": {vocab[i]: round(float(Z[i, j]), 3)
                                        for i in range(len(vocab)) if Z[i, j] > 1.0}}
                     for j, s in enumerate(secs)]
        print("  %-44s %d프레임" % (name[:44], len(secs)), flush=True)
    json.dump(out, open(os.path.join(D, "clip_det.json"), "w"))
    print("→ data/nymeria/clip_det.json")


if __name__ == "__main__":
    main()
