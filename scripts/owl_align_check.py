#!/usr/bin/env python3
"""OWLv2 프레임 키가 CLIP 인덱스와 같은 프레임을 가리키는지 검증.

    $P scripts/owl_align_check.py --frames <추출프레임 루트>

SuperMemory 프레임은 비디오에서 index.npz 의 ts 로 되찾아 뽑았다. 이 대응이
어긋나면 CLIP/OWLv2 비교 전체가 무의미해진다.

**결정적 검증**: 추출한 프레임의 CLIP 임베딩이 index.npz 의 같은 행과 최대
코사인을 이루는가. 자기 자신이 최고가 아니면 정렬이 어긋난 것이다.

⚠️ 두 지각층의 z 상관으로 검증하려던 첫 판본은 쓸 수 없었다 — s8 에서 상관이
+0.083 밖에 안 나오는데, 확인해 보니 **정렬은 완벽했고**(6/6 자기 최고, 코사인
0.97~0.99) 낮은 상관은 CLIP 과 OWLv2 가 프레임 내용에 거의 합의하지 않는다는
**실제 결과**였다. 검증 지표가 측정 대상과 뒤섞이면 안 된다.
"""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True,
                    help="sm_<세션> 디렉터리들이 있는 루트(추출 시 쓴 경로)")
    ap.add_argument("--n", type=int, default=6, help="세션당 검사 프레임 수")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    nm = "openai/clip-vit-base-patch16"
    proc = CLIPProcessor.from_pretrained(nm)
    model = CLIPModel.from_pretrained(nm, use_safetensors=True).eval().to(args.device)

    bad = 0
    for sd in ("s1", "s8", "s14", "s19", "s20"):
        fdir = os.path.join(args.frames, "sm_" + sd)
        if not os.path.isdir(fdir):
            continue
        z = np.load(os.path.join(D, sd, "index.npz"))
        E = z["emb"].astype(np.float32)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        idxs = np.linspace(0, len(E) - 1, args.n).astype(int)
        hits, cos = 0, []
        for i in idxs:
            f = os.path.join(fdir, "%06d.jpg" % i)
            if not os.path.exists(f):
                continue
            with torch.no_grad():
                e = model.get_image_features(
                    **proc(images=Image.open(f).convert("RGB"), return_tensors="pt").to(args.device))
                e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()[0]
            s = E @ e
            hits += int(s.argmax() == i)
            cos.append(float(s[i]))
        ok = hits == len(cos) and cos
        bad += not ok
        print("  %-4s 프레임 %-5d · 자기최고 %d/%d · 코사인 중앙 %.3f  %s"
              % (sd, len(E), hits, len(cos), np.median(cos) if cos else 0,
                 "OK" if ok else "⚠️ 정렬 어긋남"))
    print("\n%s" % ("모든 세션 정렬 확인" if not bad else "**%d개 세션 정렬 실패**" % bad))


if __name__ == "__main__":
    main()
