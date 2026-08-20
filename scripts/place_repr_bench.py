#!/usr/bin/env python3
"""장소를 가르는 표현을 찾는다 — **물체를 복원할 필요가 없다**는 점을 이용한다.

    $P scripts/place_repr_bench.py --root <benchmark/data>

실측(시점 프로브): CLIP CLS 는 장소를 못 가른다. 같은 장소 v1↔v2 0.933 인데
**남의 장소 최고가 0.903** 으로 여백이 +0.017 뿐이고, 같은 장소를 각도만 바꿔 본
것(0.909)이 남의 장소보다도 **덜 닮는다**(시점여백 −0.004).

장소 검색에는 물체를 복원할 필요가 없으므로, **물체 정보를 버리고 배치·구조를
남기는 표현**이 후보가 된다. 같은 프레임·같은 자로 비교한다:

  clip_cls     CLIP CLS (기준선)
  clip_patch   CLIP 패치평균 — CLS 의 전역 요약을 우회
  clip_g2      CLIP 패치를 **2×2 구역별** 평균해 이어붙임 — 배치 정보 보존
  dino_cls     DINOv2 CLS — 장소 인식·밀집 대응에서 CLIP 보다 강하다고 알려짐
  dino_patch   DINOv2 패치평균
  dino_g2      DINOv2 2×2 구역
  ijepa        I-JEPA (ViT-H/14) — JEPA 이미지판. CLS 가 없어 패치평균/2×2 로 쓴다
  vjepa2       **V-JEPA2 (ViT-L, fpc64/256)** — JEPA 영상판. 가린 시공간 영역의
               **표현**을 문맥에서 예측하도록 학습돼, 지엽적 물체보다 장면 구조를
               남긴다. 우리가 필요한 "시점 불변 + 물체 불변 장소 표현"에 가장 가깝다.
               프레임 하나가 아니라 **방문 구간(짧은 영상)** 을 한 표현으로 만든다.
  dino_vlad    **AnyLoc 계열** — DINOv2 패치 특징을 **VLAD** 로 집계한다. 전역 평균
               (CLS·패치평균)은 시점이 바뀌면 통째로 흔들리지만, VLAD 는 패치를
               시각 어휘에 배정하고 **어휘별 잔차를 누적**하므로 어느 패치가 어디
               있었는지에 덜 의존한다 — 우리가 못 넘는 지표(시점여백)를 정면으로
               겨냥한 집계다. 학습 불필요.
  owlvec       **물체 조합 벡터** — OWLv2 점수를 어휘 전체에 걸쳐 모은 것.
               "장소는 그 안의 물체 조합이다" 를 직접 쓴다. 근거: ADT 방 서명
               (TF-IDF 물체 조합 투표)이 **0.971** 로 GT 사전(0.966)과 동률이었다.
               ⚠️ 앞서 이것이 실패했던 것(장소 정밀도 0.37)은 어휘를 **변화한 물체
               38개**로만 잡았기 때문이다 — 문맥이 주변 가구가 아니라 다른 장면의
               물체가 됐다. 여기서는 어휘 1,037개(`owl_vocab.json`)를 쓴다.
               **이미 저장하는 메타**라 추가 비용이 0 이라는 점이 중요하다.
  owlvec_idf   같은 벡터에 IDF 가중 — 어디에나 있는 물체(wall·floor)의 비중을 낮춘다
  tiny         32×32 회색조 축소 — 저주파 구조만. 비용 거의 0
  colorlay     3×3 구역 × 색 히스토그램 — 배치+색. 비용 거의 0

⚠️ 환경 우회 두 가지(둘 다 stock-v2 venv 를 건드리지 않는다):
  · DINOv2 는 MPS 에 `upsample_bicubic2d` 가 없어 죽는다 → `PYTORCH_ENABLE_MPS_FALLBACK=1`
  · V-JEPA2 의 `AutoVideoProcessor` 는 torchvision 을 요구한다 → 전처리를 PIL+numpy 로
    직접 한다(짧은변 292 → 256 중앙크롭 → ImageNet 정규화)

판정 세 거리(시점 프로브와 동일):
  ① 같은 클립 안(시점만 다름) · ② 같은 장소 v1↔v2 · ③ 남의 장소 최고
  **여백 ②−③ 이 커야** 장소를 찾을 수 있고, **시점여백 ①−③ 이 양수여야**
  각도 변화에 견딘다.
"""
import argparse, glob, os, subprocess, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.scenediff_absence import frames_of, video_of   # noqa: E402


def nz(x):
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--nframes", type=int, default=8)
    ap.add_argument("--limit", type=int, default=999)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--reps", default="clip_cls,clip_patch,clip_g2,dino_cls,dino_patch,"
                                      "dino_g2,dino_vlad,ijepa,vjepa2,owlvec,owlvec_idf,tiny,colorlay")
    ap.add_argument("--vlad-k", type=int, default=32, help="VLAD 시각 어휘 수")
    ap.add_argument("--owl-vocab", default=os.path.join(ROOT, "data", "supermem", "owl_vocab.json"),
                    help="물체 조합 벡터용 어휘(1,037단어)")
    ap.add_argument("--owl-thr", type=float, default=0.05)
    ap.add_argument("--vjepa", default="facebook/vjepa2-vitl-fpc64-256",
                    help="V-JEPA2 체크포인트. GPU 가 크면 vjepa2-vitg-fpc64-384 도 가능")
    ap.add_argument("--clip", default="openai/clip-vit-base-patch16")
    ap.add_argument("--dino", default="facebook/dinov2-base")
    ap.add_argument("--vj-frames", type=int, default=4,
                    help="V-JEPA2 절반당 프레임(CPU 라 비용이 크다)")
    args = ap.parse_args()
    reps = args.reps.split(",")

    import torch
    from PIL import Image
    tmp = tempfile.mkdtemp()

    pairs = []
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if len(pairs) >= args.limit or not os.path.isdir(d):
            continue
        v1, v2 = video_of(d, 1), video_of(d, 2)
        if v1 and v2:
            pairs.append((os.path.basename(d), v1, v2))
    print("장소 %d곳 · 표현 %d종" % (len(pairs), len(reps)))

    # ── 프레임을 한 번만 뽑는다
    clips = {}
    for k, (name, v1, v2) in enumerate(pairs):
        for wv, v in (("v1", v1), ("v2", v2)):
            f = frames_of(v, args.nframes, tmp, "%03d%s" % (k, wv))
            if len(f) >= 3:
                clips[(name, wv)] = f
    print("클립 %d · 프레임 %d" % (len(clips), sum(len(v) for v in clips.values())))

    # ── 인코더들
    enc = {}
    if any(r.startswith("clip") for r in reps):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        cp = CLIPImageProcessor.from_pretrained(args.clip)
        cn = CLIPVisionModelWithProjection.from_pretrained(
            args.clip, use_safetensors=True).to(args.device).eval()
        enc["clip"] = (cp, cn, 14)
    if any(r.startswith("dino") for r in reps):
        from transformers import AutoImageProcessor, AutoModel
        dp = AutoImageProcessor.from_pretrained(args.dino)
        dn = AutoModel.from_pretrained(args.dino,
                                       use_safetensors=True).to(args.device).eval()
        enc["dino"] = (dp, dn, 16)
    if "ijepa" in reps:
        from transformers import AutoImageProcessor, AutoModel
        ip = AutoImageProcessor.from_pretrained("facebook/ijepa_vith14_1k")
        inet = AutoModel.from_pretrained("facebook/ijepa_vith14_1k",
                                         use_safetensors=True).to(args.device).eval()
        enc["ijepa"] = (ip, inet, 16)
    vj = None
    if "vjepa2" in reps:
        from transformers import AutoModel as _AM
        # ⚠️ V-JEPA2 의 튜블렛 임베딩은 **Conv3D** 이고 MPS 가 지원하지 않는다
        # ("Conv3D is not supported on MPS"). MPS 폴백도 이 경로는 못 잡으므로
        # **이 모델만 CPU** 로 돌린다. 나머지는 MPS 그대로다.
        # MPS 는 Conv3D 미지원이라 CPU 로 내린다. CUDA 면 그대로 GPU 에 올린다.
        vjdev = "cpu" if args.device == "mps" else args.device
        vj = _AM.from_pretrained(args.vjepa, use_safetensors=True).to(vjdev).eval()
        print("V-JEPA2 적재 (%.0fM)" % (sum(q.numel() for q in vj.parameters()) / 1e6))

    VMEAN = np.array([0.485, 0.456, 0.406], np.float32)
    VSTD = np.array([0.229, 0.224, 0.225], np.float32)

    def vprep(ims, short=292, crop=256):
        """V-JEPA2 전처리 — torchvision 없이 PIL+numpy 로."""
        out = []
        for im in ims:
            w, h = im.size
            sc = short / min(w, h)
            im2 = im.resize((max(crop, int(round(w * sc))), max(crop, int(round(h * sc)))),
                            Image.BILINEAR)
            w, h = im2.size
            l, t = (w - crop) // 2, (h - crop) // 2
            a = np.asarray(im2.crop((l, t, l + crop, t + crop)), np.float32) / 255.0
            out.append(((a - VMEAN) / VSTD).transpose(2, 0, 1))
        return torch.tensor(np.stack(out))[None].to(vjdev)

    def vjepa_vec(ims):
        """짧은 영상 → 한 벡터(토큰 평균). 프레임 수는 2의 배수여야 한다(튜블렛 2)."""
        if len(ims) % 2:
            ims = ims[:-1]
        if len(ims) < 2:
            return None
        with torch.no_grad():
            o = vj(pixel_values_videos=vprep(ims)).last_hidden_state
        return nz(o.mean(1).cpu().numpy())

    def vit_feats(fam, imgs):
        """CLS · 패치평균 · 2×2 구역 세 가지."""
        proc, net, G = enc[fam]
        inp = proc(images=imgs, return_tensors="pt").to(args.device)
        with torch.no_grad():
            if fam in ("dino", "ijepa"):
                o = net(**inp)
                h = o.last_hidden_state
                has_cls = fam == "dino"          # I-JEPA 는 CLS 가 없다
                cls = h[:, 0] if has_cls else h.mean(1)
                pat = h[:, 1:] if has_cls else h
                mean = pat.mean(1)
                proj = None
            elif fam == "clip":
                o = net.vision_model(**inp)
                h = net.vision_model.post_layernorm(o.last_hidden_state)
                cls = net.visual_projection(h[:, 0])
                pat = h[:, 1:]
                mean = net.visual_projection(pat.mean(1))
                proj = net.visual_projection
            n, npatch, dim = pat.shape
            g = int(round(npatch ** 0.5))
            gp = pat[:, :g * g].reshape(n, g, g, dim)
            half = g // 2
            quads = [gp[:, :half, :half], gp[:, :half, half:],
                     gp[:, half:, :half], gp[:, half:, half:]]
            qs = [q.reshape(n, -1, dim).mean(1) for q in quads]
            if proj is not None:
                qs = [proj(q) for q in qs]
            g2 = torch.cat(qs, 1)
        return nz(cls.cpu().numpy()), nz(mean.cpu().numpy()), nz(g2.cpu().numpy())

    owlnet = None
    if "owlvec" in reps or "owlvec_idf" in reps:
        import json as _json
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        OWV = _json.load(open(args.owl_vocab))
        op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        owlnet = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble").to(args.device).eval()
        # 어휘 1,037개를 한 번에 넣으면 메모리가 터진다 — 조각으로 나눠 훑는다
        CH = 128
        OWCH = [OWV[i:i + CH] for i in range(0, len(OWV), CH)]
        print("물체 조합 어휘 %d단어 · %d조각" % (len(OWV), len(OWCH)))

    def owl_vec(imgs):
        """프레임별 [어휘] 최대 검출점수 → 물체 조합 벡터."""
        V = np.zeros((len(imgs), sum(len(c) for c in OWCH)), np.float32)
        for i, im in enumerate(imgs):
            off = 0
            for ch in OWCH:
                with torch.no_grad():
                    inp = op(text=[ch], images=im, return_tensors="pt").to(args.device)
                    o = owlnet(**inp)
                    r = op.post_process_grounded_object_detection(
                        o, threshold=args.owl_thr,
                        target_sizes=torch.tensor([im.size[::-1]]).to(args.device))[0]
                for lb, sc in zip(r["labels"].tolist(), r["scores"].tolist()):
                    V[i, off + lb] = max(V[i, off + lb], float(sc))
                off += len(ch)
        return V

    def cheap(imgs):
        """tiny(32×32 회색조) · colorlay(3×3 × 색 히스토그램) — 비용 거의 0."""
        T, C = [], []
        for im in imgs:
            a = np.asarray(im.convert("L").resize((32, 32)), np.float32) / 255.0
            a = (a - a.mean()) / (a.std() + 1e-6)
            T.append(a.ravel())
            rgb = np.asarray(im.resize((96, 96)), np.float32) / 255.0
            v = []
            for i in range(3):
                for j in range(3):
                    blk = rgb[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
                    for ch in range(3):
                        v.append(np.histogram(blk[..., ch], bins=8, range=(0, 1))[0])
            C.append(np.concatenate(v).astype(np.float32))
        return nz(T), nz(C)

    # ── 모든 클립에 대해 표현 계산
    R = {r: {} for r in reps}
    DPATCH = {}
    for i, (key, files) in enumerate(clips.items()):
        ims = [Image.open(p).convert("RGB") for p in files]
        if "clip" in enc:
            a, b, c = vit_feats("clip", ims)
            for nm, v in (("clip_cls", a), ("clip_patch", b), ("clip_g2", c)):
                if nm in R:
                    R[nm][key] = v
        if "dino" in enc:
            a, b, c = vit_feats("dino", ims)
            for nm, v in (("dino_cls", a), ("dino_patch", b), ("dino_g2", c)):
                if nm in R:
                    R[nm][key] = v
        if "dino_vlad" in R:
            proc, net, _ = enc["dino"]
            with torch.no_grad():
                o = net(**proc(images=ims, return_tensors="pt").to(args.device))
            DPATCH[key] = o.last_hidden_state[:, 1:].cpu().numpy().astype(np.float32)
        if "ijepa" in enc:
            a, b, c = vit_feats("ijepa", ims)
            R["ijepa"][key] = b                      # CLS 가 없으니 패치평균
        if vj is not None:
            # 영상 모델은 클립 단위다 — ①(시점) 을 재려면 앞/뒤 절반을 각각 인코딩
            h = len(ims) // 2
            k_ = args.vj_frames
            vs = [vjepa_vec(ims[:h][:k_]), vjepa_vec(ims[h:][:k_])]
            vs = [v for v in vs if v is not None]
            if vs:
                R["vjepa2"][key] = np.concatenate(vs)
        if owlnet is not None:
            V = owl_vec(ims)
            if "owlvec" in R:
                R["owlvec"][key] = nz(V)
            if "owlvec_idf" in R:
                R["owlvec_idf"][key] = V          # IDF 는 전체를 모은 뒤 적용
        if "tiny" in R or "colorlay" in R:
            t, cl = cheap(ims)
            if "tiny" in R:
                R["tiny"][key] = t
            if "colorlay" in R:
                R["colorlay"][key] = cl
        if (i + 1) % 20 == 0:
            print("  %d/%d 클립" % (i + 1, len(clips)))

    if "dino_vlad" in R and DPATCH:
        # VLAD — 패치를 시각 어휘에 배정하고 **어휘별 잔차 합**을 기술자로.
        # 전역 평균과 달리 "무엇이 있었나" 를 어휘별로 나눠 담아 시점 이동에 덜 흔들린다.
        from sklearn.cluster import KMeans
        allp = np.concatenate([v.reshape(-1, v.shape[-1]) for v in DPATCH.values()])
        sub = allp[np.random.default_rng(0).permutation(len(allp))[:60000]]
        km = KMeans(args.vlad_k, n_init=4, random_state=0).fit(sub)
        C = km.cluster_centers_.astype(np.float32)
        print("VLAD 어휘 %d개 학습 (패치 %d개 중 %d 표본)" % (args.vlad_k, len(allp), len(sub)))
        for key, P_ in DPATCH.items():
            out = []
            for f in P_:                                   # 프레임마다
                a = np.argmin(((f[:, None] - C[None]) ** 2).sum(-1), 1)
                V = np.zeros_like(C)
                for j in range(len(C)):
                    m = a == j
                    if m.any():
                        V[j] = (f[m] - C[j]).sum(0)
                V = np.sign(V) * np.sqrt(np.abs(V))        # power normalization
                V = V.ravel()
                out.append(V / (np.linalg.norm(V) + 1e-9))
            R["dino_vlad"][key] = np.stack(out)

    if "owlvec_idf" in R and R["owlvec_idf"]:
        # IDF — 어디에나 뜨는 물체(wall·floor)의 비중을 낮춘다
        ALLV = np.concatenate(list(R["owlvec_idf"].values()))
        df = (ALLV > args.owl_thr).mean(0) + 1e-6
        idf = np.log(1.0 / df)
        R["owlvec_idf"] = {k: nz(v * idf[None]) for k, v in R["owlvec_idf"].items()}

    names = sorted({n for n, _ in clips})
    print("\n%-11s %-8s %-8s %-8s %-9s %-9s %s"
          % ("표현", "①시점", "②같은장소", "③남의장소", "여백②−③", "시점①−③", "1등비율"))
    out = []
    for r in reps:
        E = R[r]
        rows = []
        for n in names:
            if (n, "v1") not in E or (n, "v2") not in E:
                continue
            A, B = E[(n, "v1")], E[(n, "v2")]
            S = A @ A.T
            intra = float(np.median(S[np.triu_indices(len(A), 1)])) if len(A) > 1 else np.nan
            a = A.mean(0); a /= np.linalg.norm(a) + 1e-9
            same = float(np.median(B @ a))
            oth, top1 = -9.0, True
            for m in names:
                if m == n:
                    continue
                for wv in ("v1", "v2"):
                    if (m, wv) in E:
                        s = float(np.median(E[(m, wv)] @ a))
                        oth = max(oth, s)
            rows.append((intra, same, oth))
        if len(rows) < 5:
            continue
        I = np.array([x[0] for x in rows]); Sa = np.array([x[1] for x in rows])
        O = np.array([x[2] for x in rows])
        mg = Sa - O
        print("%-11s %-8.3f %-8.3f %-8.3f %-9.3f %-9.3f %.0f%%"
              % (r, np.median(I), np.median(Sa), np.median(O),
                 np.median(mg), np.median(I - O), 100 * (mg > 0).mean()))
        out.append((r, float(np.median(mg)), float((mg > 0).mean())))
    out.sort(key=lambda x: -x[1])
    print("\n여백 순위: %s" % " > ".join("%s(%+.3f·%.0f%%)" % (a, b, 100 * c) for a, b, c in out))
    print("→ 여백이 크고 1등비율이 높은 표현이 장소를 잘 가른다. 시점①−③ 이 음수면")
    print("  각도만 바뀌어도 남의 장소보다 멀어진다는 뜻이다.")


if __name__ == "__main__":
    main()
