#!/usr/bin/env python3
"""**아이디어1 상한 — 텍스트 대신 "그 물건의 모습"으로 찾는다.**

    $P scripts/it3d_imgquery.py --slices … --ann … --cache …

지금은 `"blue coffee mug"` 라는 **말**로 찾는다. 그런데 우리는 **그 컵을 본 적이 있다.**
씬그래프 노드에 **외형 임베딩**을 attribute 로 달아두면(512차원 · 2KB) 검색도 부재도
"그 범주" 가 아니라 **"그 물건"** 으로 물을 수 있다.

    노드: blue coffee mug
      ├ label       "blue coffee mug"        ← 지금 있는 것
      ├ appearance  512-d 임베딩 (목격 프레임 크롭)   ← 추가
      └ seen_at     (방, 시각, 검출도)

⚠️ 이건 **상한 측정**이다. GT bbox 로 잘라 크롭 품질을 통제한다. 여기서 텍스트를
못 이기면 우리 검출 상자로 자를 이유도 없다.

지표는 ㊱ 과 같다 — 양성 = GT bbox 가 ±0.2초 안에 있는 프레임, 음성 = 그 밖.
비교 대상: OWL 텍스트(㊱ 에서 0.812) · CLIP 텍스트(0.690).

⚠️ **CLIP 전체 프레임 임베딩과 크롭을 코사인으로 비교하면 안 된다.** 첫 시도가
그랬는데 AUC 0.523(우연)이 나왔다 — 텍스트 0.693보다 나쁘다. CLIP 공간에서
**클로즈업 크롭**과 **넓은 방 전경**은 애초에 멀어서, 유사도가 "이 물체가 들어 있나"
가 아니라 **"이것도 클로즈업인가"** 를 잰다. 텍스트는 CLIP 이 전체 이미지-캡션 쌍으로
학습돼 전경과 잘 붙으므로 애초에 유리하다.

외형을 쓰려면 **OWLv2 의 image-guided 경로**로 **패치 단위**에서 비교해야 한다:
`embed_image_query` 로 크롭을 검출 클래스 공간의 질의 임베딩으로 바꾸고,
대상 프레임의 패치들과 `class_predictor` 로 맞춘다(텍스트 경로와 같은 자리).

⚠️ 크롭은 **다른 시각**에서 뽑는다. 채점 프레임에서 자른 크롭으로 그 프레임을
채점하면 당연히 맞는다(누수). 물체별로 **가장 이른** bbox 프레임에서만 자르고,
채점은 그 프레임을 뺀 나머지로 한다.
"""
import argparse, glob, io, json, os, re, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.it3d_absence import base_label, load_ann              # noqa: E402

PVRE = re.compile(r"raw_videos/(video_\d+_scene_\d+)/pv/(\d+)\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True)
    ap.add_argument("--cache", required=True, help="*.all.npz (프레임 임베딩·시각)")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--tol", type=float, default=2e6)
    ap.add_argument("--n-crop", type=int, default=3, help="물체당 크롭 수(이른 것부터)")
    ap.add_argument("--pad", type=float, default=0.15, help="크롭 여백 비율")
    ap.add_argument("--min-pos", type=int, default=5)
    ap.add_argument("--videos", type=int, default=0, help="0=전부")
    ap.add_argument("--owl-batch", type=int, default=2)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from scipy.stats import mannwhitneyu
    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from scripts.absence_evidence import clip_text
    om = "google/owlv2-base-patch16-ensemble"
    op = Owlv2Processor.from_pretrained(om)
    onet = Owlv2ForObjectDetection.from_pretrained(om).to(args.device).eval()

    def owl_scores(imgs, qemb):
        """대상 프레임들 × 질의 임베딩 1개 → 프레임별 최대 검출점수."""
        out = []
        for i in range(0, len(imgs), args.owl_batch):
            pv = op(images=imgs[i:i + args.owl_batch],
                    return_tensors="pt")["pixel_values"].to(args.device)
            with torch.no_grad():
                fm = onet.image_embedder(pixel_values=pv)[0]
                b, ph, pw, hd = fm.shape
                lg, _ = onet.class_predictor(fm.reshape(b, ph * pw, hd),
                                             qemb.unsqueeze(0).expand(b, -1, -1), None)
                out.append(torch.sigmoid(lg).amax(1).squeeze(-1).float().cpu().numpy())
        return np.concatenate(out)

    def query_embed(crops):
        """크롭 → 검출 클래스 공간의 질의 임베딩 (OWL 자체 경로)."""
        pv = op(images=crops, return_tensors="pt")["pixel_values"].to(args.device)
        with torch.no_grad():
            fm = onet.image_embedder(pixel_values=pv)[0]
            b, ph, pw, hd = fm.shape
            feats = fm.reshape(b, ph * pw, hd)
            q = onet.embed_image_query(feats, fm)
        q = q[0] if isinstance(q, (tuple, list)) else q
        q = q.reshape(-1, q.shape[-1])
        q = q[~torch.isnan(q).any(1)]
        return None if len(q) == 0 else torch.nn.functional.normalize(q.mean(0, keepdim=True), dim=-1)

    files = sorted(glob.glob(os.path.join(args.cache, "*.all.npz")))
    if args.videos:
        files = files[:args.videos]
    rows = []
    for f in files:
        vn = os.path.basename(f)[:-len(".all.npz")]
        ad = os.path.join(args.ann, vn)
        binp = os.path.join(args.slices, vn + ".bin")
        idxp = os.path.join(args.slices, vn + ".index.json")
        if not (os.path.isdir(ad) and os.path.exists(binp)):
            continue
        z = np.load(f, allow_pickle=True)
        ts, E = z["ts"], z["emb"]
        labs, segs, box = load_ann(ad)
        words = [base_label(l) for l in labs]
        Q = clip_text(words, args.device)
        # 시각 → (오프셋, 크기)
        loc = {}
        for r in json.load(open(idxp)):
            m = PVRE.match(r["name"])
            if m and m.group(1) == vn:
                loc[int(m.group(2))] = (r["off"], r["size"])
        # bbox: 시각 → (x,y,w,h)
        bb = defaultdict(dict)
        bd = os.path.join(ad, "2d_bbox_annot")
        for fn in os.listdir(bd) if os.path.isdir(bd) else []:
            if fn.endswith(".txt"):
                oi = int(fn[:-4])
                for line in open(os.path.join(bd, fn)):
                    p = line.split()
                    if len(p) >= 5:
                        bb[oi][int(p[0])] = tuple(int(v) for v in p[1:5])

        with open(binp, "rb") as tf:
            for oi, w in enumerate(words):
                bts = np.array(sorted(bb.get(oi, {})), dtype=np.int64)
                if len(bts) == 0:
                    continue
                vis = np.array([bool(np.any(np.abs(bts - t) <= args.tol)) for t in ts])
                if vis.sum() < args.min_pos or (~vis).sum() < args.min_pos:
                    continue
                # ⚠️ 크롭은 **이른 것부터**, 채점에서는 그 프레임을 뺀다(누수 방지)
                crops, used = [], set()
                for bt in bts:
                    if len(crops) >= args.n_crop:
                        break
                    near = [t for t in loc if abs(t - bt) <= args.tol]
                    if not near:
                        continue
                    t0 = min(near, key=lambda t: abs(t - bt))
                    off, sz = loc[t0]
                    tf.seek(off)
                    try:
                        im = Image.open(io.BytesIO(tf.read(sz))).convert("RGB")
                    except Exception:
                        continue
                    x, y, bw, bh = bb[oi][int(bt)]
                    px, py = args.pad * bw, args.pad * bh
                    W, H = im.size
                    c = im.crop((max(0, x - px), max(0, y - py),
                                 min(W, x + bw + px), min(H, y + bh + py)))
                    if c.size[0] < 8 or c.size[1] < 8:
                        continue
                    crops.append(c); used.add(t0)
                if not crops:
                    continue
                qe = query_embed(crops)
                if qe is None:
                    continue
                keep = np.array([t not in used for t in ts])
                v, nv = vis & keep, (~vis) & keep
                if v.sum() < 3 or nv.sum() < 3:
                    continue
                # 대상 프레임을 다시 읽어 OWL 로 채점 (질의 = 외형)
                sel = np.nonzero(keep)[0]
                ims = []
                for k in sel:
                    o2, s2 = loc.get(int(ts[k]), (None, None))
                    if o2 is None:
                        ims.append(None); continue
                    tf.seek(o2)
                    try:
                        ims.append(Image.open(io.BytesIO(tf.read(s2))).convert("RGB"))
                    except Exception:
                        ims.append(None)
                ok = [i for i, x in enumerate(ims) if x is not None]
                if len(ok) < 6:
                    continue
                sc_img = owl_scores([ims[i] for i in ok], qe)
                sel = sel[ok]
                v = vis[sel]; nv = ~vis[sel]
                if v.sum() < 3 or nv.sum() < 3:
                    continue
                s_img = sc_img
                s_txt = (E @ Q[oi])[sel]
                a_img = mannwhitneyu(s_img[v], s_img[nv], alternative="greater")[0] / (v.sum() * nv.sum())
                a_txt = mannwhitneyu(s_txt[v], s_txt[nv], alternative="greater")[0] / (v.sum() * nv.sum())
                rows.append(dict(video=vn, obj=labs[oi], word=w, n_vis=int(v.sum()),
                                 n_not=int(nv.sum()), n_crop=len(crops),
                                 auc_img=float(a_img), auc_txt=float(a_txt)))
        print("  %-20s 누적 %d" % (vn, len(rows)), flush=True)

    if not rows:
        print("표본 없음"); return
    I = np.array([r["auc_img"] for r in rows]); Tx = np.array([r["auc_txt"] for r in rows])
    print("\n물체 %d · 영상 %d" % (len(rows), len({r["video"] for r in rows})))
    print("  **이미지 질의(외형) AUC 중앙 %.3f**  [%.3f %.3f]"
          % (np.median(I), *np.quantile(I, [.25, .75])))
    print("  CLIP 텍스트 질의(같은 프레임) AUC 중앙 %.3f  [%.3f %.3f]"
          % (np.median(Tx), *np.quantile(Tx, [.25, .75])))
    print("  이미지가 더 나은 물체 %d/%d (%.0f%%)"
          % (int((I > Tx).sum()), len(I), 100 * (I > Tx).mean()))
    from scipy.stats import wilcoxon
    print("  짝지은 검정 p=%.3g" % wilcoxon(I, Tx, alternative="greater")[1])
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
