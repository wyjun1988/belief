#!/usr/bin/env python3
"""IT3DEgo 로 **장소 수준 부재**를 잰다 — "원래 자리에 없다".

    $P scripts/it3d_absence.py --tar <…tar> --index <…jsonl> --ann <…/annotations>

### 왜 이 데이터인가

지금까지 부재를 세 가지 층위에서 쟀는데 각각 한계가 있었다:

| 층위 | 데이터 | 결과 | 한계 |
|---|---|---|---|
| 씬 수준("영상에서 사라짐") | SceneDiff | 0.714 (p=2.6e-21) | 우리 시나리오보다 쉽다 |
| 방 수준("그 방에 없다") | Nymeria·SuperMemory | 0.615·0.636 (유의 안 함) | **표본 n=17~19** |
| **장소 수준("그 자리에 없다")** | ADT | 막힘 | 라벨이 이동≠부재, 개체 ID 없음 |

IT3DEgo 가 마지막 칸을 채운다. `3d_center_annot.txt` 가
`시작 종료 개체id x y z location_id` 라 **개체 신원과 이동 시각이 GT** 다.
49영상 · 284개체 · **이동 사건 935건** — 표본 문제가 50배로 풀린다.

⚠️ **`location_id` 는 방이 아니다.** 위치 간 거리가 0.6~1.9 m로 **같은 방 안의 자리**다.
그래서 이 결과는 방 단위 결론(belief 층)과 **따로** 적어야 한다. 대신 씬 수준보다
어렵고 우리 시나리오("탁자 위에 없다")에 더 가깝다.

### 설계 — 사건마다 양성·음성을 **짝지어** 만든다

물체 O 가 위치 L 에 [t0,t1] 있다가 떠난다. 같은 O·같은 L·같은 검출기로:

    앵커  A = L 구간 중 O 의 2D bbox 가 있는 프레임 (카메라가 확실히 L 을 봄)
    게이트 G = A 를 뺀 프레임 중 **앵커 자기유사도 문턱**을 넘는 것 (= L 을 보는 프레임)
      before  = G ∩ [t0, 중간]        O 가 L 에 있음
      control = G ∩ (중간, t1]        O 가 아직 L 에 있음  ← **음성**
      test    = G ∩ (t1, 끝]          O 가 L 을 떠남      ← **양성**

앵커를 세 창에서 모두 빼 **대칭**을 맞춘다(안 그러면 before·control 만 "물체가
보이는 프레임" 을 포함해 유리해진다). 판정은 짝지은 부호검정이라 물체별 어휘
편향·시점·조도가 상쇄된다 — ㉜ 에서 물린 "물체 간 절대값 비교" 함정을 원천 차단한다.

⚠️ 장소 게이트에 **퍼센타일을 쓰지 않는다.** 종전에 상위 20% 로 자르면 L 을 전혀
안 보는 구간에서도 뭔가 뽑혀 장소 적중률이 정확히 0.00 이 됐다(㉗). 앵커 자기유사도
분위수를 절대 문턱으로 쓴다.

⚠️ 조건① — 같은 영상에 같은 기본 이름의 다른 개체가 있으면 제외한다(`pen_1`·`pen_3`).
검출기가 둘을 못 가르므로 "떠났다" 를 물을 수 없다.

### 왜 2단계인가 — OWL 이 비싸다

실측(장당): 디코드 0.014 s · CLIP 0.010 s · **OWL 0.60 s**(M1 Pro) / 1.25 s(iMac).
전 프레임에 OWL 을 돌리면 961장 × 49영상 = 7.8시간이다. 그런데 검출 점수가 필요한
곳은 **창(before·control·test)에 들어간 프레임뿐**이다. 그래서:

    1단계  CLIP 을 전 프레임에 — 장소 게이트용 (싸다)
    2단계  창이 정해진 뒤, **거기 쓰인 프레임에만** OWL

창마다 `--cap` 장으로 균등 간격 표본을 뽑아 더 줄인다. 시간 분포를 유지하려고
무작위가 아니라 **균등 간격**으로 뽑는다.
"""
import argparse, io, json, os, re, sys
from collections import defaultdict

import numpy as np


def base_label(s):
    """`coffee_cup_1` → `coffee cup`. 끝의 개체 번호만 뗀다."""
    return re.sub(r"_\d+$", "", s.strip()).replace("_", " ").strip()


def load_ann(d):
    """한 영상의 어노테이션 → (라벨목록, 개체별 위치구간, 개체별 bbox 시각)"""
    labs = [l.strip() for l in open(os.path.join(d, "labels.csv")) if l.strip()]
    segs = defaultdict(list)
    for line in open(os.path.join(d, "3d_center_annot.txt")):
        p = line.split()
        if len(p) >= 7:
            segs[int(p[2])].append((int(p[0]), int(p[1]), int(p[6])))
    for v in segs.values():
        v.sort()
    box = {}
    bd = os.path.join(d, "2d_bbox_annot")
    if os.path.isdir(bd):
        for f in os.listdir(bd):
            if f.endswith(".txt"):
                ts = []
                for line in open(os.path.join(bd, f)):
                    p = line.split()
                    if p:
                        ts.append(int(p[0]))
                box[int(f[:-4])] = np.array(sorted(ts), dtype=np.int64)
    return labs, segs, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", default=None, help="전체 tar (색인 모드)")
    ap.add_argument("--index", default=None, help="it3d_tarindex.py 산출 jsonl")
    ap.add_argument("--slices", default=None,
                    help="it3d_slice.py 산출 디렉터리 — 영상별 <이름>.bin + .index.json")
    ap.add_argument("--ann", required=True, help="annotations 디렉터리")
    ap.add_argument("--cache", default="data/it3dego/cache")
    ap.add_argument("--stride", type=int, default=0,
                    help="pv 프레임 부분추출 간격. 0이면 자동 — "
                         "⚠️ **조각 모드에서는 1이어야 한다.** 조각 자체가 이미 "
                         "무작위 부분추출본(1GB≈960프레임)이라 여기서 또 15로 자르면 "
                         "영상당 65프레임만 남아 창마다 표본이 말라버린다. "
                         "전체 tar 모드에서만 15가 맞다.")
    ap.add_argument("--anchor-q", type=float, default=0.70,
                    help="앵커 자기유사도 분위수 → 장소 문턱 (㉗ 에서 0.70 최적)")
    ap.add_argument("--topk", type=int, default=3, help="프레임 투표에 쓸 앵커 수")
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--cap", type=int, default=12,
                    help="창당 최대 프레임 — OWL 비용을 줄인다. 균등 간격으로 뽑는다.")
    ap.add_argument("--max-gap", type=float, default=0.10,
                    help="허용 최대 시간 공백(영상 길이 대비) — 균일 표본 검사")
    ap.add_argument("--cond2", type=float, default=0.0, help="before 검출도 하한(조건②)")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16, help="CLIP 배치")
    ap.add_argument("--owl-batch", type=int, default=2,
                    help="OWL 배치 — ⚠️ 960×960 어텐션이 커서 8GB MPS 에서 16은 터진다"
                         "(buffer 9.27GB). 2가 안전선.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.stride:
        args.stride = 1 if args.slices else 15
    os.makedirs(args.cache, exist_ok=True)

    # ── ① 프레임 목록 — 영상별 (시각, 오프셋, 크기) 와 그것을 담은 파일
    vid = defaultdict(list)
    src = {}                                    # 영상 → 바이트를 읽을 파일
    PVRE = re.compile(r"raw_videos/(video_\d+_scene_\d+)/pv/(\d+)\.png$")
    if args.slices:
        for f in sorted(os.listdir(args.slices)):
            # ⚠️ exFAT 에서 rsync 로 가져오면 `._<이름>` AppleDouble 이 딸려온다.
            # 셸 glob 은 점파일을 안 잡아 검증에서 놓쳤는데 `os.listdir` 은 집는다 →
            # 바이너리를 json 으로 읽어 UnicodeDecodeError 로 죽었다.
            if f.startswith(".") or not f.endswith(".index.json"):
                continue
            vn = f[:-len(".index.json")]
            binp = os.path.join(args.slices, vn + ".bin")
            if not os.path.exists(binp):
                continue
            for r in json.load(open(os.path.join(args.slices, f))):
                m = PVRE.match(r["name"])
                if m and m.group(1) == vn:
                    vid[vn].append((int(m.group(2)), r["off"], r["size"]))
            src[vn] = binp
    else:
        with open(args.index) as f:
            for line in f:
                r = json.loads(line)
                m = PVRE.match(r["name"])
                if m and r["type"] == "0":
                    vid[m.group(1)].append((int(m.group(2)), r["off"], r["size"]))
                    src[m.group(1)] = args.tar
    for v in vid.values():
        v.sort()
    names = sorted(vid, key=lambda s: (int(s.split("_")[3]), int(s.split("_")[1])))
    if args.videos:
        names = [n for n in names if n in args.videos]
    print("pv 프레임을 가진 영상 %d개 (총 프레임 %d)"
          % (len(names), sum(len(vid[n]) for n in names)))
    if not names:
        print("아직 raw_videos/*/pv 가 색인에 없다 — 다운로드가 더 진행돼야 한다.")
        return

    import torch
    from PIL import Image
    from transformers import (Owlv2Processor, Owlv2ForObjectDetection,
                              CLIPImageProcessor, CLIPVisionModelWithProjection)
    om = "google/owlv2-base-patch16-ensemble"
    op = Owlv2Processor.from_pretrained(om)
    onet = Owlv2ForObjectDetection.from_pretrained(om).to(args.device).eval()
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cnet = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()

    def owl_text(words):
        """⚠️ 어휘를 바꿔가며 `onet(**inp)` 를 부르면 **이미지 인코더가 매번** 다시 돈다.
        텍스트 임베딩만 캐시하고 프레임당 image_embedder 를 1회 돌린다
        (`place_repr_bench.py` 에서 원 경로 대비 최대오차 0.00e+00 확인)."""
        q = ["a photo of a " + w for w in words]
        dm = Image.new("RGB", (256, 256), (128, 128, 128))
        ti = op(text=[q], images=[dm], return_tensors="pt").to(args.device)
        with torch.no_grad():
            o = onet.owlv2(input_ids=ti["input_ids"], attention_mask=ti["attention_mask"],
                           pixel_values=ti["pixel_values"], return_dict=True)
        return o.text_embeds, (ti["input_ids"][:, 0] > 0)

    rows = []
    for vn in names:
        cf = os.path.join(args.cache, vn + ".npz")
        ad = os.path.join(args.ann, vn)
        if not os.path.isdir(ad):
            print("  %-20s 어노테이션 없음 — 건너뜀" % vn)
            continue
        labs, segs, box = load_ann(ad)
        words = [base_label(l) for l in labs]
        # ⚠️ **부분 다운로드된 영상을 쓰면 안 된다.** tar 를 받는 도중에는 그 영상의
        # 뒷부분 프레임이 아직 없다. 그대로 쓰면 물체가 L 을 떠난 뒤 구간이 통째로
        # 잘려 "재방문이 없었다" 로 오판하거나, 남은 조각만으로 test 창을 만든다.
        # 어노테이션이 덮는 시간 범위를 프레임이 **다 덮을 때만** 채점한다.
        at = [t for v in segs.values() for t0, t1, _ in v for t in (t0, t1)]
        ft = sorted(t for t, _, _ in vid[vn])
        if not at or len(ft) < 20:
            print("  %-20s 프레임 부족 — 보류" % vn)
            continue
        # ⚠️ 조각 모드에서는 프레임이 **무작위 표본**이라 양끝이 조금 모자랄 수 있다.
        # 대신 **큰 시간 공백**을 본다 — 공백이 크면 그 영상은 tar 안 순서가 무작위가
        # 아니라는 뜻이고, 그러면 test 창이 통째로 비어 "재방문 없음" 으로 오판한다.
        gaps = [ft[i + 1] - ft[i] for i in range(len(ft) - 1)]
        span = max(at) - min(at)
        if max(gaps) > args.max_gap * span:
            print("  %-20s 시간 공백 %.0f%% — 균일 표본이 아니다, 보류"
                  % (vn, max(gaps) * 100.0 / span))
            continue
        cov = (min(ft) - min(at)) <= 0.1 * span and (max(at) - max(ft)) <= 0.1 * span
        if not cov:
            print("  %-20s 앞뒤 커버리지 부족 — 보류" % vn)
            continue
        frames = vid[vn][::args.stride]
        if len(frames) < 20:
            print("  %-20s 프레임 %d — 부족" % (vn, len(frames)))
            continue

        cclip = os.path.join(args.cache, vn + ".clip.npz")
        cowl = os.path.join(args.cache, vn + ".owl.npz")
        if os.path.exists(cclip):
            z = np.load(cclip)
            ts, E = z["ts"], z["emb"]
        else:
            ts, E = [], []
            with open(src[vn], "rb") as tf:
                for i in range(0, len(frames), args.batch):
                    ims = []
                    for t, off, sz in frames[i:i + args.batch]:
                        tf.seek(off)
                        try:
                            ims.append(Image.open(io.BytesIO(tf.read(sz))).convert("RGB"))
                            ts.append(t)
                        except Exception:
                            pass
                    if not ims:
                        continue
                    with torch.no_grad():
                        e = cnet(**cp(images=ims, return_tensors="pt").to(
                            args.device)).image_embeds.cpu().numpy().astype(np.float32)
                    E.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
            ts = np.array(ts, np.int64); E = np.concatenate(E)
            np.savez_compressed(cclip, ts=ts, emb=E)

        # ── 조건① 같은 기본 이름의 다른 개체가 있으면 제외
        dup = {w for w in words if words.count(w) > 1}

        def pick(idx):
            """창에서 --cap 장을 **균등 간격**으로 — 시간 분포를 유지한다."""
            if len(idx) <= args.cap:
                return idx
            return idx[np.linspace(0, len(idx) - 1, args.cap).astype(int)]

        events, need = [], set()
        for oi, segl in segs.items():
            if oi >= len(words) or words[oi] in dup:
                continue
            bts = box.get(oi, np.array([], np.int64))
            if len(bts) == 0:
                continue
            for si in range(len(segl) - 1):
                t0, t1, L = segl[si]
                if segl[si + 1][2] == L:            # 위치가 안 바뀌면 이동이 아니다
                    continue
                # 앵커 — bbox 시각과 부분추출 프레임이 정확히 안 맞으므로 최근접
                cand = bts[(bts >= t0) & (bts <= t1)]
                if len(cand) == 0:
                    continue
                A = np.unique([int(np.argmin(np.abs(ts - b))) for b in cand])
                A = A[(ts[A] >= t0) & (ts[A] <= t1)]
                if len(A) < 3:
                    continue
                SS = E[A] @ E[A].T
                iu = np.triu_indices(len(A), 1)
                thr = float(np.quantile(SS[iu], args.anchor_q))
                rest = np.setdiff1d(np.arange(len(ts)), A)
                sim = np.sort(E[rest] @ E[A].T, 1)[:, -min(args.topk, len(A)):].mean(1)
                G = rest[sim >= thr]
                mid = (t0 + t1) // 2
                bef = pick(G[(ts[G] >= t0) & (ts[G] <= mid)])
                ctl = pick(G[(ts[G] > mid) & (ts[G] <= t1)])
                tst = pick(G[ts[G] > t1])
                if min(len(bef), len(ctl), len(tst)) < args.min_frames:
                    continue
                events.append((oi, L, bef, ctl, tst))
                need.update(bef.tolist() + ctl.tolist() + tst.tolist())

        if not events:
            print("  %-20s 프레임 %d · 물체 %d · 사건 0" % (vn, len(ts), len(labs)), flush=True)
            continue
        need = np.array(sorted(need))
        print("  %-20s 프레임 %d · 물체 %d · 사건 %d · OWL 필요 %d장 (%.0f%%)"
              % (vn, len(ts), len(labs), len(events), len(need),
                 len(need) * 100.0 / len(ts)), flush=True)

        # ── 2단계 OWL — 창에 쓰인 프레임에만
        if os.path.exists(cowl):
            z = np.load(cowl)
            oidx, S = z["idx"], z["owl"]
        else:
            TX, MK = owl_text(words)
            fmap = {t: (o, sz) for t, o, sz in frames}
            S = []
            with open(src[vn], "rb") as tf:
                for i in range(0, len(need), args.owl_batch):
                    ims = []
                    for k in need[i:i + args.owl_batch]:
                        o, sz = fmap[int(ts[k])]
                        tf.seek(o)
                        ims.append(Image.open(io.BytesIO(tf.read(sz))).convert("RGB"))
                    pvx = op(images=ims, return_tensors="pt")["pixel_values"].to(args.device)
                    with torch.no_grad():
                        fm = onet.image_embedder(pixel_values=pvx)[0]
                        b, ph, pw, hd = fm.shape
                        lg, _ = onet.class_predictor(
                            fm.reshape(b, ph * pw, hd),
                            TX.unsqueeze(0).expand(b, -1, -1),
                            MK.unsqueeze(0).expand(b, -1))
                        S.append(torch.sigmoid(lg).amax(1).float().cpu().numpy())
            S = np.concatenate(S); oidx = need
            np.savez_compressed(cowl, idx=oidx, owl=S)
        pos = {int(k): i for i, k in enumerate(oidx)}

        for oi, L, bef, ctl, tst in events:
            def med(ix):
                return float(np.median([S[pos[int(k)], oi] for k in ix if int(k) in pos]))
            sb, sc, st = med(bef), med(ctl), med(tst)
            rows.append(dict(video=vn, obj=labs[oi], word=words[oi], loc=L,
                             n_bef=len(bef), n_ctl=len(ctl), n_tst=len(tst),
                             s_before=sb, s_control=sc, s_test=st,
                             drop_ctl=sb - sc, drop_tst=sb - st))

    if not rows:
        print("채점 가능한 사건이 없다.")
        return
    from scipy.stats import wilcoxon, mannwhitneyu
    sel = [r for r in rows if r["s_before"] >= args.cond2]
    dc = np.array([r["drop_ctl"] for r in sel])
    dt = np.array([r["drop_tst"] for r in sel])
    print("\n사건 %d (조건②≥%.2f 적용 후 %d) · 영상 %d"
          % (len(rows), args.cond2, len(sel), len({r["video"] for r in sel})))
    print("  하락 중앙   대조(있음) %+.4f · 검정(떠남) %+.4f" % (np.median(dc), np.median(dt)))
    w, pw = wilcoxon(dt, dc, alternative="greater")
    win = int((dt > dc).sum()); tie = int((dt == dc).sum())
    print("  **짝지은 부호검정** 떠남>있음 %d / %d (동률 %d) · Wilcoxon p=%.3g"
          % (win, len(sel), tie, pw))
    u, pu = mannwhitneyu(dt, dc, alternative="greater")
    print("  (참고) 비짝 AUC %.3f · p=%.3g" % (u / (len(dt) * len(dc)), pu))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
