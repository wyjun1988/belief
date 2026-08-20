#!/usr/bin/env python3
"""S-EMBER 로 **증거 검색**과 **최신성**을 잰다 — 증거 구간이 GT 로 붙은 데이터.

    $P scripts/sember_retrieval.py --root /Volumes/External_SSD/khronos/sember

`facebook/S-EMBER`(2026) · 3,141영상 · 388시간 · Ray-Ban Meta · 613명 · 질의 9,448개.
질의마다 **증거 구간**(`answer_start_time`~`answer_end_time`)이 명시돼 있어
검색 정답이 GT 로 주어진다. `memory_recency`(질의−근거 시간차)도 붙어 있다.

  **Ⓐ 증거 검색** — 질문으로 검색했을 때 증거 구간이 상위에 오는가 (hit@k)
  **Ⓑ 최신성** — 최근성 가중 exp(−Δt/τ) 를 켰을 때 어떻게 변하는가.
      우리 운용 기본값은 **τ=12h**(SuperMemory 136문항 스윕). EgoLife 에서도
      2.8시간 창에서 τ=12h 가 최적이었다 — 여기서 세 번째로 확인한다.
  **Ⓒ 유형별** — 우리 층(프레임 검색)이 답할 수 있는 유형이 무엇인지.
      `location_trace`·`visual_detail_recall` 은 시각으로 되고, `time_duration`·
      `counting_objects_events` 는 세거나 재는 것이라 검색만으로는 안 된다.

⚠️ **인과** — 질의 시각(`question_time`) 이후 프레임은 볼 수 없다.
⚠️ 영상이 중앙 **369초**라 `memory_recency` 가 중앙 74초 · 최대 19분이다.
   **시간축이 짧아 하루 스케일 검증은 안 된다** — 표본 크기가 강점인 데이터다.
"""
import argparse, json, os, subprocess, sys, tempfile
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Volumes/External_SSD/khronos/sember")
    ap.add_argument("--videos", default=None, help="영상 디렉터리(없으면 받는다)")
    ap.add_argument("--n-video", type=int, default=40, help="쓸 영상 수")
    ap.add_argument("--every", type=float, default=2.0, help="프레임 표집 간격(초)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--object-loc", action="store_true",
                    help="**물건 위치 회상 질문만** — `where did I put ~`(189건) ·"
                         " `where is my ~`(9건) · `when I last saw`(16건). 우리 과제와"
                         " 가장 가까운 부분집합이다. 유형 라벨로는 안 갈리므로 문구로 뽑는다")
    ap.add_argument("--cats", default=None,
                    help="쉼표 구분 질문 유형. 기본은 전부. 우리 층에 맞는 것은"
                         " location_trace,visual_detail_recall")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(args.root, "sember_grounding.jsonl"))]
    if args.cats:
        keep = set(args.cats.split(","))
        rows = [r for r in rows if r["question_category"] in keep]
    if args.object_loc:
        PAT = ("where did i put", "where is my", "last saw", "where did i leave",
               "where are my", "where did i place")
        rows = [r for r in rows if any(k in r["question"].lower() for k in PAT)]
        print("물건 위치 회상 질문만: %d건" % len(rows))
    by = defaultdict(list)
    for r in rows:
        if r.get("answer_start_time") is None:
            continue
        by[r["video_id"]].append(r)
    vids = sorted(by, key=lambda v: -len(by[v]))[:args.n_video]
    print("영상 %d · 질의 %d" % (len(vids), sum(len(by[v]) for v in vids)))

    vdir = args.videos or os.path.join(args.root, "videos")
    os.makedirs(vdir, exist_ok=True)
    from huggingface_hub import hf_hub_download

    import torch, cv2
    from PIL import Image
    from transformers import (CLIPImageProcessor, CLIPVisionModelWithProjection,
                              CLIPTextModelWithProjection, CLIPTokenizer)
    cm = "openai/clip-vit-base-patch16"
    cp = CLIPImageProcessor.from_pretrained(cm)
    cn = CLIPVisionModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()
    tk = CLIPTokenizer.from_pretrained(cm)
    tn = CLIPTextModelWithProjection.from_pretrained(
        cm, use_safetensors=True).to(args.device).eval()

    def emb_img(paths, bs=32):
        out = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            with torch.no_grad():
                e = cn(**cp(images=ims, return_tensors="pt").to(args.device)).image_embeds
            out.append(e.cpu().numpy().astype(np.float32))
        E = np.concatenate(out)
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def emb_txt(ts_):
        with torch.no_grad():
            t = tk(ts_, padding=True, truncation=True, return_tensors="pt").to(args.device)
            e = tn(**t).text_embeds.cpu().numpy().astype(np.float32)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    TAUS = [None, 12.0, 1.0, 0.1]
    acc = {t: defaultdict(list) for t in TAUS}
    per_cat = defaultdict(lambda: defaultdict(list))
    tmp = tempfile.mkdtemp()
    used = 0
    for vi_, vid in enumerate(vids):
        qs = by[vid]
        rel = qs[0]["video"]
        try:
            mp4 = hf_hub_download("facebook/S-EMBER", rel, repo_type="dataset",
                                  local_dir=args.root)
        except Exception as e:
            print("  %s 내려받기 실패: %s" % (vid[:24], str(e)[:60]))
            continue
        cap = cv2.VideoCapture(mp4)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0
        secs, paths = [], []
        for i, s in enumerate(np.arange(0, dur, args.every)):
            f = int(round(s * fps))
            if not (0 <= f < total):
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            p = os.path.join(tmp, "s%03d_%05d.jpg" % (vi_, i))
            cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            secs.append(float(s)); paths.append(p)
        cap.release()
        if len(paths) < 20:
            continue
        ts = np.array(secs); E = emb_img(paths)
        T = emb_txt([q["question"] for q in qs])
        for qi, q in enumerate(qs):
            qt = float(q.get("question_time") or dur)
            m = ts <= qt                                   # 인과
            if m.sum() < 10:
                continue
            tt = ts[m]
            a0, a1 = float(q["answer_start_time"]), float(q["answer_end_time"])
            good = (tt >= a0 - args.every) & (tt <= a1 + args.every)
            if not good.any():
                continue
            base = E[m] @ T[qi]
            for tau in TAUS:
                sim = base if tau is None else base * np.exp(-(qt - tt) / (tau * 3600.0))
                order = np.argsort(-sim)
                for k in (1, 5, 10):
                    hit = bool(good[order[:k]].any())
                    acc[tau][k].append(hit)
                    if tau is None:
                        per_cat[q["question_category"]][k].append(hit)
                # **우연 기준선을 제대로** — 증거 구간이 넓어(중앙 24초·p90 168초)
                # 무작위로도 자주 맞는다. k회 뽑아 한 번이라도 드는 확률
                # 1−(1−p)^k 로 문항마다 계산해 평균한다. 종전처럼 k·p 로 잡으면
                # p=0.2 에서 1.0 이 돼 지표가 무의미해진다.
                if tau is None:
                    pp = float(good.mean())
                    for k in (1, 5, 10):
                        acc[tau]["ch%d" % k].append(1.0 - (1.0 - pp) ** k)
        used += 1
        n = len(acc[None][5])
        print("  %-30s 프레임 %-4d 질의 %-3d (누적 %d)" % (vid[:30], len(ts), len(qs), n),
              flush=True)
        if used >= args.n_video:
            break

    n = len(acc[None][5])
    if not n:
        print("판정 없음")
        return
    CH = {k: float(np.mean(acc[None]["ch%d" % k])) for k in (1, 5, 10)}
    print("\n질의 %d · 영상 %d · 표집 %.0f초" % (n, used, args.every))
    print("**우연 기준선**: hit@1 %.3f · hit@5 %.3f · hit@10 %.3f"
          " (증거 구간이 넓어 무작위로도 자주 맞는다)" % (CH[1], CH[5], CH[10]))
    print("\n%-9s %-16s %-16s %s" % ("최근성", "hit@1(우연대비)",
                                     "hit@5(우연대비)", "hit@10(우연대비)"))
    for tau in TAUS:
        h = {k: float(np.mean(acc[tau][k])) for k in (1, 5, 10)}
        print("%-9s %-16s %-16s %s"
              % ("없음" if tau is None else "τ=%.1fh" % tau,
                 "%.3f (%.1f배)" % (h[1], h[1] / max(CH[1], 1e-9)),
                 "%.3f (%.1f배)" % (h[5], h[5] / max(CH[5], 1e-9)),
                 "%.3f (%.1f배)" % (h[10], h[10] / max(CH[10], 1e-9))))
    print("\n질문 유형별 (최근성 없음 · 우연 대비):")
    for c, d in sorted(per_cat.items(), key=lambda x: -np.mean(x[1][5])):
        print("  %-28s n=%-5d hit@1 %.3f · hit@5 **%.3f** (우연대비 %.1f배)"
              % (c, len(d[5]), np.mean(d[1]), np.mean(d[5]),
                 np.mean(d[5]) / max(CH[5], 1e-9)))
    print("\n(대조: SuperMemory 물체·위치 hit@5 **0.75** · EgoLife 0.179)")
    if args.out:
        json.dump({str(t): {str(k): float(np.mean(v)) for k, v in d.items() if k != "chance"}
                   for t, d in acc.items()}, open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
