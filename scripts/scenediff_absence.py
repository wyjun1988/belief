#!/usr/bin/env python3
"""SD-K 로 부재 증거를 잰다 — ㉓ 의 두 조건을 **실제로 걸 수 있는** 첫 데이터.

    $P scripts/scenediff_absence.py --subset SD-K --limit 30

㉓ 실측: 우리 부재 지표는 장소를 GT 로 줘도 AUC 0.488 이었다. 다만 무작위가 아니라
**두 조건이 겹칠 때만** 크게 떨어졌다(하락 0.99·1.28 = 정적 중앙의 7~9배):

    조건① 그 자리에 **같은 카테고리의 다른 개체가 없다**
    조건② **있을 때 실제로 검출됐다** (이동 전 z ≥ 0)

ADT 는 카테고리 라벨뿐이라 ①을 걸 수 없었고, 두 조건을 다 만족하는 표본이 12건 중
2건이었다. SD-K 는 이것을 해결한다:

  · `objects[id]['in_video1'/'in_video2']` → **Removed 가 명시**됨
    ("이동 = 부재" 가정을 걷어낸다 — ADT 에서는 30%가 변위 1 m 미만이었다)
  · `video1_objects[id][frame]` = RLE 마스크 → **프레임별 인스턴스 신원**
    조건①을 직접 걸 수 있고, 조건②(있을 때 보였나)도 마스크로 검증된다

채점은 우리 지표를 그대로 쓴다: video1(전) 대비 video2(후) 의 키워드 존재도 하락.

    양성 = **Removed** (in_video1 & not in_video2) — 씬에서 사라짐
    음성 = **Moved**   (in_video1 & in_video2)     — 씬에 남았지만 자리가 바뀜

⚠️ SD-K 는 **변화한 객체만** 주석한다. 그래서 음성이 "그대로 있는 것" 이 아니라
"옮겨진 것" 이다. 이것은 우리에게 **더 어려운 쪽으로 유리한** 설정이다 —
"사라짐" 과 "그냥 움직임" 을 갈라야 하므로, 우연 대비 이득이 나오면 그것이 곧
"부재를 부재로 본다" 는 증거다. 반대로 ADT 처럼 정적 물체를 음성으로 쓰면
음성이 너무 쉬워 AUC 가 부풀 수 있다.

⚠️ 또한 이 지표는 **씬 수준 제거**를 잰다(영상 전체에서 키워드가 사라졌는가).
ADT 에서 재던 **장소 수준 이탈**("원래 자리에 없다")과는 다른, 더 쉬운 변형이다.
먼저 씬 수준에서 신호가 있는지 확인하고, 있으면 마스크를 써서 장소 수준으로 좁힌다.
"""
import argparse, json, os, pickle, subprocess, sys, tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "scenediff", "scenediff_benchmark")


def frames_of(mp4, n, tmp):
    """영상에서 n장을 균등 추출 → 경로 목록."""
    out = []
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", mp4], capture_output=True, text=True).stdout.strip() or 0)
    if dur <= 0:
        return out
    for i in range(n):
        t = dur * (i + 0.5) / n
        p = os.path.join(tmp, "%s_%02d.jpg" % (os.path.basename(mp4)[:-4], i))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", mp4,
                        "-frames:v", "1", "-y", p], check=False)
        if os.path.exists(p):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="SD-K", help="scenetype 필터(부분일치). 전체는 ''")
    ap.add_argument("--limit", type=int, default=30, help="시퀀스 쌍 수")
    ap.add_argument("--nframes", type=int, default=12, help="영상당 표본 프레임")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--owl-thr", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dd = os.path.join(D, "data")
    if not os.path.isdir(dd):
        print("데이터 없음: %s — 압축을 먼저 풀어야 한다" % dd)
        return
    pairs = sorted(os.listdir(dd))

    # 지각층 — OWLv2 로 프레임×어휘 점수를 만든다(우리 운용 구성과 동일)
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image
    mdl = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(mdl)
    net = Owlv2ForObjectDetection.from_pretrained(mdl).to(args.device).eval()

    def scores(imgs, words):
        """[frame, word] 최대 검출 점수."""
        S = np.zeros((len(imgs), len(words)), np.float32)
        for i, p in enumerate(imgs):
            im = Image.open(p).convert("RGB")
            with torch.no_grad():
                inp = proc(text=[words], images=im, return_tensors="pt").to(args.device)
                o = net(**inp)
                r = proc.post_process_grounded_object_detection(
                    o, threshold=0.05, target_sizes=torch.tensor([im.size[::-1]]))[0]
            for lb, sc in zip(r["labels"].tolist(), r["scores"].tolist()):
                S[i, lb] = max(S[i, lb], sc)
        return S

    tmp = tempfile.mkdtemp()
    pos, neg, rows = [], [], []
    used = 0
    for pd in pairs:
        if used >= args.limit:
            break
        sd = os.path.join(dd, pd)
        segf = os.path.join(sd, "segments.pkl")
        if not os.path.exists(segf):
            continue
        seg = pickle.load(open(segf, "rb"))
        if args.subset and args.subset.lower() not in str(seg.get("scenetype", "")).lower():
            continue
        objs = seg.get("objects", {})
        # 양성 = Removed(전엔 있고 후엔 없음) · 음성 = 양쪽에 다 있음
        rem = [(k, v) for k, v in objs.items() if v.get("in_video1") and not v.get("in_video2")]
        both = [(k, v) for k, v in objs.items() if v.get("in_video1") and v.get("in_video2")]
        if not rem:
            continue
        words = [str(v.get("label", "")).strip().lower() for _, v in rem + both]
        words = [w for w in words if w]
        if not words:
            continue
        f1 = frames_of(os.path.join(sd, "original_video1.mp4"), args.nframes, tmp)
        f2 = frames_of(os.path.join(sd, "original_video2.mp4"), args.nframes, tmp)
        if len(f1) < 3 or len(f2) < 3:
            continue
        S1, S2 = scores(f1, words), scores(f2, words)
        wi = {w: i for i, w in enumerate(words)}
        cnt = Counter(words)
        for k, v in rem + both:
            w = str(v.get("label", "")).strip().lower()
            if w not in wi:
                continue
            # 조건① 같은 이름의 다른 객체가 이 쌍에 또 있으면 인스턴스가 안 갈린다
            if cnt[w] > 1:
                continue
            a = float(np.median(S1[:, wi[w]]))
            b = float(np.median(S2[:, wi[w]]))
            # 조건② 있을 때(video1) 실제로 검출됐어야 한다
            removed = v.get("in_video1") and not v.get("in_video2")
            rows.append(dict(pair=pd, label=w, removed=bool(removed),
                             z_before=a, z_after=b, drop=a - b))
            (pos if removed else neg).append((a, a - b))
        used += 1
        print("  %-28s Removed %-2d · 양쪽 %-2d · 누적 양성 %d / 음성 %d"
              % (pd[:28], len(rem), len(both), len(pos), len(neg)))

    if not pos or not neg:
        print("표본 부족 — 양성 %d · 음성 %d" % (len(pos), len(neg)))
        return

    def auc(P, N):
        return float(np.mean([(x > y) + 0.5 * (x == y) for x in P for y in N]))

    print("\n시퀀스 쌍 %d · 양성(Removed) %d · 음성(양쪽 존재) %d" % (used, len(pos), len(neg)))
    print("%-14s %-8s %-8s %s" % ("조건②(전 검출)", "양성", "음성", "AUC"))
    for thr in (0.0, 0.05, 0.1, 0.15, 0.2):
        P = [d for a, d in pos if a >= thr]
        N = [d for a, d in neg if a >= thr]
        if len(P) < 3 or len(N) < 3:
            print("%-14.2f %-8d %-8d 표본 부족" % (thr, len(P), len(N)))
            continue
        print("%-14.2f %-8d %-8d **%.3f**" % (thr, len(P), len(N), auc(P, N)))
    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
