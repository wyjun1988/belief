#!/usr/bin/env python3
"""SceneDiff 로 부재 증거를 잰다 — ㉓ 의 두 조건을 **실제로 걸 수 있는** 첫 데이터.

    $P scripts/scenediff_absence.py --root <benchmark/data> --subset sdk

㉓ 실측: 우리 부재 지표는 장소를 GT 로 줘도 AUC 0.488 이었다. 다만 무작위가 아니라
**두 조건이 겹칠 때만** 크게 떨어졌다(하락 0.99·1.28 = 정적 중앙의 7~9배):

    조건① 그 자리에 **같은 이름의 다른 개체가 없다**
    조건② **있을 때 실제로 검출됐다**

ADT 는 카테고리 라벨뿐이라 ①을 걸 수 없었고, 두 조건을 다 만족하는 표본이 12건 중
2건이었다. SceneDiff 는 이것을 해결한다 — 라벨에 **인스턴스 번호가 붙어 있고**
(`colander_1`), 객체별로 `in_video1`/`in_video2` 가 명시돼 "이동 = 부재" 가정을
걷어낸다.

    양성 = **Removed** (in_video1 & not in_video2) — 씬에서 사라짐
    음성 = **Moved**   (in_video1 & in_video2)     — 씬에 남았지만 자리가 바뀜

⚠️ SceneDiff 는 **변화한 객체만** 주석한다. 그래서 음성이 "그대로 있는 것" 이 아니라
"옮겨진 것" 이다. 우리에게 **더 어려운** 설정이라 오히려 낫다 — "사라짐" 과 "그냥
움직임" 을 갈라야 하므로, 이득이 나오면 그것이 곧 "부재를 부재로 본다" 는 증거다.
ADT 처럼 정적 물체를 음성으로 쓰면 음성이 너무 쉬워 AUC 가 부풀 수 있다.

⚠️ 이 지표는 **씬 수준 제거**를 잰다(영상 전체에서 키워드가 사라졌는가). ADT 에서
재던 **장소 수준 이탈**("원래 자리에 없다")보다 쉬운 변형이다. 먼저 씬 수준에서
신호가 있는지 확인하고, 있으면 마스크를 써서 장소 수준으로 좁힌다.

실제 `segments.pkl` 스키마는 README 와 다르다(README 는 dict, 실제는 list):
    objects: [{label, in_video1, in_video2, video1_frame_idx, video2_frame_idx,
               video1_status, video2_status, original_obj_idx, deformability}, ...]
    video1_objects / video2_objects: {label: {frame_idx: RLE}}
"""
import argparse, glob, json, os, pickle, re, subprocess, sys, tempfile
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def base_label(lb):
    """`pasta_ box_1` → `pasta box` — 인스턴스 번호를 떼고 텍스트 질의로 만든다."""
    s = re.sub(r"_\d+$", "", str(lb).strip())
    return re.sub(r"\s+", " ", s.replace("_", " ")).strip().lower()


def video_of(d, n):
    for ext in ("mp4", "MOV", "mov", "MP4"):
        p = os.path.join(d, "original_video%d.%s" % (n, ext))
        if os.path.exists(p):
            return p
    return None


def duration_of(mp4):
    """영상 길이(초). ffprobe 가 없으면 OpenCV 로 — MBP 의 번들 ffmpeg 에는
    ffmpeg 만 있고 ffprobe 가 없다."""
    try:
        v = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", mp4],
                           capture_output=True, text=True).stdout.strip()
        if v:
            return float(v)
    except FileNotFoundError:
        pass
    try:
        import cv2
        c = cv2.VideoCapture(mp4)
        fps = c.get(cv2.CAP_PROP_FPS) or 0
        n = c.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        c.release()
        return float(n / fps) if fps > 0 else 0.0
    except Exception:
        return 0.0


def frames_of(mp4, n, tmp, tag):
    dur = duration_of(mp4)
    out = []
    if dur <= 0:
        return out
    for i in range(n):
        t = dur * (i + 0.5) / n
        p = os.path.join(tmp, "%s_%02d.jpg" % (tag, i))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "%.3f" % t, "-i", mp4,
                        "-frames:v", "1", "-y", p], check=False)
        if os.path.exists(p):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="scenediff_benchmark/data 경로")
    ap.add_argument("--subset", default="all", choices=["all", "sdk", "sdv"],
                    help="sdk = HD-EPIC 유래(P##-… 명명) · sdv = 그 외")
    ap.add_argument("--limit", type=int, default=999)
    ap.add_argument("--nframes", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--det-thr", type=float, default=0.05, help="OWLv2 후처리 문턱")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    mdl = "google/owlv2-base-patch16-ensemble"
    proc = Owlv2Processor.from_pretrained(mdl)
    net = Owlv2ForObjectDetection.from_pretrained(mdl).to(args.device).eval()

    def scores(imgs, words):
        S = np.zeros((len(imgs), len(words)), np.float32)
        for i, p in enumerate(imgs):
            im = Image.open(p).convert("RGB")
            with torch.no_grad():
                inp = proc(text=[words], images=im, return_tensors="pt").to(args.device)
                o = net(**inp)
                r = proc.post_process_grounded_object_detection(
                    o, threshold=args.det_thr,
                    target_sizes=torch.tensor([im.size[::-1]]).to(args.device))[0]
            for lb, sc in zip(r["labels"].tolist(), r["scores"].tolist()):
                S[i, lb] = max(S[i, lb], float(sc))
        return S

    tmp = tempfile.mkdtemp()
    rows, used, skipped_dup = [], 0, 0
    for d in sorted(glob.glob(os.path.join(args.root, "*"))):
        if used >= args.limit or not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        is_sdk = bool(re.match(r"^P\d", name))
        if args.subset == "sdk" and not is_sdk:
            continue
        if args.subset == "sdv" and is_sdk:
            continue
        f = os.path.join(d, "segments.pkl")
        if not os.path.exists(f):
            continue
        try:
            seg = pickle.load(open(f, "rb"))
        except Exception:
            continue
        objs = seg.get("objects") or []
        cand = []
        for o in objs:
            a, b = bool(o.get("in_video1")), bool(o.get("in_video2"))
            if not a:
                continue                                  # Added 는 이 과제와 무관
            cand.append((o, "Removed" if not b else "Moved"))
        if not cand:
            continue
        # 조건① — 같은 기본 이름이 이 쌍에 둘 이상이면 텍스트로 개체를 못 가른다
        bc = Counter(base_label(o["label"]) for o, _ in cand)
        keep = [(o, st) for o, st in cand if bc[base_label(o["label"])] == 1]
        skipped_dup += len(cand) - len(keep)
        if not keep:
            continue
        v1, v2 = video_of(d, 1), video_of(d, 2)
        if not v1 or not v2:
            continue
        f1 = frames_of(v1, args.nframes, tmp, name[:24] + "_a")
        f2 = frames_of(v2, args.nframes, tmp, name[:24] + "_b")
        if len(f1) < 3 or len(f2) < 3:
            continue
        words = [base_label(o["label"]) for o, _ in keep]
        S1, S2 = scores(f1, words), scores(f2, words)
        for i, (o, st) in enumerate(keep):
            a = float(np.median(S1[:, i])); b = float(np.median(S2[:, i]))
            rows.append(dict(pair=name, sdk=is_sdk, label=o["label"],
                             word=words[i], status=st,
                             s_before=a, s_after=b, drop=a - b,
                             deform=o.get("deformability")))
        used += 1
        nr = sum(1 for _, s in keep if s == "Removed")
        print("  %-44s %-4s Removed %-2d Moved %-2d (누적 %d)"
              % (name[:44], "SD-K" if is_sdk else "SD-V", nr, len(keep) - nr, len(rows)))

    if args.out:
        json.dump(rows, open(args.out, "w"), ensure_ascii=False)
        print("→ %s" % args.out)
    report(rows, skipped_dup)


def report(rows, skipped_dup):
    def auc(P, N):
        if len(P) < 3 or len(N) < 3:
            return None
        return float(np.mean([(x > y) + 0.5 * (x == y) for x in P for y in N]))

    print("\n쌍 %d · 객체 %d · 조건①로 제외(동명 중복) %d"
          % (len({r["pair"] for r in rows}), len(rows), skipped_dup))
    for tag, sel in (("전체", rows),
                     ("SD-K(부엌)", [r for r in rows if r["sdk"]]),
                     ("SD-V(그외)", [r for r in rows if not r["sdk"]])):
        R = [r for r in sel if r["status"] == "Removed"]
        M = [r for r in sel if r["status"] == "Moved"]
        print("\n[%s] Removed %d · Moved %d" % (tag, len(R), len(M)))
        if len(R) < 3 or len(M) < 3:
            print("  표본 부족")
            continue
        print("  하락 중앙: Removed %+.3f · Moved %+.3f"
              % (np.median([r["drop"] for r in R]), np.median([r["drop"] for r in M])))
        print("  %-16s %-8s %-8s %s" % ("조건②(전 검출≥)", "Removed", "Moved", "AUC"))
        for thr in (0.0, 0.05, 0.1, 0.2, 0.3):
            P = [r["drop"] for r in R if r["s_before"] >= thr]
            N = [r["drop"] for r in M if r["s_before"] >= thr]
            a = auc(P, N)
            print("  %-16.2f %-8d %-8d %s"
                  % (thr, len(P), len(N), "**%.3f**" % a if a else "표본 부족"))


if __name__ == "__main__":
    main()
