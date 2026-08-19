#!/usr/bin/env python3
"""비전 경로 단독 검사 — 4B vs 9B mmproj 가 프레임을 실제로 읽는가.

    LLAMA_SERVER=... $P scripts/vlm_vision_probe.py --tag 4b

QA 정답률로는 비전 품질과 추론 품질이 섞인다. 여기서는 **정답 근거 프레임**을
주고 "이 사진에 무엇이 보이나"만 물어, GT 세그 카테고리와 얼마나 겹치는지 잰다.
9B 가 QA 에서 나빴던 것이 mmproj 시각 경로 탓인지 가르는 실험이다.

채점: 모델이 나열한 명사와 그 프레임 근처에서 실제 관측된 물체(OWLv2 문턱 0.3)의
겹침 — 재현율(관측된 것 중 몇 개를 말했나)과 환각율(말한 것 중 관측 안 된 비율).
"""
import argparse, base64, io, json, os, re, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
SERVER = os.environ.get("LLAMA_SERVER", "http://192.168.219.123:8080/v1/chat/completions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="결과 파일 태그 (4b / 9b / 4b_bf16)")
    ap.add_argument("--n", type=int, default=40, help="검사 프레임 수")
    ap.add_argument("--res", type=int, default=784)
    args = ap.parse_args()

    import cv2
    import urllib.request
    from scripts.owl_presence import load_owl
    from scripts.supermem_answer import SESS, grab_frame

    owl = load_owl({sd: os.path.join(D, "owl_sm_%s.json" % sd) for sd in SESS.values()})
    # 검사 프레임: 검출이 풍부한 프레임 위주(읽을 것이 있는 프레임)
    picks = []
    for vid, sd in SESS.items():
        det = owl.get(sd, {})
        rich = sorted(det, key=lambda i: -len([1 for s in det[i].values() if s >= 0.30]))
        for i in rich[: args.n // max(len(SESS), 1)]:
            truth = {w for w, s in det[i].items() if s >= 0.30}
            if len(truth) >= 3:
                picks.append((vid, sd, i, truth))
    picks = picks[: args.n]
    print("검사 프레임 %d · 프레임당 관측 물체 중앙 %d개"
          % (len(picks), int(np.median([len(t) for _, _, _, t in picks]))))

    out, t0 = [], time.time()
    for k, (vid, sd, idx, truth) in enumerate(picks):
        b64 = grab_frame(vid, float(idx), res=args.res)   # 이미 base64 JPEG 문자열이다
        if not b64:
            continue
        body = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "List the objects you can see in this photo. "
                                     "Output ONLY a comma-separated list of short noun "
                                     "phrases, no sentences."},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
            "temperature": 0, "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False}}).encode()
        r = urllib.request.Request(SERVER, data=body,
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=300) as z:
                txt = json.loads(z.read())["choices"][0]["message"]["content"]
        except Exception as e:
            print("  [%d] 오류 %s" % (k, str(e)[:50]), flush=True)
            continue
        said = {w.strip().lower() for w in re.split(r"[,\n]", txt) if 1 <= len(w.strip()) <= 30}
        said = {w for w in said if re.fullmatch(r"[a-z][a-z \-]*", w)}
        # 겹침: 부분문자열 일치까지 인정(관측 어휘가 'kitchen counter' 처럼 구)
        hit = {t for t in truth if any(t in s or s in t for s in said)}
        out.append(dict(sess=sd, idx=idx, n_truth=len(truth), n_said=len(said),
                        n_hit=len(hit), said=sorted(said)[:12], truth=sorted(truth)[:12]))
        if k % 10 == 0:
            print("  %d/%d (%.1fs/장)" % (k, len(picks), (time.time() - t0) / max(k, 1)), flush=True)

    rec = np.mean([r["n_hit"] / max(r["n_truth"], 1) for r in out])
    hal = np.mean([1 - r["n_hit"] / max(r["n_said"], 1) for r in out])
    nsaid = np.mean([r["n_said"] for r in out])
    print("\n[%s] 프레임 %d" % (args.tag, len(out)))
    print("  관측 물체 재현율  **%.3f**  (모델이 실제 있는 것 중 몇 %% 를 말했나)" % rec)
    print("  미관측 언급 비율  %.3f  (말한 것 중 관측에 없는 비율)" % hal)
    print("  프레임당 언급 수  %.1f" % nsaid)
    p = os.path.join(D, "vision_probe_%s.json" % args.tag)
    json.dump(out, open(p, "w"), ensure_ascii=False)
    print("→ %s" % p)
    print("\n표본:")
    for r in out[:3]:
        print("  관측: %s" % ", ".join(r["truth"][:8]))
        print("  응답: %s" % ", ".join(r["said"][:8]))


if __name__ == "__main__":
    main()
