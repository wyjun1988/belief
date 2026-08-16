#!/usr/bin/env python3
"""SuperMemory-VQA 응답 하니스 — 검색 top-k 프레임 + Qwen3.5(ollama) 4지선다.

    $P scripts/supermem_answer.py --model qwen3.5:4b --mode vlm
    $P scripts/supermem_answer.py --model qwen3.5:4b --mode text   # 무영상 기준선

파이프라인(이번 스코프): QA query → 프레임 선별(CLIP 중심화+평활+NMS, 실측
hit@5 0.55 조합) → VLM 이 4지선다 응답. 채점은 correct_option_index.
기준선: 무작위(≈25%), 그리고 --mode text(질문만, 영상 없음 — 검색의 기여 분리).
결과는 jsonl 로 캐시해 중단 재개가 된다.
"""
import argparse
import base64
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
D = os.path.join(ROOT, "data", "supermem")
SESS = {
    "Person_1_session_8_03102026_glasses_1264": "s8",
    "Person_1_session_14_03152026_glasses_1266": "s14",
}
OLLAMA = "http://localhost:11434/api/chat"


def load_index():
    E, ts, sid = [], [], []
    for vid, sd in SESS.items():
        z = np.load(os.path.join(D, sd, "index.npz"))
        E.append(z["emb"].astype(np.float32))
        ts.extend(z["ts"])
        sid.extend([vid] * len(z["ts"]))
    return np.concatenate(E), np.array(ts), np.array(sid)


def questions(ts_by_sess=None):
    q = json.load(open(os.path.join(D, "qa_person_1.json")))
    out = []
    for x in q:
        ev = [(e["video_id"], e["time_span"]["start_time"], e["time_span"]["end_time"])
              for e in ((x.get("answer_evidence") or {}).get("evidence_list") or [])
              if e.get("video_id") in SESS and e.get("time_span")]
        if ev:
            out.append((x, ev))
    return out


def grab_frame(vid, sec):
    cap = cv2.VideoCapture(os.path.join(D, SESS[vid], "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    fr = cv2.resize(fr, (448, 448))
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def ask(model, prompt, images, timeout=600):
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images
    # think=False 필수 — 켜두면 내용이 사고 토큰으로 빠져 content 가 빈다(실측)
    body = json.dumps({"model": model, "messages": [msg], "stream": False,
                       "think": False,
                       "options": {"temperature": 0, "num_predict": 200}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def parse_choice(text):
    """응답에서 A~D 를 뽑는다 — 마지막에 나온 단독 대문자를 우선."""
    import re
    m = re.findall(r"\b([ABCD])\b", text.upper())
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--mode", default="vlm", choices=["vlm", "text"])
    ap.add_argument("--topk", type=int, default=4, help="VLM 에 줄 프레임 수")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_p = args.out or os.path.join(D, "answers_%s_%s.jsonl"
                                     % (args.model.replace(":", "_").replace("/", "_"), args.mode))
    done = {}
    if os.path.exists(out_p):
        for ln in open(out_p):
            r = json.loads(ln)
            done[r["question_id"]] = r

    Q = questions()
    if args.limit:
        Q = Q[:args.limit]

    # 검색 점수 (supermem_pilot 과 동일 조합)
    picked_frames = {}
    if args.mode == "vlm":
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        E, ts, sid = load_index()
        dev = torch.device("mps")
        name = "openai/clip-vit-base-patch16"
        tok = CLIPTokenizer.from_pretrained(name)
        txt = CLIPTextModelWithProjection.from_pretrained(
            name, use_safetensors=True).eval().to(dev)
        with torch.no_grad():
            tt = tok(["a photo of " + x["question"] for x, _ in Q],
                     padding=True, truncation=True, return_tensors="pt").to(dev)
            te = torch.nn.functional.normalize(txt(**tt).text_embeds, dim=-1).cpu().numpy()
        S = te @ E.T
        S = S - S.mean(0, keepdims=True)
        k15 = np.ones(15) / 15
        S = np.apply_along_axis(lambda r: np.convolve(r, k15, mode="same"), 1, S)
        for i, (x, _) in enumerate(Q):
            order = np.argsort(-S[i])
            picked = []
            for j in order:
                if all(abs(ts[j] - ts[p]) > 30 or sid[j] != sid[p] for p in picked):
                    picked.append(j)
                if len(picked) >= args.topk:
                    break
            picked_frames[x["question_id"]] = [(sid[j], float(ts[j])) for j in picked]

    letters = "ABCD"
    n_ok = n = 0
    fout = open(out_p, "a")
    for qi, (x, ev) in enumerate(Q):
        qid = x["question_id"]
        if qid in done:
            r = done[qid]
            n += 1
            n_ok += r["correct"]
            continue
        choices = "\n".join("%s. %s" % (letters[i], c) for i, c in enumerate(x["choices"]))
        if args.mode == "vlm":
            frames = picked_frames[qid]
            images = [im for im in (grab_frame(v, s) for v, s in frames) if im]
            prompt = ("These are frames retrieved from my past egocentric video "
                      "(my first-person view at home).\n"
                      "Question: %s\n%s\n"
                      "Answer with the single letter of the best choice."
                      % (x["question"], choices))
        else:
            images = []
            prompt = ("Question about my past activities at home: %s\n%s\n"
                      "Answer with the single letter of the best choice."
                      % (x["question"], choices))
        try:
            resp = ask(args.model, prompt, images)
        except Exception as e:
            print("[%d] 오류: %s" % (qid, e))
            continue
        ch = parse_choice(resp)
        correct = int(ch == letters[x["correct_option_index"]])
        n += 1
        n_ok += correct
        rec = dict(question_id=qid, skill=x["metadata"]["skill"], choice=ch,
                   correct=correct, gt=letters[x["correct_option_index"]],
                   resp=resp[-200:])
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if qi % 10 == 0:
            print("%d/%d · 정답률 %.2f" % (n, len(Q), n_ok / max(n, 1)))
    fout.close()

    # 집계
    from collections import defaultdict
    by = defaultdict(lambda: [0, 0])
    allr = [json.loads(ln) for ln in open(out_p)]
    seen = {}
    for r in allr:
        seen[r["question_id"]] = r
    for r in seen.values():
        by[r["skill"]][0] += r["correct"]
        by[r["skill"]][1] += 1
    tot_ok = sum(h for h, _ in by.values())
    tot = sum(t for _, t in by.values())
    print("\n[%s · %s] 전체 정답률 **%.2f** (%d/%d) · 무작위 ≈0.25"
          % (args.model, args.mode, tot_ok / max(tot, 1), tot_ok, tot))
    for sk, (h, t) in sorted(by.items()):
        print("  %-24s %.2f (%d)" % (sk, h / t, t))


if __name__ == "__main__":
    main()
