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
# ⚠️ 채점 문항 수가 여기서 결정된다. 2세션이면 물체·위치 31문항(strict 16),
# 5세션이면 107문항 — 4B/9B 차이가 5문항 단위였던 것이 표본 문제였는지 가른다.
# 세션 추가는 index.npz + video.mp4 가 있어야 한다.
SESS = {
    "Person_1_session_8_03102026_glasses_1264": "s8",
    "Person_1_session_14_03152026_glasses_1266": "s14",
}
if os.environ.get("SESS5"):
    SESS.update({
        "Person_1_session_1_01312026_glasses_1266": "s1",
        "Person_1_session_19_03292026_glasses_1266sm": "s19",
        "Person_1_session_20_03292026_glasses_1284": "s20",
    })
# 기본은 ollama(CPU). --llama-server 로 llama.cpp Vulkan 서버(GPU, 4.6×)를 쓴다.
OLLAMA = "http://localhost:11434/api/chat"
LLAMA_SERVER = os.environ.get("LLAMA_SERVER",
                              "http://localhost:8080/v1/chat/completions")


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


def grab_frame(vid, sec, res=448, crop=None):
    cap = cv2.VideoCapture(os.path.join(D, SESS[vid], "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    if crop:
        # 중앙 크롭 후 확대 — 원본 1408² 를 448 로 줄이면 "어느 서랍" 같은 세부가 사라진다
        h, w = fr.shape[:2]
        c = int(min(h, w) * crop)
        y0, x0 = (h - c) // 2, (w - c) // 2
        fr = fr[y0:y0 + c, x0:x0 + c]
    fr = cv2.resize(fr, (res, res))
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode()


def ask_server(prompt, images, timeout=900, grammar=None, max_tokens=400):
    """llama.cpp 서버(OpenAI 호환) — 이미지는 data URL 로 넣는다."""
    content = [{"type": "text", "text": prompt}]
    for im in images:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + im}})
    payload = {"messages": [{"role": "user", "content": content}],
               "temperature": 0, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    if grammar:
        # 문법 제약 디코딩 — 출력 토큰을 문법이 허용하는 것으로 강제한다.
        # 기권·장황한 추론·파싱 실패를 원천 차단(실측: 기권 59%, 파싱 사고 다수).
        payload["grammar"] = grammar
    body = json.dumps(payload).encode()
    req = urllib.request.Request(LLAMA_SERVER, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def ask(model, prompt, images, timeout=600, grammar=None, max_tokens=400):
    if model.startswith("server"):      # server / server9b — 이름은 결과 파일 구분용
        return ask_server(prompt, images, timeout, grammar, max_tokens)
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images
    # think=False 필수 — 켜두면 내용이 사고 토큰으로 빠져 content 가 빈다(실측)
    body = json.dumps({"model": model, "messages": [msg], "stream": False,
                       "think": False,
                       "options": {"temperature": 0, "num_predict": 400}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def parse_choice(text):
    """'Answer: X' 를 최우선으로 찾는다.

    ⚠️ 이전 판은 '마지막 단독 대문자' 를 답으로 봤는데, 모델이 추론 중에 선택지를
    열거하면("- C. This option is...") 그게 답으로 잡히고, 게다가 토큰 한도(200)에
    걸려 최종 답 전에 잘렸다. 실측 증상: GPU 판 선택 분포가 A 47% 로 편중.
    """
    import re
    up = text.upper()
    m = re.findall(r"ANSWER\s*[:\-]?\s*\(?([ABCD])\b", up)
    if m:
        return m[-1]
    m = re.findall(r"^\s*\(?([ABCD])[\.\)]?\s*$", up, re.M)   # 단독 줄
    if m:
        return m[-1]
    m = re.findall(r"\b([ABCD])\b", up)
    return m[-1] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5:4b")
    ap.add_argument("--mode", default="vlm", choices=["vlm", "text"])
    ap.add_argument("--topk", type=int, default=4, help="VLM 에 줄 프레임 수")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--strict", action="store_true",
                    help="① '답할 수 없음' 선택지를 제시에서 제거(그 답이 정답인 문항은 제외)"
                         " ② 문법 제약으로 출력을 한 글자로 강제. 실측 배경: 모델이 73문항 중"
                         " 43~44회 기권해 정답률이 무작위 아래로 떨어졌다")
    ap.add_argument("--absence", action="store_true",
                    help="부재 증거 필터 — 씬그래프 문맥으로 각 선택지 장소를 찾고(keyword 제거)"
                         " 그 장소 프레임에서 keyword 존재도를 잰다. 관측된 적 없는 선택지는"
                         " 프롬프트에 '증거 없음' 으로 표시해 VLM 판단을 돕는다")
    ap.add_argument("--oracle", action="store_true",
                    help="검색 GT(answer_evidence.time_spans)를 그대로 프레임으로 준다."
                         " '검색이 완벽하면' 의 상한 — 판독 병목의 크기를 확정한다")
    ap.add_argument("--res", type=int, default=448,
                    help="VLM 에 주는 프레임 해상도. 원본 1408² — 448 축소가 판독 병목일 수"
                         " 있다(실측: 검색이 맞아도 물체위치 정답률 0.45)")
    ap.add_argument("--crop", type=float, default=None,
                    help="중앙 크롭 비율(0.6 = 가운데 60%%). 시선 방향 세부를 키운다")
    ap.add_argument("--owl-rerank", type=float, default=0.0,
                    help="검색 점수에 OWLv2 키워드 검출 z 를 이 가중치로 더한다."
                         " CLIP 은 '비슷해 보이는' 프레임을, OWLv2 는 '그 물건이 있는'"
                         " 프레임을 올린다. 0 이면 비활성")
    ap.add_argument("--answer-aware", action="store_true",
                    help="선택지를 검색 질의에 결합(answer-aware retrieval). 실측: 물체위치"
                         " hit@5 0.55→0.70. **개방형에서는 선택지가 없으므로, 그 자리를"
                         " 씬그래프의 후보 위치가 맡는다** — v2 설계의 근거가 되는 실측이다")
    ap.add_argument("--temporal", action="store_true",
                    help="시간 논리 — 근거는 항상 질의보다 앞선다(실측 98/98). 인과"
                         " 마스크로 미래 프레임 제거: 전체 hit@5 0.34→0.47")
    ap.add_argument("--force-guess", action="store_true",
                    help="기권 금지 — 실측: text 모드에서 73문항 중 70회 '답불가' 기권 → 0.11")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_p = args.out or os.path.join(D, "answers_%s_%s%s.jsonl"
                                     % (args.model.replace(":", "_").replace("/", "_"),
                                        args.mode,
                                        ("_fg" if args.force_guess else "")
                                        + ("_t" if args.temporal else "")
                                        + ("_aa" if args.answer_aware else "")
                                        + ("_r%d" % args.res if args.res != 448 else "")
                                        + ("_c%g" % args.crop if args.crop else "")
                                        + ("_oracle" if args.oracle else "")
                                        + ("_strict" if args.strict else "")
                                        + ("_k%d" % args.topk if args.topk != 4 else "")
                                        + ("_abs" if args.absence else "")
                                        # ⚠️ 재순위 가중치가 파일명에 없으면 서로 다른
                                        # 설정이 같은 캐시를 읽는다(전에 topk 로 겪음)
                                        + ("_owl%g" % args.owl_rerank if args.owl_rerank else "")))
    done = {}
    if os.path.exists(out_p):
        for ln in open(out_p):
            r = json.loads(ln)
            done[r["question_id"]] = r

    Q = questions()
    # strict: 기권 선택지를 제거하고 인덱스를 다시 매긴다. 기권이 정답인 문항은 제외
    # (그 문항은 '못 맞히는 게 정답' 이라 강제추측과 양립하지 않는다 — 73→66).
    if args.strict:
        keep = []
        for x, ev in Q:
            ci = x["correct_option_index"]
            if x["choices"][ci].startswith("This question can not"):
                continue
            idx = [i for i, c in enumerate(x["choices"])
                   if not c.startswith("This question can not")]
            y = dict(x)
            y["choices"] = [x["choices"][i] for i in idx]
            y["correct_option_index"] = idx.index(ci)
            keep.append((y, ev))
        print("strict: 기권 선택지 제거 · 문항 %d → %d (무작위 기준선 %.2f)"
              % (len(Q), len(keep), 1.0 / len(keep[0][0]["choices"])))
        Q = keep
    if args.limit:
        Q = Q[:args.limit]

    # 검색 점수 (supermem_pilot 과 동일 조합)
    picked_frames = {}
    if args.mode == "vlm" and args.oracle:
        # 오라클: 근거 구간을 균등 표집 — 검색 성능을 100% 로 고정한 대조군
        for x, ev in Q:
            fr = []
            per = max(1, args.topk // max(len(ev), 1))
            for v, a, b in ev:
                for k in range(per):
                    fr.append((v, a + (b - a) * (k + 0.5) / per))
            picked_frames[x["question_id"]] = fr[:args.topk]
    elif args.mode == "vlm":
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        E, ts, sid = load_index()
        dev = torch.device("mps")
        name = "openai/clip-vit-base-patch16"
        tok = CLIPTokenizer.from_pretrained(name)
        txt = CLIPTextModelWithProjection.from_pretrained(
            name, use_safetensors=True).eval().to(dev)
        import re as _re
        _stop = set("the a an i my me you he she it they them his her their of to in on "
                    "at was were is are am did do does done what where who when why how "
                    "that this these those after before with for and or as from left "
                    "right just thinking about need want get got take took put place "
                    "placed leave store stored".split())

        def _kp(t):
            ws = [w for w in _re.findall(r"[a-z]+", t.lower())
                  if w not in _stop and len(w) > 2]
            return " ".join(ws[-4:]) if ws else t

        def _qtext(x):
            if not args.answer_aware:
                return "a photo of " + x["question"]
            ch = " ".join(c for c in x["choices"] if not c.startswith("This"))
            return "a photo of " + _kp(x["question"]) + ", " + _kp(ch)

        with torch.no_grad():
            tt = tok([_qtext(x) for x, _ in Q],
                     padding=True, truncation=True, return_tensors="pt").to(dev)
            te = torch.nn.functional.normalize(txt(**tt).text_embeds, dim=-1).cpu().numpy()
        S = te @ E.T
        S = S - S.mean(0, keepdims=True)
        k15 = np.ones(15) / 15
        S = np.apply_along_axis(lambda r: np.convolve(r, k15, mode="same"), 1, S)
        if args.owl_rerank:
            import re as _reo
            from scripts.owl_presence import load_owl, owl_z, report_src
            kwj = json.load(open(os.path.join(D, "v3_keywords.json")))
            order_f = []
            for vid, sd_ in SESS.items():
                z_ = np.load(os.path.join(D, sd_, "index.npz"))
                order_f += [(sd_, i) for i in range(len(z_["ts"]))]
            owl = load_owl({sd_: os.path.join(D, "owl_sm_%s.json" % sd_)
                            for sd_ in SESS.values()})
            def _nm(w):
                t = [y for y in _reo.findall(r"[a-z]+", w.lower()) if len(y) > 1]
                return " ".join(t[-2:]) if t else w.lower()
            kwl = [_nm(kwj.get(str(x["question_id"]), {}).get("keyword")
                       or _kp(x["question"])) for x, _ in Q]
            OZ, osrc = owl_z(owl, order_f, kwl, E=E, device="mps")
            report_src(osrc, "검색 재순위")
            OZ = np.apply_along_axis(lambda r: np.convolve(r, k15, mode="same"), 1, OZ)
            S = S + args.owl_rerank * OZ
        if args.temporal:
            starts = json.load(open(os.path.join(D, "session_starts.json")))
            abst = np.array([starts[v] + t for v, t in zip(sid, ts)], float)
            qabs = []
            for x, _ in Q:
                qe = (x.get("question_evidence") or {}).get("time_spans") or []
                qabs.append(x["metadata"]["primary_video_start_time"]
                            + (qe[0]["start_time"] if qe else 0))
            S = np.where(abst[None, :] <= np.array(qabs)[:, None], S, -np.inf)
        for i, (x, _) in enumerate(Q):
            order = np.argsort(-S[i])
            picked = []
            for j in order:
                if not np.isfinite(S[i][j]):
                    break
                if all(abs(ts[j] - ts[p]) > 30 or sid[j] != sid[p] for p in picked):
                    picked.append(j)
                if len(picked) >= args.topk:
                    break
            picked_frames[x["question_id"]] = [(sid[j], float(ts[j])) for j in picked]

    # 부재 증거: 선택지별 (장소 관측 여부, keyword 존재도) 사전 계산
    absinfo = {}
    if args.absence:
        import re as _re2
        from scripts.absence_evidence import PLACES, pmi_graph, presence
        E2, ts2, sid2 = load_index()
        starts2 = json.load(open(os.path.join(D, "session_starts.json")))
        abst2 = np.array([starts2[v] + t for v, t in zip(sid2, ts2)], float)
        words = set(PLACES)
        for x, _ in Q:
            for t in [x["question"]] + list(x["choices"]):
                words |= {w for w in _re2.findall(r"[a-z]+", t.lower()) if len(w) > 3}
        voc = sorted(words)
        Z2, P2 = presence(E2, voc, "mps")
        G2 = pmi_graph(P2, voc)
        vi2 = {w: i for i, w in enumerate(voc)}
        for x, _ in Q:
            qa = x["metadata"]["primary_video_start_time"] + \
                (((x.get("question_evidence") or {}).get("time_spans") or [{}])[0]
                 .get("start_time") or 0)
            past = np.nonzero(abst2 <= qa)[0]
            kws = [vi2[w] for w in _re2.findall(r"[a-z]+", x["question"].lower()) if w in vi2]
            marks = []
            for ch in x["choices"]:
                lw = [vi2[w] for w in _re2.findall(r"[a-z]+", ch.lower()) if w in vi2]
                if not lw or not kws or len(past) < 10:
                    marks.append(""); continue
                cs = Z2[lw][:, past].max(0)
                ok = past[cs >= 1.0]
                if len(ok) < 5:
                    marks.append(" [no observation of this place]"); continue
                sel = ok[np.argsort(-Z2[lw][:, ok].max(0))[:10]]
                pk = float(np.median(Z2[kws][:, sel].max(0)))
                marks.append(" [object NOT seen at this place]" if pk < 0.3 else "")
            absinfo[x["question_id"]] = marks

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
        mk = absinfo.get(qid) or [""] * len(x["choices"])
        choices = "\n".join("%s. %s%s" % (letters[i], c, mk[i] if i < len(mk) else "")
                            for i, c in enumerate(x["choices"]))
        if args.mode == "vlm":
            frames = picked_frames[qid]
            images = [im for im in (grab_frame(v, s, args.res, args.crop)
                                    for v, s in frames) if im]
            fg = ("You MUST choose the single most plausible option even if the "
                  "frames are inconclusive. Guessing is required; there is no "
                  "option to decline. " if (args.force_guess or args.strict) else "")
            prompt = ("These are frames retrieved from my past egocentric video "
                      "(my first-person view at home).\n"
                      "Question: %s\n%s\n%s"
                      "Answer with the single letter of the best choice, and end your "
                      "response with a final line: Answer: <letter>"
                      % (x["question"], choices, fg))
        else:
            images = []
            fg = ("You MUST pick the single most plausible concrete option even "
                  "if unsure; do not pick 'can not be answered'. "
                  if args.force_guess else "")
            prompt = ("Question about my past activities at home: %s\n%s\n%s"
                      "Answer with the single letter of the best choice, and end your "
                      "response with a final line: Answer: <letter>"
                      % (x["question"], choices, fg))
        gram = None
        if args.strict:
            letters_ok = "".join(letters[:len(x["choices"])])
            gram = 'root ::= [%s]' % letters_ok
        try:
            resp = ask(args.model, prompt, images, grammar=gram,
                       max_tokens=4 if args.strict else 400)
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
    base = 1.0 / len(Q[0][0]["choices"]) if Q else 0.25
    print("\n[%s · %s%s] 전체 정답률 **%.2f** (%d/%d) · 무작위 ≈%.2f"
          % (args.model, args.mode, " strict" if args.strict else "",
             tot_ok / max(tot, 1), tot_ok, tot, base))
    for sk, (h, t) in sorted(by.items()):
        print("  %-24s %.2f (%d)" % (sk, h / t, t))


if __name__ == "__main__":
    main()
