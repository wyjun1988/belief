#!/usr/bin/env python3
"""데모 영상 생성 — "이 영상에서 이 질문에 시스템이 이렇게 답했다". (서버, 프레임 필요)

    THOR_ROOT=data/thor7_t7view DUMP=online_dump.jsonl OUT=demo \\
      python scripts/thor_demo.py

입력: eval_online 의 DUMP_JSONL 산출(타겟별 분기·답·근거 프레임).
경우별 대표 1건씩 자동 선정(정답이고 근거가 풍부한 것):
  case1  새 방 목격 → 그 방   (c0 · 이동 · 정답)
  case2  재방문+부재 → belief (c2 · 이동 · 정답)
  case3  관측 없음 → 기록 방  (rec · 이동 없음/미목격 · 정답)
각각 mp4: 질문 카드 → 근거 프레임(주석: 시각·방·역할) → 판정 카드.
ffmpeg 필요. 글자는 영문(서버 한글 폰트 부재 대비).
"""
import glob, json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.environ.get("THOR_ROOT", "data/thor7_t7view")
DUMP = os.environ.get("DUMP", "online_dump.jsonl")
OUT = os.environ.get("OUT", "demo")
FPS = 5
os.makedirs(OUT, exist_ok=True)

def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            try: return ImageFont.truetype(p, sz)
            except Exception: pass
    return ImageFont.load_default()

F28, F20 = font(30), font(20)

def words(t): return "".join(" " + c.lower() if c.isupper() else c for c in t).strip()

def card(lines, size, sec=2.0):
    im = Image.new("RGB", size, (18, 22, 30))
    dr = ImageDraw.Draw(im)
    y = size[1] // 2 - 22 * len(lines)
    for ln, big in lines:
        f = F28 if big else F20
        w = dr.textlength(ln, font=f)
        dr.text(((size[0] - w) / 2, y), ln, fill=(235, 235, 225), font=f)
        y += 44 if big else 30
    return [im] * int(sec * FPS)

def annot(img, top, bottom, box=None):
    im = img.copy(); dr = ImageDraw.Draw(im)
    if box: dr.rectangle(box, outline=(255, 210, 60), width=4)
    for txt, y0 in ((top, 6), (bottom, im.height - 34)):
        if not txt: continue
        w = dr.textlength(txt, font=F20)
        dr.rectangle([4, y0 - 2, 12 + w, y0 + 26], fill=(0, 0, 0))
        dr.text((8, y0), txt, fill=(255, 255, 255), font=F20)
    return im

recs = [json.loads(l) for l in open(DUMP)]
pick = {}
for r in recs:
    if not r["ok"]: continue
    if r["branch"] == "c0" and r["moved"] and r.get("picked"):
        k = "case1"
    elif r["branch"] == "c2" and r["moved"]:
        k = "case2"
    elif r["branch"] == "rec" and not r["moved"]:
        k = "case3"
    else:
        continue
    sc = len(r.get("picked") or [])
    if k not in pick or sc > pick[k][0]:
        pick[k] = (sc, r)

for k, (_sc, r) in sorted(pick.items()):
    hd = os.path.join(ROOT, r["house"])
    g = json.load(open(hd + "/gt.json"))
    live = {m["t"]: m for m in g["live"]}
    lv = {int(os.path.basename(p)[:-4]): p for p in glob.glob(hd + "/live/*.jpg")}
    if not lv:
        print("%s: 프레임 없음(%s) — 건너뜀. 프레임 있는 서버에서 돌릴 것" % (k, hd)); continue
    rt = g["room_types"]
    mv = [m for m in g["moves"] if m["oid"] == r["oid"]]
    frames = []
    q = "Q: Where is the %s now?" % words(r["type"])
    frames += card([(q, True), ("episode %s / 4h ego video" % r["house"], False)],
                   Image.open(next(iter(lv.values()))).size, 2.5)
    # 근거 1: 이동 전 마지막 목격 (기록 방)
    pre = [m["t"] for m in g["live"]
           if r["oid"] in (m.get("vis") or []) and (not mv or m["t"] <= mv[-1]["t"])]
    for t in pre[-2:]:
        if t not in lv: continue
        m = live[t]; c = (m.get("ctr") or {}).get(r["oid"])
        box = [c[0] - 60, c[1] - 60, c[0] + 60, c[1] + 60] if c else None
        frames += [annot(Image.open(lv[t]), "earlier sighting  t=%ds  room=%s" % (t, rt.get(m["room"], m["room"])),
                         "system record: %s" % rt.get(r["record"], r["record"]), box)] * int(1.4 * FPS)
    # 근거 2: 분기별 핵심 프레임
    if k == "case1":
        for t in (r.get("picked") or [])[:3]:
            if t not in lv: continue
            m = live[t]; c = (m.get("ctr") or {}).get(r["oid"])
            box = [c[0] - 60, c[1] - 60, c[0] + 60, c[1] + 60] if c else None
            frames += [annot(Image.open(lv[t]), "VERIFIED fresh sighting  t=%ds" % t,
                             "VLM identity check + geometric projection", box)] * int(1.6 * FPS)
        why = "fresh verified sighting in a new room"
    elif k == "case2":
        rev = [m["t"] for m in g["live"] if m["room"] == r["record"]
               and r["oid"] not in (m.get("vis") or [])]
        for t in rev[-3:]:
            if t not in lv: continue
            frames += [annot(Image.open(lv[t]), "revisit of recorded room  t=%ds" % t,
                             "object NOT seen -> absence evidence", None)] * int(1.4 * FPS)
        why = "absence confirmed -> handed to belief"
    else:
        why = "no new observation -> answer the record"
    ans, tgt = rt.get(r["ans"], r["ans"]), rt.get(r["tgt"], r["tgt"])
    frames += card([("ANSWER: %s" % ans, True), ("(%s)" % why, False),
                    ("ground truth: %s  %s" % (tgt, "[correct]" if r["ok"] else "[wrong]"), False)],
                   frames[-1].size if frames else (768, 768), 3.0)
    tmp = os.path.join(OUT, "_f_%s" % k); os.makedirs(tmp, exist_ok=True)
    for i, im in enumerate(frames):
        im.save(os.path.join(tmp, "%05d.jpg" % i), quality=90)
    mp4 = os.path.join(OUT, "demo_%s_%s_%s.mp4" % (k, r["house"], r["type"]))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(FPS),
                    "-i", os.path.join(tmp, "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4], check=True)
    print("%s: %s · %s → 답 %s (GT %s) · %s" % (k, r["house"], r["type"], ans, tgt, mp4))
print("→", OUT, "/demo_*.mp4  (DriveSyncFiles 로)")
