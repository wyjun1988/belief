#!/usr/bin/env python3
"""SuperMemory 실데이터 belief — "그 물건 지금 어느 방?" 을 실제로 재는 첫 시험.

    $P scripts/supermem_belief.py

ADT 는 GT 씬그래프가 있지만 SuperMemory 에는 물체·가구 주석이 전혀 없다.
그래서 **전부 지각으로 만든다**:

    ① 방      CLIP 으로 프레임을 방(kitchen/living/bedroom/hallway)으로 분류
    ② 가구    장소 어휘를 CLIP 으로 검출 → 가장 자주 보인 방에 소속시켜 수용체 노드로
    ③ 관측    질문 키워드(열린 어휘)를 프레임에서 검출 → POS 이벤트(그 틱의 방·가구)
    ④ 질의    문항의 질의 시각에서 "지금 어느 방?" · GT = answer_evidence 의 방 라벨

비교 기준선: last-known(마지막 목격 방) · 최빈 방 · 무작위.
belief 모델은 home-jepa two_head_v5 (클래스는 미지 슬롯 — 열린 어휘 실측 근거).
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HJ = os.path.expanduser("~/work/home-jepa")
sys.path.insert(0, HJ)
sys.path.insert(0, os.path.join(HJ, "scripts"))
D = os.path.join(ROOT, "data", "supermem")

ROOMS = ["kitchen", "living_room", "bedroom", "entrance"]     # home-jepa 어휘 내
ROOM_TEXT = {"kitchen": "a kitchen with counter, sink and stove",
             "living_room": "a living room with a sofa and tv",
             "bedroom": "a bedroom with a bed",
             "entrance": "a hallway or entrance corridor"}
NORM = {"kitchen": "kitchen", "apartment kitchen": "kitchen",
        "living room": "living_room", "apartment living area": "living_room",
        "an apartment living area": "living_room", "living area": "living_room",
        "bedroom": "bedroom", "hallway": "entrance", "entrance": "entrance"}
# (표시 이름, home-jepa 수용체 타입) — 타입은 고정 어휘라 유효값으로 사상한다
PLACES = {"kitchen": [("kitchen counter", "counter"), ("kitchen drawer", "drawer_top"),
                      ("kitchen cabinet", "cabinet_top"), ("refrigerator", "fridge_top"),
                      ("stove", "counter"), ("sink", "sink"),
                      ("kitchen island", "counter"), ("dining table", "dining_table")],
          "living_room": [("sofa", "sofa"), ("coffee table", "coffee_table"),
                          ("tv stand", "tv_stand"), ("bookshelf", "shelf")],
          "bedroom": [("bed", "bed"), ("nightstand", "nightstand"),
                      ("dresser", "wardrobe_top")],
          "entrance": [("shoe rack", "shoe_cabinet"), ("console table", "console")]}
TICK_S = 60


def norm_room(s):
    s = (s or "").strip().lower()
    for k, v in NORM.items():
        if s.startswith(k):
            return v
    if "kitchen" in s:
        return "kitchen"
    if "living" in s:
        return "living_room"
    if "bed" in s:
        return "bedroom"
    return None


def clip_text(texts, device="mps"):
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    nm = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(nm)
    txt = CLIPTextModelWithProjection.from_pretrained(nm, use_safetensors=True).eval().to(device)
    out = []
    for i in range(0, len(texts), 256):
        with torch.no_grad():
            tt = tok(texts[i:i + 256], padding=True, truncation=True,
                     return_tensors="pt").to(device)
            out.append(torch.nn.functional.normalize(
                txt(**tt).text_embeds, dim=-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model", default="supervised_two_head_v5")
    ap.add_argument("--z-obj", type=float, default=1.5, help="물체 검출 z 문턱")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from homejepa.model import CLASS_NAMES, EpTensors, make_batch
    from homejepa.world import ROOM_TYPES
    from scripts.belief_engine import load_model
    from scripts.supermem_answer import questions

    # 5세션 통합 — 방 다양성 확보(s8·s14 만으로는 GT 95% 가 부엌이라 최빈 기준선이 1.00)
    SESS5 = {
        "Person_1_session_1_01312026_glasses_1266": "s1",
        "Person_1_session_8_03102026_glasses_1264": "s8",
        "Person_1_session_14_03152026_glasses_1266": "s14",
        "Person_1_session_19_03292026_glasses_1266sm": "s19",
        "Person_1_session_20_03292026_glasses_1284": "s20",
    }
    Es, tss, sids = [], [], []
    for vid, sd in SESS5.items():
        f = os.path.join(D, sd, "index.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        Es.append(z["emb"].astype(np.float32))
        tss.extend(z["ts"])
        sids.extend([vid] * len(z["ts"]))
    E = np.concatenate(Es)
    ts, sid = np.array(tss), np.array(sids)
    print("세션 %d개 · 프레임 %d" % (len(Es), len(E)))
    starts = json.load(open(os.path.join(D, "session_starts.json")))
    abst = np.array([starts[v] + t for v, t in zip(sid, ts)], float)
    kwj = json.load(open(os.path.join(D, "v3_keywords.json")))

    # ① 프레임 → 방
    RV = clip_text([ROOM_TEXT[r] for r in ROOMS], args.device)
    RS = RV @ E.T
    RSz = (RS - RS.mean(1, keepdims=True)) / (RS.std(1, keepdims=True) + 1e-9)
    frame_room = np.argmax(RSz, 0)
    print("① 프레임 방 분류: %s" % dict(Counter(ROOMS[i] for i in frame_room)))

    # ② 장소 어휘 → 수용체 노드 (방 소속 고정)
    plist, prooms, ptypes = [], [], []
    for r, ps in PLACES.items():
        for p, t in ps:
            plist.append(p)
            prooms.append(r)
            ptypes.append(t)
    PV = clip_text(["a photo of a " + p for p in plist], args.device)
    PS = PV @ E.T
    PSz = (PS - PS.mean(1, keepdims=True)) / (PS.std(1, keepdims=True) + 1e-9)
    print("② 수용체 노드 %d개 (방 %d)" % (len(plist), len(ROOMS)))

    # 질의 대상: 물체·위치 문항 중 방 라벨이 있는 것
    allq = json.load(open(os.path.join(D, "qa_person_1.json")))
    Q = []
    for x in allq:
        if x["metadata"]["skill"] != "object_location_memory":
            continue
        ev = [(e.get("room"), e.get("start_time"))
              for e in ((x.get("answer_evidence") or {}).get("evidence_list") or [])]
        gtr = next((norm_room(r) for r, _ in ev if norm_room(r)), None)
        if gtr and str(x["question_id"]) in kwj:
            Q.append((x, gtr))
    if args.limit:
        Q = Q[:args.limit]
    print("③ 질의 %d문항 · GT 방 분포 %s"
          % (len(Q), dict(Counter(g for _, g in Q))))

    # 물체 키워드 임베딩
    kws = [kwj[str(x["question_id"])]["keyword"] for x, _ in Q]
    KV = clip_text(["a photo of a " + k for k in kws], args.device)
    KS = KV @ E.T
    KSz = (KS - KS.mean(1, keepdims=True)) / (KS.std(1, keepdims=True) + 1e-9)

    # ④ 에피소드 구성 + 채점
    dev = torch.device("cpu")
    model = load_model(args.model, dev)
    NC = len(CLASS_NAMES)
    rows = []
    for qi, (x, gtr) in enumerate(Q):
        qabs = x["metadata"]["primary_video_start_time"] + \
            (((x.get("question_evidence") or {}).get("time_spans") or [{}])[0]
             .get("start_time") or 0)
        past = np.nonzero(abst <= qabs)[0]
        if len(past) < 20:
            continue
        # 관측 이벤트: 키워드가 검출된 프레임 → 그 프레임의 방·최적 수용체
        det = past[KSz[qi, past] >= args.z_obj]
        if len(det) == 0:
            rows.append(dict(q=x["question_id"], gt=gtr, lastknown=None, model=None))
            continue
        seen_rooms = [ROOMS[frame_room[f]] for f in det]
        lastk = seen_rooms[-1]
        # 에피소드(단일 물체) — home-jepa 입력
        recepts, rid_of = [], {}
        for i, (p, r) in enumerate(zip(plist, prooms)):
            rid_of[i] = len(recepts)
            recepts.append(dict(id=len(recepts), type=ptypes[i], tidx=i,
                                room=ROOMS.index(r), pos=[0.5, 0.5]))
        home = dict(id=1, n_agents=1,
                    rooms=[dict(id=i, type=r, tidx=i,
                                recepts=[rid_of[j] for j in range(len(plist))
                                         if prooms[j] == r]) for i, r in enumerate(ROOMS)],
                    recepts=recepts,
                    objects=[dict(id=0, cls=CLASS_NAMES[0], cidx=0, owner=1, size="s",
                                  home_recept=0, src_instance="q%s" % x["question_id"],
                                  gt_instance=None, src_category=kws[qi],
                                  cls_unknown=True, substituted=False)])
        events = []
        last_rec, last_t = rid_of[0], 0
        for f in det:
            t = int((abst[f] - abst[past[0]]) // TICK_S)
            rm = frame_room[f]
            cand = [j for j in range(len(plist)) if prooms[j] == ROOMS[rm]]
            best = max(cand, key=lambda j: PSz[j, f]) if cand else 0
            events.append(dict(t0=t, t1=t, type="POS", obj=0,
                               recept=rid_of[best], room=int(rm)))
            last_rec, last_t = rid_of[best], t
        qt = int((qabs - abst[past[0]]) // TICK_S)
        ep = dict(home=home, days=1, events=events, glances=[],
                  gt={}, gt_moves=[], seed=0, source="supermem",
                  queries=[dict(obj=0, qt=qt, gt_recept=last_rec,
                                gt_room=ROOMS.index(gtr),
                                last_recept=last_rec, last_t=last_t)])
        try:
            E_t = EpTensors(ep, 256, noid=True)
            E_t.cls[:] = NC
            for q in E_t.queries:
                q["qcls"] = NC
            with torch.no_grad():
                b = make_batch([E_t], [(0, 0)], dev)
                p = model.log_prob(b).exp().cpu().numpy()[0]
            # 수용체 확률 → 방 확률
            pr = defaultdict(float)
            for k, rr in enumerate(E_t.recepts):
                pr[ROOMS[recepts[k]["room"]]] += float(p[k])
            pred = max(pr, key=pr.get)
        except Exception as e:
            if qi < 2:
                import traceback; traceback.print_exc()
            pred = None
        rows.append(dict(q=x["question_id"], gt=gtr, lastknown=lastk, model=pred))
        if qi % 20 == 0:
            print("   %d/%d" % (qi, len(Q)))

    ok = [r for r in rows if r["lastknown"]]
    n = len(ok)
    if not n:
        print("판정 가능 문항 없음")
        return
    acc_lk = np.mean([r["lastknown"] == r["gt"] for r in ok])
    acc_md = np.mean([r["model"] == r["gt"] for r in ok if r["model"]])
    maj = Counter(r["gt"] for r in ok).most_common(1)[0]
    print("\n판정 가능 %d/%d 문항" % (n, len(rows)))
    print("**방 단위 belief 정확도**")
    print("  최빈 방(%s)      %.2f" % (maj[0], maj[1] / n))
    print("  무작위(4방)       %.2f" % (1 / len(ROOMS)))
    print("  **last-known**    %.2f" % acc_lk)
    print("  **home-jepa**     %.2f" % acc_md)


if __name__ == "__main__":
    main()
