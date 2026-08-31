#!/usr/bin/env python3
"""**온라인 상태기계 v1** — 동적 씬그래프를 시간 순서로 통째로 돌린다. (헤드라인)

    THOR_ROOT=data/thor4 SG_INIT=gt|hybrid python scripts/eval_online.py

물체마다 기록(record) 하나. 시간 순서로:
  초기화   SG_INIT=gt → t=0 GT 방 / hybrid → initmap(있으면)+온라인 등록(첫 군집)
  갱신     확신 군집(문턱 q0.98·2장+)을 검증기(실측 운용점 모의: 진짜 0.42 수용,
          혼동 개체당 0.05 로 전부-속음)로 확인 → 군집 2장+ 확인되면 그 방으로 덮어씀
          ⚠️ 덮어쓰기는 기록과 **다른 방**이 복수 확인될 때만 (T4 보호 비대칭 규칙)
  질의(끝) 기록 방 부재 게이팅(온라인) 발동 시 belief(기록 제외) / 아니면 기록
"""
import json, glob, os
import numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "~/khcache/h4/cache/a3_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "~/khcache/h4/cache/qc_"))
AXP = os.path.expanduser(os.environ.get("AX_PREFIX", "~/khcache/h4/cache/ax_"))
SG_INIT = os.environ.get("SG_INIT", "gt")
P_ACC, P_ID = 0.42, 0.05
ABS_TH = float(os.environ.get("ABS_TH", "0.055"))   # ⚠ thor4 384 절대값 — 도메인마다 재라 (§89)
# ── 실검·투영 배선 (§110): 실점수+exnew+삼각측량이 마진 게이트를 대체.
#    실점수 없는 타겟은 종전 모의 경로로 후퇴. ──
FRAME_W = int(os.environ.get("FRAME_W", "768"))
VTH = float(os.environ.get("VERIFY_TH", "0"))
VTH2 = float(os.environ.get("VERIFY_TH2", "-1e9"))
VSC = None
if os.environ.get("VERIFY_JSONL"):
    VSC = {}
    for _l in open(os.environ["VERIFY_JSONL"]):
        _d = json.loads(_l)
        VSC[(_d["house"], _d["oid"])] = _d["scored"]
    print("실검증 %d타겟" % len(VSC), flush=True)
GDEP = None
if os.environ.get("GEO_DEPTH"):
    GDEP = {}
    for _l in open(os.environ["GEO_DEPTH"]):
        _d = json.loads(_l)
        GDEP[(_d["house"], _d["t"], _d["oid"])] = _d["d"]
    print("mono-depth %d표본" % len(GDEP), flush=True)
PR = json.load(open("data/thor_prior.json"))

res = {"rec": [], "sys": [], "static": [], "moved_sys": [], "moved_rec": [],
       "case": Counter()}
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json")); sm = g.get("scene_meta")
    if not sm: continue
    live = {m["t"]: m for m in g["live"]}
    rt = g["room_types"]; rids = sorted(rt)
    nrt = Counter(rt[r] for r in rids)
    rtypes = {}
    for v in sm["static"].values():
        rtypes.setdefault(v["room"], Counter())[v["type"]] += 1
    allt = set().union(*[set(c) for c in rtypes.values()])
    idf = {t: 1.0/max(sum(t in rtypes.get(r, ()) for r in rids), 1) for t in allt}
    adj = {r: set() for r in rids}
    for a, b in sm["doors"]:
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    py, px = P // pw, P % pw
    # 투영 재료 (§106-110): 방 폴리곤·앵커 좌표·검출 앵커 캐시
    _geo = None
    if VSC is not None and sm.get("polys"):
        _stp = {k: v["pos"] for k, v in sm["static"].items() if v.get("pos")}
        _zx = np.load(AXP + hn + ".npz", allow_pickle=True) if os.path.exists(AXP + hn + ".npz") else None
        if _stp and _zx is not None:
            _axids = [a for a in list(_zx["anch"]) if a in sm["static"]]
            _cols = [k for k, a in enumerate(list(_zx["anch"])) if a in sm["static"]]
            _XS = _zx["s"][:, _cols]
            _XSc = _XS - np.median(_XS, axis=0, keepdims=True)
            _XPp = _zx["p"][:, _cols]
            _byt = {}
            for k, v in sm["static"].items():
                if v.get("pos"): _byt.setdefault(v["type"], []).append(v["pos"])
            _geo = (sm["polys"], _stp, _byt)
    arm = np.array([live[t]["room"] for t in ts])
    AS = S[:, nT:]
    im = {}
    imf = os.path.join(os.path.realpath(hd), "initmap_owl.json")
    if os.path.exists(imf):
        best = {}
        for i2 in json.load(open(imf)):
            if i2["w"] > best.get(i2["type"], (0,))[0]:
                best[i2["type"]] = (i2["w"], i2["room"])
        im = {t: r for t, (w, r) in best.items()}
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]
        TS = QS[:, j] + STx[:, j]
        vis = np.array([oid in live[t].get("vis", []) for t in ts])
        base = float(np.median(TS))
        _vr = np.random.default_rng(hash((hn, oid)) % 2**31)

        def loc_of(idx):
            acc = {r: 0.0 for r in rids}
            for i in idx:
                cy, cx = py[i, ti], px[i, ti]
                sc = {r: 0.0 for r in rids}
                for c in range(nT, len(vocab)):
                    t = vocab[c]
                    if t not in idf or S[i, c] < .05: continue
                    d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                    w = float(S[i, c]) / (1 + d/6) * idf[t]
                    for r in rids:
                        if t in rtypes.get(r, ()): sc[r] += w
                nb = {arm[i]} | adj.get(arm[i], set())
                for r in rids:
                    if r not in nb: sc[r] *= .25
                t2 = sum(sc.values()) + 1e-9
                for r in rids: acc[r] += max(0., float(TS[i]) - base) * sc[r]/t2
            return max(acc, key=acc.get) if acc else None

        # 군집 (시간순)
        th = np.quantile(TS, 0.98)
        hits = sorted(np.where(TS >= th)[0], key=lambda i: ts[i])
        evs = []
        for i in hits:
            if evs and ts[i] - ts[evs[-1][-1]] <= 90: evs[-1].append(i)
            else: evs.append([i])
        evs = [e for e in evs if len(e) >= 2]

        def ident(i):
            m = live[ts[i]]
            cx = (P[i, ti] % pw + .5) / pw * 384
            cy = (P[i, ti] // pw + .5) / ph * 384
            best = (1e9, None)
            for o2, c in list((m.get("ctr") or {}).items()) + \
                         list((m.get("anch") or {}).items()):
                if not c: continue
                d = np.hypot(c[0]-cx, c[1]-cy)
                if d < best[0]: best = (d, o2)
            return best[1] if best[0] < 40 else "bg%d" % (i // 20)

        coin = {}
        def verify(i):
            if vis[i]: return _vr.random() < P_ACC
            k = ident(i)
            if k not in coin: coin[k] = _vr.random() < P_ID
            return coin[k]

        # ── 투영 국소화 (§107 투표 yaw · §110 삼각측량+depth 후퇴) ──
        def _room_pt(pt):
            polys = _geo[0]
            for r in polys:
                pl = polys[r]; x, z = pt; n2 = len(pl); c2 = False
                for j2 in range(n2):
                    x1, z1 = pl[j2]; x2, z2 = pl[(j2 + 1) % n2]
                    if (z1 > z) != (z2 > z) and x < (x2-x1)*(z-z1)/(z2-z1+1e-12)+x1:
                        c2 = not c2
                if c2: return r
            return min(polys, key=lambda r: min((pt[0]-v[0])**2 + (pt[1]-v[1])**2
                                                for v in polys[r]))
        def _geo_ray(i):
            polys, stp, byt = _geo
            m = live[ts[i]]; ap = m.get("apos")
            if ap is None: return None
            FF = FRAME_W / 2.0
            def pb(cx): return np.degrees(np.arctan((cx - FRAME_W/2.0) / FF))
            def br(dx, dz): return np.degrees(np.arctan2(dx, dz))
            def pxof(pi): return (pi % pw + .5) / pw * FRAME_W
            hyp = []
            for k2 in np.where(_XSc[i] >= 0.15)[0]:
                a = _axids[k2]
                if a not in stp: continue
                hyp.append((br(stp[a][0]-ap[0], stp[a][1]-ap[1]) - pb(pxof(_XPp[i, k2])), 2.0))
            for c in range(nT, len(vocab)):
                inst = byt.get(vocab[c], [])
                if not inst or len(inst) > 4 or S[i, c] < 0.15: continue
                cx = pxof(P[i, c])
                for pos in inst:
                    hyp.append((br(pos[0]-ap[0], pos[1]-ap[1]) - pb(cx), 1.0/len(inst)))
            if not hyp: return None
            best = (0.0, None)
            for y0, _w in hyp:
                w = sum(w2 for y2, w2 in hyp if abs((y2-y0+180) % 360 - 180) <= 12)
                if w > best[0]: best = (w, y0)
            if best[0] < 2.0: return None
            ys = [(np.radians(y), w) for y, w in hyp if abs((y-best[1]+180) % 360 - 180) <= 12]
            sw = sum(w for _y, w in ys)
            yaw = np.degrees(np.arctan2(sum(np.sin(y)*w for y, w in ys)/sw,
                                        sum(np.cos(y)*w for y, w in ys)/sw))
            return ap, yaw + pb(pxof(P[i, ti]))
        def _tri(r1, r2):
            (p1, b1), (p2, b2) = r1, r2
            if abs((b1-b2+180) % 360 - 180) < 15: return None
            if np.hypot(p1[0]-p2[0], p1[1]-p2[1]) < 0.5: return None
            u1 = np.array([np.sin(np.radians(b1)), np.cos(np.radians(b1))])
            u2 = np.array([np.sin(np.radians(b2)), np.cos(np.radians(b2))])
            try:
                t2 = np.linalg.solve(np.stack([u1, -u2], 1),
                                     np.array(p2, float) - np.array(p1, float))
            except np.linalg.LinAlgError:
                return None
            if not (0.3 < t2[0] < 12 and 0.3 < t2[1] < 12): return None
            return np.array(p1, float) + t2[0]*u1
        def _geo_room_d(i):
            ray = _geo_ray(i)
            if ray is None: return None
            ap, b = ray
            d = (GDEP.get((hn, int(ts[i]), oid)) if GDEP is not None
                 else (live[ts[i]].get("dist") or {}).get(oid))
            if d is None: return None
            return _room_pt([ap[0] + d*np.sin(np.radians(b)),
                             ap[1] + d*np.cos(np.radians(b))])

        # ── 상태기계 ──
        # v3: 기록은 **불변** — 질의 시점에 검증된 최신 증거와 마진 비교만 한다.
        # (영구 덮어쓰기는 국소화 노이즈로 정지 물체 기록을 오염시킨다 — v1 실측 −0.11)
        record = (v0["room"] if SG_INIT == "gt" else im.get(v0["type"]))
        ver_all = []
        for e in evs:
            ver = [i for i in e[:6] if verify(i)]
            if len(ver) >= 2:
                ver_all += ver
        if record is None and ver_all:
            record = loc_of(sorted(ver_all)[:5])   # 온라인 등록 (이른 것)
        # 질의용: 최신 검증 3장의 방 질량
        alt = None
        if ver_all:
            recent = sorted(ver_all, key=lambda i: -ts[i])[:3]
            mass = {r: 0.0 for r in rids}
            for i in recent:
                cy, cx = py[i, ti], px[i, ti]
                sc = {r: 0.0 for r in rids}
                for c in range(nT, len(vocab)):
                    t = vocab[c]
                    if t not in idf or S[i, c] < .05: continue
                    d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                    w = float(S[i, c]) / (1 + d/6) * idf[t]
                    for r in rids:
                        if t in rtypes.get(r, ()): sc[r] += w
                nb = {arm[i]} | adj.get(arm[i], set())
                for r in rids:
                    if r not in nb: sc[r] *= .25
                t2 = sum(sc.values()) + 1e-9
                for r in rids: mass[r] += sc[r]/t2
            top = max(mass, key=mass.get)
            if record and top != record and mass[top] > 2.0 * mass.get(record, 0):
                alt = top                           # 마진 게이트 통과 — 경우 0
        # ── 실검·투영 경로 (있으면 모의 마진 게이트를 대체) ──
        if VSC is not None and _geo is not None:
            _rr = VSC.get((hn, oid))
            if _rr is None:
                # 실점수 없는 타겟(배포에선 질의 시 채점될 것; 여기선 미채점=대부분 정지).
                # §104: T4 는 기록이 0.99 로 옳다 — 모의 게이트의 재라우팅은 순손실.
                alt = None
            else:
                alt = None                          # 실점수 있는 타겟은 실경로가 판정
                _pas = [(int(e[0]), e[1]) for e in _rr
                        if e[1] >= VTH and (len(e) < 3 or e[2] >= VTH2)]
                if len(_pas) >= 4:                  # exnew 외형 게이트
                    _qv = [QS[i2, j] for i2, _s in _pas]
                    _qm = float(np.median(_qv))
                    _pas = [(i2, s_) for (i2, s_), q in zip(_pas, _qv) if q >= _qm] or _pas
                _pick = [i2 for i2, _s in _pas][:3]
                if len(_pick) >= 2:
                    _rays = [r for r in (_geo_ray(i2) for i2 in _pick) if r]
                    _pts = []
                    for _a in range(len(_rays)):
                        for _b in range(_a+1, len(_rays)):
                            _pt = _tri(_rays[_a], _rays[_b])
                            if _pt is not None: _pts.append(_pt)
                    if _pts:                        # 삼각측량 성공 = 강한 증거
                        _rm = _room_pt(np.median(np.array(_pts), 0))
                        if _rm and _rm != record: alt = _rm
                    else:                           # 프레임별 투영 2장+ 합의 요구
                        _rms = [x for x in (_geo_room_d(i2) for i2 in _pick) if x]
                        _cc = Counter(_rms).most_common(1)
                        if _cc and _cc[0][1] >= 2 and _cc[0][0] != record:
                            alt = _cc[0][0]
        if record is None:
            record = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]],1), r)
                          for r in rids))[1]
        # 질의: 기록 방 부재 게이팅 (온라인 앞/뒤 1/3 + 앵커 게이팅)
        inr = np.where(arm == record)[0]
        fired = False
        if len(inr) >= 9:
            e_, l_ = inr[:len(inr)//3], inr[-len(inr)//3:]
            k2 = max(3, len(e_)//3)
            sig = AS[e_[np.argsort(-TS[e_])[:k2]]].mean(0)
            sig /= (np.linalg.norm(sig) + 1e-9)
            pe = AS[e_] @ sig/(np.linalg.norm(AS[e_],axis=1)+1e-9)
            pl = AS[l_] @ sig/(np.linalg.norm(AS[l_],axis=1)+1e-9)
            thp = np.quantile(pe, .7)
            ge, gl = e_[pe >= thp], l_[pl >= thp]
            if len(ge) >= 3 and len(gl) >= 3:
                drop = float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9))
                fired = drop >= ABS_TH       # 기본 0.055 = thor4 근사 — thor7 은 ABS_TH 격자로
        if alt is not None:
            ans = alt
            res["case"]["c0"] += 1
        elif fired:
            ans = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]],1), r)
                       for r in rids if r != record))[1]
            res["case"]["c2"] += 1
        else:
            ans = record
            res["case"]["rec"] += 1
        _br = ("c0" if alt is not None else "c2" if fired else "rec")
        # 3경우 분해 (GT 기준): ①이동없음 ②이동+재촬영 ③이동+미재촬영(→부재확인
        # 가능하면 belief 로 넘겨야 하는 부류 — 실제로 넘긴 비율이 핵심)
        if mv:
            _t0 = mv[-1]["t"]
            _seen = bool(np.any(vis & (ts > _t0)))
            _revis = bool(np.any((arm == mv[-1]["frm"]) & (ts > _t0)))
            _ck = "②재촬영" if _seen else ("③belief대상" if _revis else "③재방문없음")
        else:
            _ck = "①이동없음"
        res.setdefault("ck", Counter())[(_ck, _br, ans == tgt)] += 1
        if os.environ.get("DUMP_JSONL"):
            _dp = dict(house=hn, oid=oid, type=v0["type"], branch=_br, ans=ans,
                       tgt=tgt, record=record, ok=bool(ans == tgt), moved=bool(mv))
            try:
                _dp["picked"] = [int(ts[i2]) for i2 in _pick]
            except Exception:
                pass
            open(os.environ["DUMP_JSONL"], "a").write(json.dumps(_dp) + "\n")
        res.setdefault("br", Counter())[(_br, bool(mv), ans == tgt)] += 1
        res["sys"].append(ans == tgt)
        res["rec"].append(record == tgt)
        res["static"].append(v0["room"] == tgt)
        if mv:
            res["moved_sys"].append(ans == tgt); res["moved_rec"].append(record == tgt)

n = len(res["sys"])
print("=== 온라인 상태기계 v1 · %s · SG_INIT=%s · n=%d ===" % (ROOT, SG_INIT, n))
print("  정지 지도(t=0 GT)        %.3f" % np.mean(res["static"]))
print("  **기록(갱신 후)**         **%.3f**" % np.mean(res["rec"]))
print("  **최종 답(부재분기 포함)** **%.3f**" % np.mean(res["sys"]))
print("  이동만: 기록 %.3f · 최종 %.3f (n=%d)"
      % (np.mean(res["moved_rec"]), np.mean(res["moved_sys"]), len(res["moved_sys"])))
print("  분기: %s" % dict(res["case"]))
br = res.get("br", Counter())
ck = res.get("ck", Counter())
print("  ── 3경우 분해 (GT 기준) ──")
for c3 in ("①이동없음", "②재촬영", "③belief대상", "③재방문없음"):
    tot = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3)
    if not tot: continue
    okc = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and o_)
    line = "  %-12s n=%-4d 정답 %.3f" % (c3, tot, okc / tot)
    if c3 == "③belief대상":
        h = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c2")
        ho = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c2" and o_)
        no = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ != "c2" and o_)
        line += " | 실제 belief 인계 %.2f (%d건) · 인계분 정답 %.3f · 미인계분 정답 %.3f" % (
            h / tot, h, ho / max(h, 1), no / max(tot - h, 1))
    if c3 == "②재촬영":
        c0n = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c0")
        c0o = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c0" and o_)
        line += " | 목격채택(c0) %.2f · 채택분 정답 %.3f" % (c0n / tot, c0o / max(c0n, 1))
    print(line)
print("  분기별 정오 (분기, 이동?, 건수, 정답률):")
for b2 in ("c0", "c2", "rec"):
    for mvf in (True, False):
        tot = br.get((b2, mvf, True), 0) + br.get((b2, mvf, False), 0)
        if tot:
            print("    %-3s %-4s n=%-4d %.3f" % (b2, "이동" if mvf else "정지",
                                                 tot, br.get((b2, mvf, True), 0) / tot))
