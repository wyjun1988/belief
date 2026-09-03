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
import re as _re
_rtn = lambda x: _re.sub(r"\.\d+$", "", x or "")   # 방유형 정규화 (HSSD region)
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.path.expanduser(os.environ.get("A3_PREFIX", "~/khcache/h4/cache/a3_"))
QCP = os.path.expanduser(os.environ.get("QC_PREFIX", "~/khcache/h4/cache/qc_"))
AXP = os.path.expanduser(os.environ.get("AX_PREFIX", "~/khcache/h4/cache/ax_"))
SG_INIT = os.environ.get("SG_INIT", "gt")
if SG_INIT == "gt":
    print("⚠️  SG_INIT=gt — **초기 씬그래프가 GT 다**. ①(안 움직인 물체) 정답률은\n"
          "    '정답을 그대로 답한' 자명한 값이며 시스템 능력이 아니다.\n"
          "    무GT 수치는 SG_INIT=hybrid (build_initmap 산출 필요). (2026-09-01 반복 실수 가드)",
          flush=True)
P_ACC, P_ID = 0.42, 0.05
ABS_TH = float(os.environ.get("ABS_TH", "0.055"))   # ⚠ thor4 384 절대값 — 도메인마다 재라 (§89)
# ── 실검·투영 배선 (§110): 실점수+exnew+삼각측량이 마진 게이트를 대체.
#    실점수 없는 타겟은 종전 모의 경로로 후퇴. ──
FRAME_W = int(os.environ.get("FRAME_W", "768"))
VTH = float(os.environ.get("VERIFY_TH", "0"))
VTH2 = float(os.environ.get("VERIFY_TH2", "-1e9"))
C0_MIN = int(os.environ.get("C0_MIN", "2"))   # c0 최소 수용 장수 — 1이면 단장+depth 투영 허용
ABS_GEO = os.environ.get("ABS_GEO", "0") == "1"   # 기하 부재: 기록 좌표가 시야에 든 프레임에서 미검출
ABS_ANG = float(os.environ.get("ABS_ANG", "35"))
ABS_DIST = float(os.environ.get("ABS_DIST", "4.0"))
ABS_MINE = int(os.environ.get("ABS_MINE", "3"))
ABS_MINL = int(os.environ.get("ABS_MINL", "3"))
ABS_Q = float(os.environ.get("ABS_Q", "0.5"))      # 전반 비교 분위
                                                   # (gt0.pos + live.yaw 필요 — thor8c3+ 세대)
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
PRIOR_JSON = os.environ.get("PRIOR_JSON", "data/thor_prior.json")
PR = json.load(open(PRIOR_JSON))
# ── 위치·yaw 를 SfM 추정으로 대체 (사다리 '위치:SfM'): POSE_JSONL = {house,t,apos,yaw} ──
POSE = None
if os.environ.get("POSE_JSONL"):
    POSE = {}
    for _l in open(os.environ["POSE_JSONL"]):
        _d = json.loads(_l); POSE[(_d["house"], int(_d["t"]))] = (_d["apos"], _d["yaw"])
    print("SfM 포즈 %d프레임" % len(POSE), flush=True)
if isinstance(PR, dict) and isinstance(PR.get("dest"), dict):
    PR = PR["dest"]          # hssd_move.json 형식 {"dwell","mobility","dest"}
# ⚠️ 종전엔 thor_prior.json(THOR 어휘)을 HSSD 타입에 적용해 전부 미등록 → 인계분 답이
#    사실상 난수였다 (AUDIT_20260902). PRIOR_JSON 으로 도메인 어휘를 지정할 것.

# ── 재료 사다리 (AUDIT_20260902 조치1): 어떤 GT 가 들어갔는지 사람이 아니라 코드가 찍는다 ──
_LG = os.environ.get("LOC_GEO", "0") == "1"
_ANCH_EX = float(os.environ.get("ANCH_EX", "0.80")); _ANCH_TY = float(os.environ.get("ANCH_TY", "0.10")); _ANCH_DP = int(os.environ.get("ANCH_DP", "2"))
LADDER = "초기맵:%s · 위치:%s · 포즈:%s · 거리:%s · 검증:%s · vis:GT(인스턴스선택·부재기하) · 카메라방:GT · 사전확률:%s · c0창:%s%s%s%s · 앵커게이트:%.2f/%.2f/%d%s · 부재:%s" % (
    "GT" if SG_INIT == "gt" else "검출",
    "SfM" if POSE is not None else "GT(apos)",
    ("SfM" if POSE is not None and os.environ.get("LOC_YAW_GT") != "1" else "GT" if os.environ.get("LOC_YAW_GT") == "1" else "투표") if _LG else "융합(비기하)",
    ("DA" if os.environ.get("GEO_DEPTH") else "GT") if _LG else "—",
    "실측" if VSC is not None else "모의(GT vis)",
    os.path.basename(PRIOR_JSON), os.environ.get("C0_WIN", "3"), "(광선만)" if os.environ.get("C0_RAYPICK") == "1" else "",
    ("(≤%sm)" % os.environ.get("C0_MAXD")) if os.environ.get("C0_MAXD") else "", "(방위다양)" if os.environ.get("C0_DIVERSE") == "1" else "", _ANCH_EX, _ANCH_TY, _ANCH_DP,
    (" · yaw:이동방향우선(정지시 투표)" if os.environ.get("YAW_ORDER") == "motion_first" else " · yaw대체:이동방향" if os.environ.get("YAW_FALLBACK") == "motion" else ""),
    ("기하(%s)" % os.environ.get("ABS_MODE", "spot")) if ABS_GEO else "점수마진")
_NGT = sum(k in LADDER for k in ("포즈:GT", "거리:GT", "초기맵:GT", "모의(GT", "위치:GT"))
print("재료 사다리 → " + LADDER, flush=True)
if _NGT:
    print("⚠️  GT 재료 %d종 포함 — 이 수치를 '무GT' 라 부르지 말 것" % _NGT, flush=True)

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
    if POSE is not None:
        _nrep = 0
        for _t, _m in live.items():
            _pv = POSE.get((hn, int(_t)))
            _m["apos_gt"], _m["yaw_gt"] = _m.get("apos"), _m.get("yaw")
            if _pv: _m["apos"], _m["yaw"] = _pv[0], _pv[1]; _nrep += 1
            else: _m["apos"] = None                   # 포즈 없는 프레임은 기하에서 기권
        print("  %s SfM 포즈 대체 %d/%d" % (hn, _nrep, len(live)), flush=True)
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
        _stp = {k: ([v["pos"][0], v["pos"][2]] if len(v["pos"]) == 3 else v["pos"])
                for k, v in sm["static"].items() if v.get("pos")}
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
    im = {}; im_inst = {}
    imf = os.path.join(os.path.realpath(hd), os.environ.get("INITMAP_FILE", "initmap_owl.json"))
    if os.path.exists(imf):
        best = {}
        for i2 in json.load(open(imf)):
            if i2["w"] > best.get(i2["type"], (0,))[0]:
                best[i2["type"]] = (i2["w"], i2["room"])
            if i2.get("pos"):        # 인스턴스판: 타입당 여러 (방, 좌표)
                im_inst.setdefault(i2["type"], []).append((i2["pos"], i2["room"], i2["w"]))
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
        _vr = np.random.default_rng(__import__('zlib').crc32((hn + '|' + oid).encode()))   # 문자열 hash 는 프로세스마다 달라 결과가 흔들렸다

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
            if os.environ.get("LOC_YAW_GT") == "1" and m.get("yaw") is not None:
                # 사다리 ①: 포즈 = 시뮬 제공물. ⚠️ 이 분기는 2026-09-02 밤까지 **없었다** — "포즈:GT" 로
                # 표기된 HSSD 판 전부가 실제로는 앵커 투표(구 게이트)로 돌았다 (§129 정정).
                return ap, float(m["yaw"]) + pb(pxof(P[i, ti]))
            if os.environ.get("YAW_ORDER") == "motion_first":
                # 이동 중이면 진행 방향을 yaw 로 먼저 쓴다 (v2.2 실측: 오차 중앙 0.8°·<10° 0.86 — 투표 3.4°·0.67 보다
                # 낫다). 정지·회전 중(이동량 <0.2m)에는 앵커 투표로. 배포에선 VO 의 상대 포즈가 이 역할.
                _t = ts[i]; _a, _b2 = live.get(int(_t) - 1), live.get(int(_t) + 1)
                if _a and _b2 and _a.get("apos") and _b2.get("apos"):
                    _dx, _dz = _b2["apos"][0] - _a["apos"][0], _b2["apos"][1] - _a["apos"][1]
                    if np.hypot(_dx, _dz) >= 0.2:
                        return ap, br(_dx, _dz) + pb(pxof(P[i, ti]))
            hyp = []
            # 앵커 **존재 게이트** (2026-09-02): exemplar 점수만으로는 보이지 않는 앵커도 통과해
            # 쓰레기 투표가 된다(오차 23~54°). exemplar ≥ ANCH_EX 이고 그 타입의 OWL 검출이
            # ≥ ANCH_TY 이며 두 패치가 ≤ ANCH_DP 칸 안에서 일치할 때만 가설 (1채 실측: 커버리지
            # 0.55 · 오차 중앙 3.4° · <10° 0.67).
            for k2 in np.where(_XS[i] >= _ANCH_EX)[0]:      # 원점수 (중앙값 뺀 _XSc 에 0.80 을 걸면 아무것도 안 남는다)
                a = _axids[k2]
                if a not in stp: continue
                _ty = sm["static"].get(a, {}).get("type")
                _c = vocab.index(_ty) if _ty in vocab else None
                if _c is None or S[i, _c] < _ANCH_TY: continue
                _pe, _pt = int(_XPp[i, k2]), int(P[i, _c])
                if abs(_pe % pw - _pt % pw) > _ANCH_DP or abs(_pe // pw - _pt // pw) > _ANCH_DP: continue
                hyp.append((br(stp[a][0]-ap[0], stp[a][1]-ap[1]) - pb(pxof(_pe)), 2.0))
            for c in (range(nT, len(vocab)) if not hyp else []):     # 타입 가설은 앵커 가설이 없을 때만
                inst = byt.get(vocab[c], [])
                if not inst or len(inst) > 4 or S[i, c] < 0.25: continue
                cx = pxof(P[i, c])
                for pos in inst:
                    hyp.append((br(pos[0]-ap[0], pos[1]-ap[1]) - pb(cx), 1.0/len(inst)))
            if (not hyp) and os.environ.get("YAW_FALLBACK") == "motion":
                # 투표 불가 프레임: 앞뒤 1프레임 위치 차이의 방위를 yaw 로 (생성기는 카메라를
                # 진행 방향으로 부드럽게 돌린다). 회전 중에는 틀리므로 이동량 ≥0.2m 일 때만.
                _t = ts[i]; _a, _b2 = live.get(int(_t) - 1), live.get(int(_t) + 1)
                if _a and _b2 and _a.get("apos") and _b2.get("apos"):
                    _dx, _dz = _b2["apos"][0] - _a["apos"][0], _b2["apos"][1] - _a["apos"][1]
                    if np.hypot(_dx, _dz) >= 0.2:
                        return ap, br(_dx, _dz) + pb(pxof(P[i, ti]))
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
        if SG_INIT == "gt":
            record = v0["room"]
        else:
            # 인스턴스판이 있으면 **첫 목격 프레임의 투영 위치**에 가장 가까운 인스턴스를
            # 고른다(타입당 방 1개로 접지 않는다 — 실제 주거는 같은 타입이 여러 방에).
            record = im.get(v0["type"])
            _cands = im_inst.get(v0["type"])
            if _cands and _geo is not None:
                _first = next((i2 for i2 in range(len(ts)) if vis[i2]), None)
                if _first is not None:
                    _ry = _geo_ray(_first)
                    _d0 = (live[ts[_first]].get("dist") or {}).get(oid)
                    if _ry and _d0:
                        _ap, _b = _ry
                        _p0 = [_ap[0] + _d0 * np.sin(np.radians(_b)),
                               _ap[1] + _d0 * np.cos(np.radians(_b))]
                        record = min(_cands, key=lambda c3: (c3[0][0]-_p0[0])**2 +
                                     (c3[0][1]-_p0[1])**2)[1]
        ver_all = []
        for e in evs:
            ver = [i for i in e[:6] if verify(i)]
            if len(ver) >= 2:
                ver_all += ver
        if record is None and ver_all:
            record = loc_of(sorted(ver_all)[:5])   # 온라인 등록 (이른 것)
        record0 = record                       # 기준선 '초기맵만' (갱신 전)
        # 기준선 '최신 강검출': 점수 상위 10% 중 가장 최근 프레임의 카메라 방
        _top = np.where(TS >= np.quantile(TS, 0.90))[0]
        base_new = arm[int(max(_top, key=lambda i2: ts[i2]))] if len(_top) else None
        # 기준선 '사전확률만': 그 타입이 있을 법한 방 (c2 와 같은 식, 기록 제외 없음)
        base_pri = max(((PR.get(v0["type"], {}).get(_rtn(rt[r]), .25)/max(nrt[rt[r]],1), r)
                        for r in rids))[1]
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
                _pick = [i2 for i2, _s in _pas][:int(os.environ.get("C0_WIN", "3"))]   # 채택 창 (기본 최신 3장)
                _c0maxd = float(os.environ.get("C0_MAXD", "0"))
                if _c0maxd > 0:
                    # 원거리 목격은 광선 방위 오차가 방을 넘긴다 (v3: k=0 타겟 거짓 인계 3/25) → 거리 상한
                    def _dist_of(i2):   # 실물 거리(DA) 가 있으면 그것, 없으면 GT dist (사다리 재료와 일치)
                        _d = GDEP.get((hn, int(ts[i2]), oid)) if GDEP is not None else None
                        return _d if _d is not None else ((live[ts[i2]].get("dist") or {}).get(oid) or 0)
                    _pas = [(i2, s_) for i2, s_ in _pas if _dist_of(i2) <= _c0maxd]
                    _pick = [i2 for i2, _s in _pas][:int(os.environ.get("C0_WIN", "3"))]
                if os.environ.get("C0_RAYPICK") == "1":
                    # 광선(국소화 가능)이 있는 채택만 창에 — 투표 커버리지 0.64 에서 창이 비는 것을 막는다
                    _pick = [i2 for i2, _s in _pas if _geo_ray(i2)][:int(os.environ.get("C0_WIN", "3"))]
                if os.environ.get("C0_DIAG") == "1":
                    _dg = dict(house=hn, oid=oid, n_rr=len(_rr), n_pas=len(_pas), pick=[int(ts[i2]) for i2 in _pick],
                               rays=[bool(_geo_ray(i2)) for i2 in _pick], record=record)
                if os.environ.get("C0_DIVERSE") == "1" and len(_pas) >= 2:
                    # 거리 불필요한 삼각측량을 살리려면 **방위차가 큰 두 시점**이 필요하다. 최신 3장은 같은 방문(같은 각도)이기
                    # 쉬워 단일 투영(거리 오차)으로 떨어진다. 통과 크롭 전체에서 방위차 최대 쌍 + 최신 1장을 고른다.
                    _cand = [(i2, _geo_ray(i2)) for i2, _s in _pas[:12]]
                    _cand = [(i2, r) for i2, r in _cand if r]
                    _bestp, _bestd = None, 0.0
                    for _a in range(len(_cand)):
                        for _b in range(_a + 1, len(_cand)):
                            (_p1, _b1), (_p2, _b2) = _cand[_a][1], _cand[_b][1]
                            _dd = abs((_b1 - _b2 + 180) % 360 - 180)
                            if _dd > _bestd and np.hypot(_p1[0]-_p2[0], _p1[1]-_p2[1]) >= 0.5: _bestd, _bestp = _dd, (_cand[_a][0], _cand[_b][0])
                    if _bestp and _bestd >= 15:
                        _pick = list(_bestp) + [i2 for i2, _s in _pas[:1] if i2 not in _bestp]
                if len(_pick) >= C0_MIN:
                    _rays = [r for r in (_geo_ray(i2) for i2 in _pick) if r]
                    _pts = []
                    for _a in range(len(_rays)):
                        for _b in range(_a+1, len(_rays)):
                            _pt = _tri(_rays[_a], _rays[_b])
                            if _pt is not None: _pts.append(_pt)
                    if _pts:                        # 삼각측량 성공 = 강한 증거
                        _rm = _room_pt(np.median(np.array(_pts), 0))
                        if _rm and _rm != record: alt = _rm
                        if os.environ.get("C0_DIAG") == "1": _dg.update(tri_room=_rm, n_pts=len(_pts))
                    else:                           # 프레임별 투영 2장+ 합의 요구
                        _msd = float(os.environ.get("C0_MAXD_SINGLE", "0"))
                        if _msd > 0:                # 단일 투영(거리 사용)에만 거리 상한 — 삼각측량용 원거리 크롭은 살린다
                            _pick = [i2 for i2 in _pick if (GDEP.get((hn, int(ts[i2]), oid)) if GDEP is not None else None) is None
                                     or GDEP.get((hn, int(ts[i2]), oid)) <= _msd]
                        _rms = [x for x in (_geo_room_d(i2) for i2 in _pick) if x]
                        _cc = Counter(_rms).most_common(1)
                        if _cc and _cc[0][1] >= min(2, C0_MIN) and _cc[0][0] != record:
                            alt = _cc[0][0]
                        if os.environ.get("C0_DIAG") == "1": _dg.update(proj_rooms=_rms)
                if os.environ.get("C0_DIAG") == "1" and "_dg" in dir():
                    _dg.update(alt=alt, geo=_geo is not None); print("C0_DIAG " + json.dumps(_dg, ensure_ascii=False), flush=True)
        if record is None:
            record = max(((PR.get(v0["type"], {}).get(_rtn(rt[r]), .25)/max(nrt[rt[r]],1), r)
                          for r in rids))[1]
        # 질의: 기록 방 부재 게이팅 (온라인 앞/뒤 1/3 + 앵커 게이팅)
        inr = np.where(arm == record)[0]
        fired = False
        _nlate = -1                                 # 부재확인 기회(자리 본 후반 프레임 수)
        if ABS_GEO and v0.get("pos") is not None:
            # v2 (v1 은 ①을 0.97→0.59 로 붕괴시켜 기각):
            #  ⓐ 같은 방에서 본 프레임만 — v1 은 벽 너머 방향 응시도 "봤다"로 셌다
            #  ⓑ 자기참조 기준 — v1 의 "TS<자기 q0.98=미검출" 은 정지 물체일수록
            #    자동 성립. 대신 "후반 최고 목격 < 전반(있던 시절) 중앙값" 비교
            spot = v0["pos"]
            vis_i = []
            for i in range(len(ts)):
                m = live[ts[i]]
                if m.get("yaw") is None or m.get("apos") is None: continue
                if arm[i] != record: continue
                dx = spot[0] - m["apos"][0]; dz = spot[2] - m["apos"][1]
                if np.hypot(dx, dz) > ABS_DIST: continue
                b = np.degrees(np.arctan2(dx, dz))
                if abs((b - m["yaw"] + 180) % 360 - 180) > ABS_ANG: continue
                vis_i.append(i)
            # 분할점: seen=마지막 목격 시각(기본 — 정지는 후반 창이 좁아 ① 보호,
            # 이동-미재촬영은 이동 전이라 창이 넓어 ③ 회수) / half=에피소드 절반
            if os.environ.get("ABS_SPLIT", "seen") == "seen":
                _sv = np.where(vis)[0]
                cut = ts[int(_sv[-1])] if len(_sv) else ts[len(ts) // 2]
            else:
                cut = ts[len(ts) // 2]
            _e = [i for i in vis_i if ts[i] <= cut]
            _l = [i for i in vis_i if ts[i] > cut]
            _nlate = len(_l)
            _mode = os.environ.get("ABS_MODE", "spot")
            if _mode in ("spot", "both"):
                # v4 자리-국소: 프레임 전역 TS 는 오검출 바닥에 묻힌다(진단 §DIAG).
                # 자리의 예상 화면 x(방위-yaw)와 타겟 패치 위치가 가까우면 "자리에서 잡힘".
                def _at_spot(i):
                    m2 = live[ts[i]]
                    dx2 = spot[0] - m2["apos"][0]; dz2 = spot[2] - m2["apos"][1]
                    db2 = (np.degrees(np.arctan2(dx2, dz2)) - m2["yaw"] + 180) % 360 - 180
                    u = FRAME_W / 2 + np.tan(np.radians(np.clip(db2, -80, 80))) * FRAME_W / 2
                    pu = (P[i, ti] % pw + .5) / pw * FRAME_W
                    return abs(pu - u) < 130
                if len(_e) >= ABS_MINE and len(_l) >= ABS_MINL:
                    r_e = np.mean([_at_spot(i) for i in _e])
                    r_l = np.mean([_at_spot(i) for i in _l])
                    fired = r_e >= 0.5 and r_l <= 0.15
            if not fired and _mode in ("cmp", "both")                and len(_e) >= ABS_MINE and len(_l) >= ABS_MINL:
                fired = float(np.max(TS[_l])) < float(np.quantile(TS[_e], ABS_Q))
            if os.environ.get("ABS_DIAG") == "1" and mv and not np.any(vis & (ts > mv[-1]["t"])):
                print("DIAG %s %s e=%d l=%d fired=%d maxL=%.3f qE=%.3f"
                      % (hn, v0["type"], len(_e), len(_l), int(fired),
                         float(np.max(TS[_l])) if len(_l) else -9,
                         float(np.quantile(TS[_e], ABS_Q)) if len(_e) else -9), flush=True)
        if not fired and len(inr) >= 9:
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
            ans = max(((PR.get(v0["type"], {}).get(_rtn(rt[r]), .25)/max(nrt[rt[r]],1), r)
                       for r in rids if r != record))[1]
            res["case"]["c2"] += 1
        else:
            ans = record
            res["case"]["rec"] += 1
        _bel2 = max(((PR.get(v0["type"], {}).get(_rtn(rt[r]), .25)/max(nrt[rt[r]],1), r)
                     for r in rids if r != (alt if alt else record)))[1]
        _ans2 = (record if alt is not None else record if fired else _bel2)
        res.setdefault("sys2", []).append(tgt in (ans, _ans2))
        _br = ("c0" if alt is not None else "c2" if fired else "rec")
        # 3경우 분해 (GT 기준): ①이동없음 ②이동+재촬영 ③이동+미재촬영(→부재확인
        # 가능하면 belief 로 넘겨야 하는 부류 — 실제로 넘긴 비율이 핵심)
        _outk = ("outdoor", "balcony", "porch", "garage", "yard")
        _is_out = any(k in (rt.get(tgt, tgt) or "").lower() for k in _outk) or (tgt not in rids)   # 범위 밖 방/outside 반출도 ④
        if mv:
            _t0 = mv[-1]["t"]
            _seen = bool(np.any(vis & (ts > _t0)))
            _revis = bool(np.any((arm == mv[-1]["frm"]) & (ts > _t0)))
            if _is_out:
                _ck = "④집밖반출"          # 답이 실내 방이 아님 — 별도 채점
            elif _seen:
                _ck = "②재촬영"
            elif _revis:
                _ck = ("③belief대상" if _nlate < 0 else
                       "③확인기회O" if _nlate >= 2 else "③확인기회X")
            else:
                _ck = "③재방문없음"
        else:
            _ck = "①이동없음"
        res.setdefault("ck", Counter())[(_ck, _br, ans == tgt)] += 1
        res.setdefault("rows", []).append((hn, _ck, ans == tgt, record0 == tgt,
                                           base_new == tgt, base_pri == tgt))
        # ── 증거 조건부 ② (평가 프로토콜 v2, 2026-09-02): 시나리오가 준 목격 수와 시스템 능력을 분리 ──
        if mv and VSC is not None and (hn, oid) in VSC:
            _t0e = mv[-1]["t"]; _k = 0; _at = 0; _af = 0; _fr = []
            for _i, _sab, _sac in VSC[(hn, oid)]:
                _tt = int(ts[_i]); _m = live.get(_tt, {})
                _tr = _tt > _t0e and oid in (_m.get("vis") or [])
                _ok = _sab >= VTH and _sac >= VTH2
                if _tr:
                    _d = (_m.get("dist") or {}).get(oid, 99); _k += (_d < 5); _at += _ok
                    _fr.append(("<2m" if _d < 2 else "2-5m" if _d < 5 else "5m+", _ok))
                else:
                    _af += _ok
            res.setdefault("ev", []).append((hn, _k, _at, _af, ans == tgt, _br, _fr))
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
print("  재료: " + LADDER)
print("  정지 지도(t=0 GT)        %.3f" % np.mean(res["static"]))
print("  **기록(갱신 후)**         **%.3f**" % np.mean(res["rec"]))
print("  **최종 답(부재분기 포함)** **%.3f**" % np.mean(res["sys"]))
print("  top-1+2(후보 2개 누적 — 답변형식 지표, 증거능력 아님) %.3f"
      % np.mean(res.get("sys2", [0])))
print("  이동만: 기록 %.3f · 최종 %.3f (n=%d)"
      % (np.mean(res["moved_rec"]), np.mean(res["moved_sys"]), len(res["moved_sys"])))
print("  분기: %s" % dict(res["case"]))
br = res.get("br", Counter())
ck = res.get("ck", Counter())
print("  ── 3경우 분해 (GT 기준) ──")
for c3 in ("①이동없음", "②재촬영", "③belief대상", "③확인기회O", "③확인기회X",
           "③재방문없음", "④집밖반출"):
    tot = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3)
    if not tot: continue
    okc = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and o_)
    line = "  %-12s n=%-4d 정답 %.3f" % (c3, tot, okc / tot)
    if c3 in ("③belief대상", "③확인기회O", "③확인기회X"):
        h = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c2")
        ho = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c2" and o_)
        if c3 == "③확인기회X":
            # 증거 없음 → 안 넘기는 게 옳은 행동. 지표 = 무인계 준수율 (GT 정답률은 무의미)
            line += " | **무인계 준수 %.2f** (기록 답변이 정책상 옳음)" % (1 - h / tot)
        else:
            # 시스템 몫 = 인계율. 인계분 정답은 belief 모델 몫 (참고 표기)
            line += " | **인계율 %.2f** (%d건, 시스템 몫) · 인계분 정답 %.3f (belief 몫)" % (
                h / tot, h, ho / max(h, 1))
    if c3 == "②재촬영":
        c0n = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c0")
        c0o = sum(v for (c_, b_, o_), v in ck.items() if c_ == c3 and b_ == "c0" and o_)
        line += " | 목격채택(c0) %.2f · 채택분 정답 %.3f" % (c0n / tot, c0o / max(c0n, 1))
    print(line)
# ── 기준선 대조 + 집 단위 부트스트랩 95% CI (AUDIT 조치3) ──
rows = res.get("rows", [])
if rows:
    _rng = np.random.default_rng(0)
    _hs = sorted({r[0] for r in rows}); _byh = {h: [r for r in rows if r[0] == h] for h in _hs}
    def _acc_ci(c3, k):
        sel = [r for r in rows if c3 == "전체" or r[1] == c3]
        if not sel: return None
        acc = float(np.mean([r[k] for r in sel])); bs = []
        for _ in range(1000):
            pick = _rng.choice(_hs, size=len(_hs), replace=True)
            v = [r[k] for h in pick for r in _byh[h] if c3 == "전체" or r[1] == c3]
            if v: bs.append(np.mean(v))
        return acc, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(sel)
    print("  ── 기준선 대조 (집 %d채 부트스트랩 95%% CI) ──" % len(_hs))
    print("  %-12s %-5s %-18s %-18s %-18s %-18s" % ("경우", "n", "시스템", "초기맵만", "최신강검출", "사전확률만"))
    for c3 in ("전체", "①이동없음", "②재촬영", "③belief대상", "③확인기회O", "③확인기회X", "③재방문없음", "④집밖반출"):
        cells = [_acc_ci(c3, k) for k in (2, 3, 4, 5)]
        if cells[0] is None: continue
        print("  %-12s %-5d %s" % (c3, cells[0][3],
              " ".join("%.3f[%.2f,%.2f]  " % (a, lo, hi) for a, lo, hi, _n in cells)))
    print("  (CI 폭이 차이보다 넓으면 그 차이는 결론이 아니다)")
ev = res.get("ev", [])
if ev:
    print("  ── ② 증거 조건부 (능력 곡선 — 시나리오 목격 분포와 분리) ──")
    fr = {"<2m": [0, 0], "2-5m": [0, 0], "5m+": [0, 0]}
    for e in ev:
        for b, ok in e[6]: fr[b][1] += 1; fr[b][0] += ok
    print("  진짜 목격 프레임 검증 통과율: " + " · ".join("%s %d/%d=%.2f" % (b, v[0], v[1], v[0]/max(v[1], 1)) for b, v in fr.items()))
    _nfa = sum(e[3] for e in ev); _ncand = sum(len(VSC[(e[0], None)]) if False else 0 for e in ev)
    print("  오채택(물체 없는 후보 통과) 총 %d장 / 타겟 %d" % (_nfa, len(ev)))
    print("  %-10s %-4s %-8s %-10s %-8s %-8s" % ("근거리목격k", "n", "정답률", "진짜채택≥1", "c0발동", "c0정답"))
    for lo, hi, lab in ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 999, "6+")):
        sel = [e for e in ev if lo <= e[1] <= hi]
        if not sel: continue
        c0 = [e for e in sel if e[5] == "c0"]
        print("  %-10s %-4d %-8.2f %-10.2f %-8.2f %-8s" % (lab, len(sel), np.mean([e[4] for e in sel]),
              np.mean([e[2] >= 1 for e in sel]), len(c0) / len(sel),
              ("%.2f" % np.mean([e[4] for e in c0])) if c0 else "—"))
    print("  (k=0 행의 c0발동 = 거짓 인계율 → 0 이어야 · 6+ 행의 정답률 = 증거가 충분할 때의 판정 능력)")
print("  분기별 정오 (분기, 이동?, 건수, 정답률):")
for b2 in ("c0", "c2", "rec"):
    for mvf in (True, False):
        tot = br.get((b2, mvf, True), 0) + br.get((b2, mvf, False), 0)
        if tot:
            print("    %-3s %-4s n=%-4d %.3f" % (b2, "이동" if mvf else "정지",
                                                 tot, br.get((b2, mvf, True), 0) / tot))
