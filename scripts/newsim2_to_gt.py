#!/usr/bin/env python3
"""새 시뮬레이터 2차분(260914_belief_daily_routes_v2) → 우리 gt.json 형식 (2026-09-15).

1차분 변환기(newsim_to_gt.py)와 포맷이 완전히 달라 새로 쓴다. 이쪽이 훨씬 풍부하다 —
프레임마다 카메라 포즈가 오고, 사건 라벨(unchanged/relocated/absent)이 대본에 박혀 있다.

  python scripts/newsim2_to_gt.py ~/khcache/newsim2/260914_belief_daily_routes_v2 data/newsim2 [--fps 1]
→ data/newsim2/house_0000/{gt.json, live/*.jpg, map/*.jpg}  (에피소드 하나가 집 하나)

좌표: 언리얼 왼손 Z-up · cm → 우리 규약(오른손 Y-up · m).  x=X/100 · y=Z/100(높이) · z=-Y/100.
yaw 는 언리얼 yaw(왼손, +Z 회전) → 우리 yaw(+Y 위, atan2(-dx,-dz) 기준)로 부호 반전 후 오프셋.
검증은 --check: 가시 물체의 GT 위치를 카메라 포즈로 투영해 주석 박스 중심과 맞는지 본다.
"""
import argparse, json, glob, os, math, collections
import numpy as np
import cv2
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--fps", type=float, default=1.0); ap.add_argument("--max-eps", type=int, default=0)
ap.add_argument("--check", action="store_true"); ap.add_argument("--no-frames", action="store_true")
ap.add_argument("--flat", action="store_true", help="3차분 배치 구조: <root>/<scene>/<episode>/<run>/ (episodes/ 계층 없음)")
ap.add_argument("--scene-graph", default="", help="scene_graph.json 이 배치에 없을 때 쓸 파일(방 기하는 같은 장면이면 동일). 3차분은 scenes/ 가 빠져 왔다")
ap.add_argument("--scan", default="", help="스캔 에피소드 디렉터리 이름(예: 11_ego_coverage_scan). 지정하면 그 프레임을 map 으로 쓴다")
ap.add_argument("--scan-step", type=int, default=30, help="스캔 프레임 솎기(30fps → 30이면 1초마다)")
a = ap.parse_args()
CM = 100.0
def P(v):
    """언리얼(cm, Z-up) → 우리(m, Y-up). **x↔y 축이 바뀐다** — 전수 탐색으로 확정(2026-09-15).
    후보 32종 중 이 조합만 방위 오차 중앙 4.3°(나머지는 42° 이상)."""
    return [v[1] / CM, v[2] / CM, v[0] / CM]      # x←UE_Y · 높이←UE_Z · z←UE_X
def yaw_of(rot):              # rotator [pitch, yaw, roll] — 부호·오프셋 없이 그대로
    return float(rot[1]) % 360.0
def rect_poly(r):             # rect_xy(UE X,Y) → 우리 polys ([x, z] = [UE_Y, UE_X])
    lo, hi = r["min"], r["max"]
    c = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
    return [[y / CM, x / CM] for x, y in c]
def pip(pt, poly):
    x, z = pt; ins = False; n = len(poly)
    for k in range(n):
        x1, z1 = poly[k]; x2, z2 = poly[(k + 1) % n]
        if (z1 > z) != (z2 > z) and x < (x2 - x1) * (z - z1) / (z2 - z1 + 1e-12) + x1: ins = not ins
    return ins
def parse_objs(rec):
    o = rec.get("objects")
    if isinstance(o, str):
        try: o = json.loads(o.replace("'", '"'))
        except Exception:
            import ast
            o = ast.literal_eval(o)
    return o or []
def _dump(mp4, want, outdir, pat, order=None):
    """mp4 에서 지정 프레임만 뽑아 저장. order 가 있으면 그 순서대로 0..n-1 로 번호를 다시 매긴다(map 용)."""
    if not os.path.exists(mp4) or not want: return 0
    os.makedirs(outdir, exist_ok=True)
    idx = {t: i for i, t in enumerate(order)} if order else None
    if all(os.path.exists(os.path.join(outdir, pat % (idx[t] if idx else t))) for t in want): return len(want)
    cap = cv2.VideoCapture(mp4); n = 0; t = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if t in want:
            cv2.imwrite(os.path.join(outdir, pat % (idx[t] if idx else t)), fr, [cv2.IMWRITE_JPEG_QUALITY, 88]); n += 1
        t += 1
    cap.release(); return n
eps = sorted(glob.glob(os.path.join(a.root, "*/*/*/" if a.flat else "episodes/*/*/*/")))
if a.scan: eps = [e for e in eps if ("/%s/" % a.scan) not in e]
_SCANS = {}      # 장면 → 스캔 디렉터리
if a.scan:
    for sd in sorted(glob.glob(os.path.join(a.root, "*/%s/*/" % a.scan if a.flat else "episodes/*/%s/*/" % a.scan))):
        _SCANS[sd.split("/")[-4 if a.flat else -4]] = sd
    print("스캔 에피소드:", {k: v.split("/")[-2] for k, v in _SCANS.items()})
if a.max_eps: eps = eps[:a.max_eps]
os.makedirs(a.out, exist_ok=True)
stat = collections.Counter(); checks = []
for i, e in enumerate(eps):
    hn = "house_%04d" % i; hd = os.path.join(a.out, hn)
    os.makedirs(hd, exist_ok=True); os.makedirs(hd + "/live", exist_ok=True); os.makedirs(hd + "/map", exist_ok=True)
    _sgp = e + "scene_graph.json"
    if not os.path.exists(_sgp):
        if not a.scene_graph: stat["장면그래프 없음"] += 1; continue
        _sgp = a.scene_graph          # 같은 장면이면 방 기하가 동일하다(2차분 10개 파일이 전부 같았다)
    sg = json.load(open(_sgp))
    ent = json.load(open(e + "entities.json"))["entities"]
    ev = json.load(open(e + "evidence.json"))["events"]
    ann = [json.loads(l) for l in open(e + "annotations.jsonl")]
    obs = [json.loads(l) for l in open(e + "observed_graph_updates.jsonl")]
    rooms = [{"id": r["id"], "type": r.get("room_type") or r["id"]} for r in sg["rooms"]]
    polys = {r["id"]: rect_poly(r["rect_xy"]) for r in sg["rooms"]}
    rtypes = {r["id"]: (r.get("room_type") or r["id"]) for r in sg["rooms"]}
    by_actor = {(x.get("actor_name") or "").strip(): x for x in ent}
    by_iid = {x["instance_id"]: x for x in ent}
    # ── gt0 : 대본 사건 물체 + queryable 엔티티(① 질의 확대용)
    gt0 = {}; moves = []
    for x in ev:
        ac = x["object"]; m = by_actor.get(ac.strip())
        oid = "%s|%s" % (x["category"], m["instance_id"] if m else ac)
        gt0[oid] = {"type": x["category"], "room": x["source_room"], "pos": P(x["source_position"]),
                    "_actor": ac, "_iid": (m or {}).get("instance_id"), "_event": x["event_id"]}
        if x["case"] != "unchanged":
            t_mv = x.get("departure_frame")
            if t_mv is None: t_mv = int(round(float(x.get("departure_time_s", 0)) * a.fps))
            moves.append({"t": int(t_mv), "oid": oid, "frm": x["source_room"], "to": x["destination_room"],
                          "intended": x["destination_room"], "pos": P(x["destination_position"]),
                          "witness": False, "supported": True,
                          "role": "c3" if x["case"] == "absent" else "c2", "_case": x["case"]})
        stat[x["case"]] += 1
    # ── live : 프레임마다 카메라 포즈 + 가시 물체
    iid2oid = {v["_iid"]: k for k, v in gt0.items() if v.get("_iid") is not None}
    cam = {}
    for r in obs:
        cs = r.get("cameras") or []
        if cs: cam[int(r["t"])] = cs[0]
    live = []
    for r in ann:
        t = int(r["frame"]); c = cam.get(t)
        if c is None: continue
        ap_ = P(c["location"]); yaw = yaw_of(c["rotation_pyr_deg"]); pitch = float(c["rotation_pyr_deg"][0])
        vr = c.get("visible_regions") or []
        vis, ctr, dist, box = [], {}, {}, {}
        for o in parse_objs(r):
            oid = iid2oid.get(o.get("instance_id"))
            if oid is None: continue
            b = o.get("bbox")
            if not b: continue
            vis.append(oid); ctr[oid] = [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]; box[oid] = b
            gp = np.array(gt0[oid]["pos"], float)
            # 이동 후면 목적지 좌표로
            for mv in moves:
                if mv["oid"] == oid and t > mv["t"]: gp = np.array(mv["pos"], float)
            dist[oid] = round(float(np.hypot(gp[0] - ap_[0], gp[2] - ap_[2])), 3)
        # 카메라가 **서 있는** 방은 기하로 판정한다 — visible_regions 는 '보이는 방' 이라
        # 복도에 서서 거실을 볼 때 거실로 적히고, 그러면 우리 카메라방 지표가 틀어진다(2026-09-15).
        _rm = next((r for r, pl in polys.items() if pip([ap_[0], ap_[2]], pl)), None)
        live.append({"t": t, "room": _rm, "vis": vis, "ctr": ctr, "dist": dist, "seen_rooms": vr,
                     "box": box, "apos": [round(ap_[0], 3), round(ap_[2], 3)], "yaw": round(yaw, 2),
                     "pitch": round(pitch, 2)})
    # ── map : 기록 프레임 = 각 사건의 before_window (스캔 에피소드가 없다)
    # map = 스캔 에피소드(있으면). 없으면 각 사건의 before_window 로 후퇴한다.
    mp = []
    _scene = e.split("/")[-4 if a.flat else -4]
    _sd = _SCANS.get(_scene)
    if _sd:
        _sann = [json.loads(l) for l in open(_sd + "annotations.jsonl")]
        _scam = {}
        for r in (json.loads(l) for l in open(_sd + "observed_graph_updates.jsonl")):
            cs = r.get("cameras") or []
            if cs: _scam[int(r["t"])] = cs[0]
        _sent = json.load(open(_sd + "entities.json"))["entities"]
        _siid = {}
        for x in _sent:
            for oid, v in gt0.items():
                if v.get("_actor") and (x.get("actor_name") or "").strip() == v["_actor"].strip(): _siid[x["instance_id"]] = oid
        for r in _sann:
            t = int(r["frame"])
            if t % a.scan_step: continue
            c = _scam.get(t)
            if c is None: continue
            _ap = P(c["location"]); _yaw = yaw_of(c["rotation_pyr_deg"])
            _rm = next((rr for rr, pl in polys.items() if pip([_ap[0], _ap[2]], pl)), None)
            _ctr, _dist, _box = {}, {}, {}
            for o in parse_objs(r):
                oid = _siid.get(o.get("instance_id"))
                b = o.get("bbox")
                if oid is None or not b: continue
                _ctr[oid] = [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]; _box[oid] = b
                gp = np.array(gt0[oid]["pos"], float)
                _dist[oid] = round(float(np.hypot(gp[0] - _ap[0], gp[2] - _ap[2])), 3)
            mp.append({"room": _rm, "yaw": round(_yaw, 2), "apos": [round(_ap[0], 3), round(_ap[2], 3)],
                       "box": _box, "ctr": _ctr, "dist": _dist, "_t": t, "_scan": 1})
        stat["스캔 지도프레임"] += len(mp)
    seen_t = set()
    for x in ([] if mp else ev):
        bw = x.get("before_window") or []
        for t in range(int(bw[0]), int(bw[1]) + 1) if len(bw) == 2 else []:
            if t in seen_t: continue
            seen_t.add(t)
            l = next((z for z in live if z["t"] == t), None)
            if l is None: continue
            mp.append({"room": l["room"], "yaw": l["yaw"], "apos": l["apos"],
                       "box": l["box"], "ctr": l["ctr"], "dist": l["dist"], "_t": t})
    g = {"house": hn, "rooms": rooms, "room_types": rtypes, "gt0": gt0, "moves": moves,
         "live": live, "map": mp, "fps": a.fps, "T": len(live),
         "scene_meta": {"polys": polys, "doors": []},
         "_src": e, "_scene": os.path.basename(os.path.dirname(os.path.dirname(e.rstrip("/")))),
         "_newsim2": True}
    # ── 프레임 추출: 사슬은 live/%06d.jpg · map/%04d.jpg 를 읽는다. 이게 없으면 초기맵부터 못 돈다.
    if not a.no_frames:
        _want_live = {int(l["t"]) for l in live}
        _n = _dump(e + "ego.mp4", _want_live, hd + "/live", "%06d.jpg")
        stat["라이브 이미지"] += _n
        if mp and _sd:
            _n2 = _dump(_sd + "ego.mp4", {int(m["_t"]) for m in mp}, hd + "/map", "%04d.jpg", order=[int(m["_t"]) for m in mp])
            stat["지도 이미지"] += _n2
        elif mp:
            _n2 = _dump(e + "ego.mp4", {int(m["_t"]) for m in mp}, hd + "/map", "%04d.jpg", order=[int(m["_t"]) for m in mp])
            stat["지도 이미지"] += _n2
        for k, m in enumerate(mp): m["_k"] = k          # map 파일 색인 = 리스트 순서
    json.dump(g, open(hd + "/gt.json", "w"), ensure_ascii=False)
    stat["집"] += 1; stat["프레임"] += len(live); stat["지도프레임"] += len(mp)
    if a.check:
        # 투영 검증: 가시 물체의 GT 위치가 박스 중심 방향과 맞나 (방위각 오차)
        errs = []
        for l in live[: min(len(live), 60)]:
            for oid in l["vis"]:
                gp = gt0[oid]["pos"]
                for mv in moves:
                    if mv["oid"] == oid and l["t"] > mv["t"]: gp = mv["pos"]
                dx = gp[0] - l["apos"][0]; dz = gp[2] - l["apos"][1]
                bear = math.degrees(math.atan2(dx, dz))
                u = (l["ctr"][oid][0] / 1280.0 - 0.5) * 90.0     # fov 90 근사
                errs.append(abs(((bear - l["yaw"]) - u + 180) % 360 - 180))
        if errs: checks.append((hn, float(np.median(errs)), len(errs)))
print("NEWSIM2_TO_GT_DONE", dict(stat), "→", a.out)
if checks:
    md = float(np.median([c[1] for c in checks]))
    print("투영 검증(방위 오차 중앙, 낮을수록 좌표 규약이 맞음): 전체 중앙 %.1f°" % md)
    for hn, m, n in checks[:6]: print("   %s %.1f° (n=%d)" % (hn, m, n))
