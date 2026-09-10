#!/usr/bin/env python3
"""새 시뮬레이터(AIUE, maru-detector-tracker-gt/v1) 출력 → 우리 gt.json + map/·live/ 프레임 (2026-09-09).
입력 구조: <root>/<MAP>/{01_unchanged,02_relocated,03_absent,11_ego_coverage_scan}/<episode>/{ego.mp4, annotations.jsonl,
observed_graph_updates.jsonl, ground_truth_updates.jsonl, scene_graph.json, entities.json, scenario.json}
  python scripts/newsim_to_gt.py incoming/new_sim data/newsim [--stride 15]
좌표: Unreal 왼손(X앞·Y오른쪽·Z위, cm) → 우리(x=-Y, y=Z, z=-X, m; 평가기의 x-미러 프레임) · yaw_ours = yaw_unreal + 180 (b=atan2(dx,dz) 규약).
미러(좌우)는 GT 박스 중심과 투영을 비교해 집마다 자동 판정해 _mirror 에 적는다."""
import os, sys, json, re, glob, math, argparse, collections, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--stride", type=int, default=15)
ap.add_argument("--maps", default=""); ap.add_argument("--no-frames", action="store_true"); a = ap.parse_args()
STRUCT_PAT = re.compile(r"(building|door|window|wall|floor|ceiling|socket|sill|skirting|radiator|enviro|decoration|frame$|hook|switch|handle|curtain|blind|stair|beam|pipe|vent|panel|baseboard|molding|light_switch)", re.I)
ROOM_ALIAS = {"living_room": "living room", "bed_room": "bedroom", "bedroom": "bedroom", "kitchen": "kitchen", "hallway": "hallway", "utility_room": "utilityroom",
              "bathroom": "bathroom", "dining_room": "dining room", "office": "office", "corridor": "hallway", "toilet": "toilet"}
def rtype(rid):
    k = re.sub(r"\d+$", "", rid.strip().lower())          # Balcony1 → balcony
    return ROOM_ALIAS.get(k, ROOM_ALIAS.get(rid, k.replace("_", " ").strip()))
ROOM_WORDS = {"kitchen", "bedroom", "bed", "living", "bath", "bathroom", "utility", "hall", "hallway", "balcony", "room", "office", "dining"}
JUNK = {"sm", "ai", "am", "vol", "agent", "wood", "static", "mesh", "actor", "prop", "asset", "new", "old", "big", "small"}
def norm_type(cls):
    """자산 이름(SM_AI_vol8_01_kitchen_jug_3 · AI62_005_book_I4 · am253_021_elephant) → 범주 명사(jug · book · elephant).
    벤더 접두·번호·코드·방 단어를 버리고 마지막 명사(+앞 형용사 1개)만 남긴다. 남는 게 없으면 None(제외)."""
    toks = [t for t in re.split(r"[\s_]+", cls.strip().lower()) if t]
    keep = [t for t in toks if t.isalpha() and len(t) >= 3 and not re.match(r"^(vol|am|ai|sm)\d*$", t) and t not in JUNK]
    keep = [t for t in keep if t not in ROOM_WORDS]
    if not keep: return None
    if any(t in ("agent", "character", "cinecamera", "ego", "mover", "firstperson") for t in toks): return None
    SING = {"books": "book", "plates": "plate", "mugs": "mug", "cups": "cup", "jars": "jar", "bowls": "bowl", "chairs": "chair", "pillows": "pillow", "magazines": "magazine", "flowers": "flower", "tiles": "tile", "candles": "candle", "figures": "figure", "paintings": "painting", "pictures": "picture", "glasses": "glass"}
    keep = [SING.get(t, t) for t in keep]
    return " ".join(keep[-2:]) if len(keep) >= 2 and len(keep[-2]) <= 9 else keep[-1]
def ours(p):   # cm (X,Y,Z) → m (x=-Y, y=Z, z=-X). 평가기 좌표계는 HSSD gt.json 과 같은 x-미러(왼손) 프레임: 방위 b=atan2(dx,dz), fwd=(sin θ, cos θ), rgt=(cos θ, -sin θ)
    return [-p[1] / 100.0, p[2] / 100.0, -p[0] / 100.0]
def yaw_ours(yaw_u): return (yaw_u + 180.0) % 360.0    # UE fwd=(cos ψ, sin ψ) → 우리 (x,z)=(-sin ψ, -cos ψ) = (sin θ, cos θ) with θ=ψ+180 (2026-09-09 실측: 받침 가구 방위 잔차 -170° 로 확인)
W, H, FOV = 720, 540, 90.0; FX = (W / 2) / math.tan(math.radians(FOV / 2)); CX, CY = W / 2, H / 2
def project(cam, yaw, pitch, p, mirror):
    """우리 좌표 카메라(cam[x,y,z], yaw°, pitch°) → 픽셀 (u, v) 와 깊이. 수평 방위만 정확하면 된다(pitch 는 v 에만)."""
    d = np.array(p) - np.array(cam); th = math.radians(yaw); pt = math.radians(pitch)
    fwd = np.array([math.sin(th), 0, math.cos(th)]); rgt = np.array([math.cos(th), 0, -math.sin(th)]); up = np.array([0, 1.0, 0])   # 평가기 규약(build_initmap pbx)
    # pitch 적용(아래를 보면 fwd 가 내려감)
    fwd2 = fwd * math.cos(pt) + up * math.sin(pt); up2 = up * math.cos(pt) - fwd * math.sin(pt)
    zc = float(d @ fwd2); xr = float(d @ rgt); yu = float(d @ up2)
    if zc <= 0.05: return None
    u = CX + FX * xr / zc * (-1 if mirror else 1); v = CY - FX * yu / zc
    return u, v, zc
def load_jsonl(p): return [json.loads(l) for l in open(p)]
def episode_dir(sess_dir): return next(os.path.join(sess_dir, d) for d in os.listdir(sess_dir) if os.path.isdir(os.path.join(sess_dir, d)))
def extract(mp4, idxs, outdir, pat, n_ann=None):
    """주석 행 idxs 에 해당하는 비디오 프레임을 뽑는다. 인코더가 프레임을 떨어뜨려 비디오가 주석보다 짧으면(vol8_02 스캔 7332/9000, 2026-09-10 발견:
    인덱스로 뽑으면 100 s 이후 GT 박스가 엉뚱한 곳에 찍힘) 주석 행 i ↔ 비디오 프레임 round(i·n_video/n_ann) 로 선형 재사상한다(육안 검증: 소파·식탁 정렬)."""
    import cv2
    os.makedirs(outdir, exist_ok=True); c = cv2.VideoCapture(mp4); nv = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); scale = 1.0
    if n_ann and nv > 0 and nv < n_ann - 1:
        scale = nv / float(n_ann); print("   ⚠️ %s: 비디오 %d장 < 주석 %d행 → 프레임 선형 재사상 ×%.4f" % (os.path.relpath(mp4), nv, n_ann, scale), flush=True)
    want = {}; va = os.path.join(os.path.dirname(mp4), "video_align.json")
    if scale != 1.0 and os.path.exists(va):                    # newsim_align_video.py 의 DP 정렬(주석 행 → 비디오 프레임)이 있으면 그것을 쓴다
        m = json.load(open(va))["ann2video"]; print("   video_align.json 사용(DP 정렬)", flush=True)
        for i, k in enumerate(idxs): want.setdefault(int(m[k]), i)
    else:
        if scale != 1.0: print("   ⚠️ video_align.json 없음 → 선형 재사상(드롭이 불균일하면 틀린다; scripts/newsim_align_video.py 를 먼저 돌릴 것)", flush=True)
        for i, k in enumerate(idxs): want.setdefault(int(round(k * scale)), i)
    k = 0; n = 0
    while True:
        ok = c.grab()
        if not ok: break
        if k in want:
            ok, fr = c.retrieve()
            if ok: cv2.imwrite(os.path.join(outdir, pat % want[k]), fr, [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
        k += 1
        if k > max(want): break
    return n
maps = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)) and not d.startswith("."))
if a.maps: maps = [m for m in maps if m in a.maps.split(",")]
os.makedirs(a.out, exist_ok=True); summary = []
for mp in maps:
    scan_sess = os.path.join(a.root, mp, "11_ego_coverage_scan"); scan = episode_dir(scan_sess)
    sg = json.load(open(os.path.join(scan, "scene_graph.json")))
    rooms = {r["id"]: r for r in sg["rooms"]}
    polys = {}
    for rid, r in rooms.items():
        (x0, y0, _), (x1, y1, _) = r["rect_xy"]["min"], r["rect_xy"]["max"]
        cs = [ours([x0, y0, 0]), ours([x1, y0, 0]), ours([x1, y1, 0]), ours([x0, y1, 0])]; polys[rid] = [[round(c[0], 3), round(c[2], 3)] for c in cs]
    room_types = {rid: rtype(rid) for rid in rooms}
    def room_of_xy(X, Y):
        for rid, r in rooms.items():
            if r["rect_xy"]["min"][0] <= X <= r["rect_xy"]["max"][0] and r["rect_xy"]["min"][1] <= Y <= r["rect_xy"]["max"][1]: return rid
        return min(rooms, key=lambda rid: math.hypot(X - (rooms[rid]["rect_xy"]["min"][0] + rooms[rid]["rect_xy"]["max"][0]) / 2, Y - (rooms[rid]["rect_xy"]["min"][1] + rooms[rid]["rect_xy"]["max"][1]) / 2))
    objs = {}
    for o in sg["objects"]:
        if o.get("is_structure") or o.get("is_opening"): continue
        cls = o.get("class") or ""; t = norm_type(cls)
        if not t or STRUCT_PAT.search(cls): continue
        loc = o["transform"]["location"]; rid = o.get("room_id") or room_of_xy(loc[0], loc[1])
        objs[o["id"]] = dict(type=t, room=rid, pos=ours(loc), loc_u=loc)
    # 스캔 프레임
    scan_cams = load_jsonl(os.path.join(scan, "observed_graph_updates.jsonl")); scan_ann = load_jsonl(os.path.join(scan, "annotations.jsonl"))
    ent_scan = {e["actor_name"].strip(): e["instance_id"] for e in json.load(open(os.path.join(scan, "entities.json")))["entities"]}
    inst2oid_scan = {v: k for k, v in ent_scan.items()}
    idxs = list(range(0, len(scan_cams), a.stride))
    va_path = os.path.join(scan, "video_align.json")
    if os.path.exists(va_path):                            # 인코더 정지(stall)로 영상에 없는 주석 행은 지도에서 뺀다 (vol8_02: 834행 × 2회 = 1,668행 손실)
        va = json.load(open(va_path)); v2a = va["video2ann"]; cov = set()
        for r in v2a: cov.update((r - 2, r - 1, r, r + 1, r + 2))
        runs = []; prev = v2a[0]
        for r in v2a[1:]:
            if r - prev - 1 >= 30: runs.append((prev + 1, r - 1))
            prev = r
        n0 = len(idxs); idxs = [k for k in idxs if k in cov]
        print("   %s: 영상 %d장/주석 %d행 · 정지 구간 %s · 지도 프레임 %d → %d (영상에 없는 행 제외)" % (os.path.basename(scan), va["n_video"], va["n_ann"], ["행 %d~%d(%d)" % (a0, b0, b0 - a0 + 1) for a0, b0 in runs], n0, len(idxs)), flush=True)
    def frame_rec(cam, ann, inst2oid, positions, kind):
        loc = cam["location"]; pyr = cam["rotation_pyr_deg"]; cpos = ours(loc); yaw = yaw_ours(pyr[1]); pitch = float(pyr[0])
        rec = dict(room=room_of_xy(loc[0], loc[1]), yaw=round(yaw, 2), pitch=round(pitch, 2), apos=[round(cpos[0], 3), round(cpos[2], 3)], ctr={}, dist={}, box={})
        for ob in ann["objects"]:
            oid = inst2oid.get(ob["instance_id"])
            if oid not in positions: continue
            b = ob["bbox"]; rec["box"][oid] = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]; rec["ctr"][oid] = [round((b[0] + b[2]) / 2, 1), round((b[1] + b[3]) / 2, 1)]
            p = positions[oid]; rec["dist"][oid] = round(math.hypot(p[0] - cpos[0], p[2] - cpos[2]), 2)
        return rec, cpos, yaw, pitch
    positions0 = {k: v["pos"] for k, v in objs.items()}
    mapping = []; mir_err = [0.0, 0.0]; mir_n = 0
    for i in idxs:
        rec, cpos, yaw, pitch = frame_rec(scan_cams[i]["cameras"][0], scan_ann[i], inst2oid_scan, positions0, "map")
        rec["room_walk"] = rec["room"]; mapping.append(rec)
        for oid, c in list(rec["ctr"].items())[:20]:      # 미러 판정 표본
            for m_ in (0, 1):
                pr = project(cpos, yaw, pitch, positions0[oid], m_)
                if pr: mir_err[m_] += abs(pr[0] - c[0]); 
            mir_n += 1
    mirror = int(mir_err[1] < mir_err[0]); 
    for s in ("01_unchanged", "02_relocated", "03_absent"):
        sd = os.path.join(a.root, mp, s)
        if not os.path.isdir(sd): continue
        ep = episode_dir(sd); sc = json.load(open(os.path.join(ep, "scenario.json"))); ev = sc["evidence"]
        cams = load_jsonl(os.path.join(ep, "observed_graph_updates.jsonl")); ann = load_jsonl(os.path.join(ep, "annotations.jsonl")); gtu = load_jsonl(os.path.join(ep, "ground_truth_updates.jsonl"))
        ent = {e["actor_name"].strip(): e["instance_id"] for e in json.load(open(os.path.join(ep, "entities.json")))["entities"]}; inst2oid = {v: k for k, v in ent.items()}
        oid = ev["object"]; moves = []; positions = dict(positions0)
        if ev.get("destination_room"):
            t_arr = next((r["t"] for r in cams + gtu for u in r.get("updates", []) if (u.get("s") == oid or u.get("id") == oid) and u.get("op") == "relationship" and u.get("p") == "in" and u.get("o") == ev["destination_room"] and r["t"] > 0), None)
            t_leave = next((r["t"] for r in gtu for u in r.get("updates", []) if u.get("s") == oid and u.get("op") == "relationship_remove" and u.get("p") == "in"), None)
            t_mv = t_arr if t_arr is not None else (t_leave if t_leave is not None else ev["after_window"][0])
            dpos = ours(ev["destination_position"])
            if oid not in objs: objs[oid] = dict(type=norm_type(oid) or "object", room=ev["source_room"], pos=ours(ev["source_position"]))
            moves.append(dict(t=int(t_mv), oid=oid, frm=ev["source_room"], to=ev["destination_room"], intended=ev["destination_room"], pos=[round(v, 3) for v in dpos],
                              witness=False, supported=True, role=("c3" if s.startswith("03") else "c2"), t_leave=t_leave, t_arrive=t_arr))
        live = []
        for t in range(len(cams)):
            if moves and t >= moves[0]["t"]: positions[oid] = moves[0]["pos"]
            rec, cpos, yaw, pitch = frame_rec(cams[t]["cameras"][0], ann[t], inst2oid, positions, "live")
            vis = sorted(rec["ctr"]); anch = {o: rec["ctr"][o] for o in vis if o != oid}
            live.append(dict(t=t, room=rec["room"], vis=vis, ctr=rec["ctr"], anch=anch, dist=rec["dist"], apos=rec["apos"], yaw=rec["yaw"], pitch=rec["pitch"], visible_regions=cams[t]["cameras"][0].get("visible_regions", [])))
        hn = "house_%s_%s" % (mp.lower().replace("aiue_", ""), s[:2]); hd = os.path.join(a.out, hn); os.makedirs(hd, exist_ok=True)
        gt0 = {k: dict(type=v["type"], room=v["room"], pos=[round(x, 3) for x in v["pos"]]) for k, v in objs.items()}
        static = {k: v for k, v in gt0.items() if k not in {m["oid"] for m in moves}}
        payload = dict(house=hn, source=dict(sim="AIUE", map=mp, session=s, scan=os.path.relpath(scan, a.root), scenario=ev), rooms=list(rooms), room_types=room_types,
                       gt0=gt0, moves=moves, live=live, map=mapping, fps=1.0, T=len(live),
                       scene_meta=dict(polys=polys, static=static, doors=[], intrinsics=dict(W=W, H=H, fx=round(FX, 3), cx=CX, cy=CY, fov_deg=FOV), units="m", coord="x=Y_u,y=Z_u,z=-X_u"),
                       _mirror_fixed=True, _remap=True, _map_room_by_pos=True, _mirror=mirror, _mirror_err=[round(mir_err[0] / max(1, mir_n), 1), round(mir_err[1] / max(1, mir_n), 1)])
        json.dump(payload, open(os.path.join(hd, "gt.json"), "w"), ensure_ascii=False)
        if not a.no_frames:
            n_l = extract(os.path.join(ep, "ego.mp4"), list(range(len(cams))), os.path.join(hd, "live"), "%06d.jpg", n_ann=len(cams))
            n_m = extract(os.path.join(scan, "ego.mp4"), idxs, os.path.join(hd, "map"), "%04d.jpg", n_ann=len(scan_cams))
        else:
            n_l = len(glob.glob(os.path.join(hd, "live", "*.jpg"))); n_m = len(glob.glob(os.path.join(hd, "map", "*.jpg")))
        # mp4 의 실제 디코딩 프레임 수가 메타보다 적을 수 있다(vol8_02 스캔: 489 예상 → 실제 추출 수만큼) → gt 의 map/live 를 추출된 장수로 자른다
        if 0 <= n_m < len(payload["map"]): payload["map"] = payload["map"][:n_m]
        if 0 <= n_l < len(payload["live"]): payload["live"] = payload["live"][:n_l]; payload["T"] = n_l
        json.dump(payload, open(os.path.join(hd, "gt.json"), "w"), ensure_ascii=False)
        cnt = collections.Counter(v["type"] for v in gt0.values()); uniq = sum(1 for v in gt0.values() if cnt[v["type"]] == 1)
        if moves: print("   증거 물체 %s → 타입 '%s' · 집 안 같은 타입 %d개%s" % (oid, gt0[oid]["type"], cnt[gt0[oid]["type"]], "" if cnt[gt0[oid]["type"]] == 1 else "  ⚠️ 타입 유일 아님 → 현재 평가기는 질의에서 제외"), flush=True)
        summary.append((hn, len(gt0), uniq, len(moves), n_m, n_l, mirror, payload["_mirror_err"]))
        print("%s: 물체 %d(타입 유일 %d) · 이동 %s · 지도 %d장 · 라이브 %d장 · 미러 %d (오차 정/역 %s px) · 방 %s" % (hn, len(gt0), uniq, [(m["oid"], m["frm"], "→", m["to"], "t=%d" % m["t"]) for m in moves], n_m, n_l, mirror, payload["_mirror_err"], list(rooms)), flush=True)
print("NEWSIM_TO_GT_DONE · 집 %d" % len(summary))
