#!/usr/bin/env python3
"""유령 물체 명부: 생성기가 cluttered scene_instance 를 읽고 sim 은 uncluttered 를 로드해, 렌더에 없는 물체가
gt0 에 들어간 집(2026-09-08 §166-25). 집마다 장면을 물체 id 집합으로 맞추고, cluttered 목록의 각 인스턴스를
uncluttered 목록에서 (템플릿, 최근접 좌표 ≤0.05 m) 로 찾아 못 찾으면 유령으로 적는다.
    python scripts/phantom_census.py <DATA_ROOT> [out.json]   (기본 out: <DATA_ROOT>/phantom_ids.json)
평가기는 PHANTOM_JSON=<out.json> 으로 이 물체를 질의에서 뺀다."""
import json, glob, os, sys, csv, collections, math, re
root = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "phantom_ids.json")
HS = os.environ.get("HSSD_ROOT", os.path.expanduser("~/hssd-hab")); HASH = {}
for mf in glob.glob(HS + "/metadata/fpmodels*.csv"):
    for row in csv.DictReader(open(mf, newline="")):
        c = (row.get("main_category") or "").strip()
        if row.get("id") and c: HASH[row["id"]] = c.replace("_", " ").lower()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hab_episode.py")).read()
m = re.search(r"^STRUCT\s*=\s*(\{[^}]*\})", src, re.M); STRUCT = eval(m.group(1)) if m else set()
def labeled(inst):
    for k, oi in enumerate(inst):
        base = oi["template_name"].split("/")[-1]; lab = HASH.get(base)
        if not lab or lab in STRUCT: continue
        yield "%s|%d" % (lab, k), base, [float(x) for x in oi["translation"]]
scenes = {}
for f in glob.glob(HS + "/scenes/*.scene_instance.json"):
    sc = os.path.basename(f).split(".")[0]; fu = HS + "/scenes-uncluttered/%s.scene_instance.json" % sc
    if not os.path.exists(fu): continue
    a = list(labeled(json.load(open(f))["object_instances"])); b = list(labeled(json.load(open(fu))["object_instances"]))
    # 템플릿별로 개수 차이만큼만 유령 — 좌표가 두 파일에서 조금 다른 실물(4%)을 유령으로 잘못 찍지 않도록
    # 거리 임계 없이 가까운 순 greedy 로 짝을 맺고 남는 것을 유령으로 둔다.
    byt = collections.defaultdict(list)
    for _, base, p in b: byt[base].append(p)
    bya = collections.defaultdict(list)
    for oid, base, p in a: bya[base].append((oid, p))
    ph = set()
    for base, items in bya.items():
        cand = list(byt.get(base) or []); pairs = sorted(((math.dist(p, q), i, j) for i, (_, p) in enumerate(items) for j, q in enumerate(cand)))
        used_i, used_j = set(), set()
        for d, i, j in pairs:
            if i in used_i or j in used_j: continue
            used_i.add(i); used_j.add(j)
        ph.update(oid for i, (oid, _) in enumerate(items) if i not in used_i)
    # 수정판 생성기(2026-09-08 이후)는 uncluttered 목록 순서로 id 를 매긴다 → 그 순서의 id 집합도 함께 두고, 집마다 더 잘 맞는 쪽으로 판정한다
    scenes[sc] = ({o for o, _, _ in a}, ph, {o for o, _, _ in b})
res = {}; tot = collections.Counter()
for hd in sorted(glob.glob(os.path.join(root, "house_*"))):
    hn = os.path.basename(hd); gp = os.path.join(os.path.realpath(hd), "gt.json")
    if not os.path.exists(gp): continue
    g = json.load(open(gp)); keys = set(g["gt0"])
    sc, jac, mode = max((((s, len(keys & v[0]) / max(1, len(keys | v[0])), "cluttered")) for s, v in scenes.items()) +
                        [((s, len(keys & v[2]) / max(1, len(keys | v[2])), "uncluttered")) for s, v in scenes.items()], key=lambda x: x[1])
    if mode == "uncluttered": phset = set()          # 생성기가 uncluttered 목록을 읽었다 = 전부 렌더됨 → 유령 없음
    else: phset = scenes[sc][1]
    ph = sorted(phset & keys); mv = [x["oid"] for x in g["moves"]]; mvph = [o for o in mv if o in phset]
    # 유령 이동의 '숙주': 생성기가 유령 대신 같은 좌표(핸들 대체)의 실물 가구를 옮겼다 → 그 실물의 GT(안 움직임)도 틀리므로 같이 뺀다
    hosts = sorted({o for p_ in mvph for o, v in g["gt0"].items() if o not in phset and v["pos"] == g["gt0"][p_]["pos"]})
    res[hn] = dict(scene=sc, jaccard=round(jac, 3), n_obj=len(keys), phantom=ph, moves_phantom=mvph, hosts=hosts, id_order=mode)
    tot["hosts"] += len(hosts)
    tot["obj"] += len(keys); tot["ph"] += len(ph); tot["mv"] += len(mv); tot["mvph"] += len(res[hn]["moves_phantom"])
    if jac < 0.9: print("⚠ %s 장면 매칭 낮음 %s %.2f" % (hn, sc, jac))
json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
print("%s: 집 %d · 물체 %d 중 유령 %d (%.1f%%) · 이동 %d 중 유령 이동 %d (숙주 실물 %d) → %s" % (root, len(res), tot["obj"], tot["ph"], 100 * tot["ph"] / max(1, tot["obj"]), tot["mv"], tot["mvph"], tot["hosts"], out))
