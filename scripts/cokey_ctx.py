#!/usr/bin/env python3
"""공존키(co-located key) 문맥 프레임 — 포즈 없이 "옛 자리를 본 프레임" 을 이동성 작은 이웃 물체 집합으로 찾는다 (§166-100, 2026-09-21).

키   = 기록 자리(rec_pos, 초기맵) 반경 NBR_R m 안의 정적 물체(scene_meta.static) 중 이동성 < MOB_MAX · 타겟 타입과 단어 안 겹침 · exemplar 캐시(hs2_x_)에 있는 것.
가시 = 앵커 프레임 i 에서 exemplar 중앙값-보정 점수 XSc[i,k] ≥ TH 인 키 (≥ KMIN 개).
배치 = 보이는 키들의 (바닥 3D 좌표 → 화면 x) 를 1차식으로 맞춰 자리의 화면 x 를 예측(키 2개면 선분 보간 + 수직거리 벌점). 예측점이 화면 안(여백 MARGIN)이고
       화면 척도(px/m)가 SCALE_MIN 이상일 때만 문맥 프레임. 크롭 = 예측점 주변, 크기 = OBJ_M × 척도.
출력 = anchor_ctx.jsonl 형식(abs_verify_mlx RETR=anchor 가 그대로 읽음): {house, oid, type, anchor_id:"cokey", anchor_room, rel:[0.5,0.5], n_ctx, frames:[[t,x0,y0,x1,y1,sim],…]}
  THOR_ROOT=data/hssd_c3big AX_PREFIX=~/khcache/bench-c3big/cache/hs2_x_ REC_JSONL=~/khcache/bench-c3big/rec_sel.jsonl OUT_JSONL=... python scripts/cokey_ctx.py
진단(GT): live.vis 로 키 가시성 정밀도/재현율, GT 포즈로 "자리를 향함" 대비 문맥 프레임의 정밀도(자리 향한 비율)를 찍는다 — 선택 자체는 GT 를 쓰지 않는다.
"""
import os, json, glob, collections, numpy as np
ROOT = os.environ.get("THOR_ROOT", "data/hssd_c3big"); AXP = os.path.expanduser(os.environ["AX_PREFIX"]); REC = os.path.expanduser(os.environ["REC_JSONL"])
OUT = os.path.expanduser(os.environ.get("OUT_JSONL", "/tmp/cokey_ctx.jsonl")); PRIOR = os.environ.get("PRIOR_JSON", "data/hssd_move.json")
NBR_R = float(os.environ.get("NBR_R", "4.0")); MOB_MAX = float(os.environ.get("MOB_MAX", "0.1")); TH = float(os.environ.get("CK_TH", "0.03")); KMIN = int(os.environ.get("CK_KMIN", "2"))
MARGIN = float(os.environ.get("CK_MARGIN", "0.08")); SCALE_MIN = float(os.environ.get("CK_SCALE_MIN", "40")); OBJ_M = float(os.environ.get("CK_OBJ_M", "0.35")); PERP_MAX = float(os.environ.get("CK_PERP", "1.5"))
FRAME_W = float(os.environ.get("FRAME_W", "768")); MAXF = int(os.environ.get("CK_MAXF", "40"))
MODE = os.environ.get("CK_MODE", "resect"); FX = float(os.environ.get("FRAME_FX", "0")) or FRAME_W / 2.0; RES_MAX = float(os.environ.get("CK_RES", "6.0"))   # hfov 90° 기본
mob = json.load(open(PRIOR)).get("mobility", {})
recs = [json.loads(l) for l in open(REC)]
by_h = collections.defaultdict(list)
for r in recs: by_h[r["house"]].append(r)
st = collections.Counter(); diag = collections.Counter(); prec = []; n_out = 0
def facing(ap, yaw, spot, dmax=4.0, amax=35.0):
    dx, dz = spot[0] - ap[0], spot[1] - ap[1]; d = np.hypot(dx, dz)
    return 0.3 <= d <= dmax and abs((np.degrees(np.arctan2(dx, dz)) - yaw + 180) % 360 - 180) <= amax
fo = open(OUT, "w")
for hn in sorted(by_h):
    fx = AXP + hn + ".npz"
    if not os.path.exists(fx): st["exemplar 캐시 없음"] += len(by_h[hn]); continue
    z = np.load(fx, allow_pickle=True); S = z["s"]; P = z["p"]; ts = z["ts"]; anch = list(z["anch"]); ph, pw = int(z["ph"]), int(z["pw"])
    XSc = S - np.median(S, axis=0, keepdims=True)
    g = json.load(open(os.path.join(ROOT, hn, "gt.json"))); stat = (g.get("scene_meta") or {}).get("static") or {}; live = {m["t"]: m for m in g["live"]}; gt0 = g.get("gt0", {})
    lv = {int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(ROOT, hn, "live", "*.jpg"))}
    for r in by_h[hn]:
        oid, typ, rp = r["oid"], r["type"], r.get("rec_pos")
        if rp is None: st["rec_pos 없음"] += 1; continue
        tw = set(typ.split())
        keys = []
        for k, v in stat.items():
            if k not in anch or not v.get("pos"): continue
            if mob.get(v["type"], 0.0) >= MOB_MAX or (set(v["type"].split()) & tw): continue
            if np.hypot(v["pos"][0] - rp[0], v["pos"][2] - rp[1]) > NBR_R: continue
            keys.append((anch.index(k), k, np.array([v["pos"][0], v["pos"][2]])))
        if len(keys) < KMIN: st["키 <%d" % KMIN] += 1; fo.write(json.dumps(dict(house=hn, oid=oid, type=typ, anchor_id="cokey", anchor_room=r.get("record"), rel=[0.5, 0.5], n_ctx=0, n_keys=len(keys), frames=[])) + "\n"); continue
        spot = np.array(rp, float); frames = []
        for i in range(len(ts)):
            t = int(ts[i])
            if t not in lv: continue
            vis = [(c, k, pos) for c, k, pos in keys if XSc[i, c] >= TH]
            if len(vis) < KMIN: continue
            U = np.array([((P[i, c] % pw) + .5) / pw * FRAME_W for c, _, _ in vis]); V = np.array([((P[i, c] // pw) + .5) / ph * FRAME_W for c, _, _ in vis])
            X = np.array([pos for _, _, pos in vis])
            if len(vis) >= 3 and MODE == "resect":
                # 물체 기반 재국소화(2D): 키의 화면 x → 상대 방위, 키 3D 위치와 함께 (x, z, yaw) 를 푼다. yaw 를 1° 격자로 훑고 각 yaw 에서
                # 키에서 카메라로 향하는 직선들의 최소제곱 교점을 카메라 위치로, 방위 잔차(°)가 RES_MAX 이하일 때만 채택. 그 포즈로 챔피언과 같은 "자리 향함" 검사.
                pb = np.degrees(np.arctan((U - FRAME_W / 2.0) / FX))
                best = None
                for yaw in range(0, 360, 2):
                    b = np.radians(yaw + pb); d = np.c_[np.sin(b), np.cos(b)]          # 카메라→키 방향 (x=sin, z=cos)
                    # 카메라 c 는 각 키 k 에 대해 c = X_k − t_k d_k (t_k>0) ⇒ 직선 최소제곱: Σ (I − d dᵀ)(c − X_k) = 0
                    _M = np.zeros((2, 2)); _v = np.zeros(2)
                    for dk, xk in zip(d, X): Pm = np.eye(2) - np.outer(dk, dk); _M += Pm; _v += Pm @ xk
                    try: _c = np.linalg.solve(_M, _v)
                    except Exception: continue
                    rel = X - _c; dep = (rel * d).sum(1)
                    if np.any(dep < 0.3): continue
                    res = np.abs((np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) - np.degrees(b) + 180) % 360 - 180)
                    _r = float(np.mean(res))
                    if best is None or _r < best[0]: best = (_r, _c, yaw)
                if best is None or best[0] > RES_MAX: diag["재국소화 잔차 초과/실패"] += 1; continue
                _r, _c, yaw = best
                if not facing(_c, yaw, spot, dmax=4.0, amax=35.0): diag["포즈상 자리 안 향함"] += 1; continue
                dxs, dzs = spot[0] - _c[0], spot[1] - _c[1]; bs = (np.degrees(np.arctan2(dxs, dzs)) - yaw + 180) % 360 - 180
                u = float(FRAME_W / 2.0 + np.tan(np.radians(bs)) * FX); scale = float(FX / max(np.hypot(dxs, dzs), 0.3))
            elif len(vis) >= 3:
                A = np.c_[X, np.ones(len(vis))]
                try: coef, *_ = np.linalg.lstsq(A, U, rcond=None); u = float(np.r_[spot, 1.0] @ coef)
                except Exception: continue
                scale = float(np.linalg.norm(coef[:2]))            # px per m (화면 x 방향)
            elif MODE == "resect": diag["키 <3 (재국소화 불가)"] += 1; continue
            else:
                (c1, _, p1), (c2, _, p2) = vis; d = p2 - p1; L = np.linalg.norm(d)
                if L < 0.3: continue
                lam = float(np.dot(spot - p1, d) / (L * L)); perp = float(abs(np.cross(d, spot - p1)) / L)
                if perp > PERP_MAX: continue
                u = float(U[0] + lam * (U[1] - U[0])); scale = float(abs(U[1] - U[0]) / L)
            if scale < SCALE_MIN: diag["척도 미달"] += 1; continue
            if not (MARGIN * FRAME_W <= u <= (1 - MARGIN) * FRAME_W): diag["예측점 화면 밖"] += 1; continue
            v = float(np.mean(V)); h2 = max(48.0, OBJ_M * scale)
            sim = float(np.mean([XSc[i, c] for c, _, _ in vis])) + 0.01 * len(vis)
            frames.append([t, u - 2 * h2, v - 2 * h2, u + 2 * h2, v + 2 * h2, sim, len(vis)])
        # 프레임 상한: 점수 상위 MAXF 를 시간순으로
        frames = sorted(sorted(frames, key=lambda f: -f[5])[:MAXF], key=lambda f: f[0])
        n_out += 1; st["문맥 ≥2" if len(frames) >= 2 else ("문맥 1" if frames else "문맥 0")] += 1
        # 진단: GT 포즈로 자리 향함 대비
        gpos = (gt0.get(oid) or {}).get("pos")
        if gpos and frames:
            fac = [f for f in frames if live.get(f[0], {}).get("apos") and live[f[0]].get("yaw") is not None and facing(live[f[0]]["apos"], live[f[0]]["yaw"], (gpos[0], gpos[2]))]
            prec.append(len(fac) / len(frames))
        fo.write(json.dumps(dict(house=hn, oid=oid, type=typ, anchor_id="cokey", anchor_room=r.get("record"), rel=[0.5, 0.5], n_ctx=len(frames), n_keys=len(keys), frames=[[f[0], round(f[1], 1), round(f[2], 1), round(f[3], 1), round(f[4], 1), round(f[5], 4)] for f in frames])) + "\n")
fo.close()
print("대상 %d · %s · 거른 프레임 %s" % (len(recs), dict(st), dict(diag)))
if prec: print("진단(GT 포즈): 문맥 프레임 중 실제로 GT 자리를 향한 비율 — 중앙 %.2f · 평균 %.2f (n=%d 물체)" % (np.median(prec), np.mean(prec), len(prec)))
print("→", OUT)
