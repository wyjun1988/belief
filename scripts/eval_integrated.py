#!/usr/bin/env python3
"""**통합 4경우 파이프라인 — 프로젝트의 최종 한 숫자.**

    THOR_ROOT=data/thor3 python scripts/eval_integrated.py

  0. 최신 관측에서 물체 발견(검색 확신 ≥ θf)  → 검색이 답한 방
  2. 씬그래프 방을 재방문했고 부재 증거 ≥ θa  → belief (씬그래프 방 제외 argmax)
  1/3. 그 외                                 → 씬그래프 방

⚠️ **시간 오라클 없음.** 이동 시각(t0)을 쓰지 않는다 — 전체 프레임에서 답한다.
⚠️ 씬그래프 초기 방은 아직 GT 다(초기 맵 구축은 미검증 P0).
비교선: 정지 지도(항상 씬그래프 방) · 검색 단독 · belief 단독.
θf, θa 를 훑어 이득 구간이 존재하는지 본다. 점수는 물체별 자기보정(중앙값 차감).
"""
import json, glob, os, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor3")
A3P = os.environ.get("A3_PREFIX", "/tmp/a3_")
SG_SRC = os.environ.get("SG_SRC", "gt")   # gt | owl — 씬그래프 초기 방의 출처
AXP = os.environ.get("AX_PREFIX", "")     # 있으면 §78 결합(타입가방+개체앵커) 국소화
QCP = os.environ.get("QC_PREFIX", "/tmp/qc_")
K = 10
PR = json.load(open("data/thor_prior.json"))


def ci(a):
    a = np.asarray(a, float); n = len(a)
    b = [a[np.random.randint(0, n, n)].mean() for _ in range(2000)]
    return a.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


rows = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + "%s.npz" % hn, QCP + "%s.npz" % hn
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    ZX = None
    if AXP and os.path.exists(AXP + hn + ".npz"):
        ZX = np.load(AXP + hn + ".npz", allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    g = json.load(open(hd + "/gt.json")); sm = g.get("scene_meta")
    if not sm: continue
    sg_det = {}
    if SG_SRC == "owl":
        imf = os.path.join(os.path.realpath(hd), "initmap_owl.json")
        if not os.path.exists(imf): continue
        best_t = {}
        for i2 in json.load(open(imf)):
            if i2["w"] > best_t.get(i2["type"], (0,))[0]:
                best_t[i2["type"]] = (i2["w"], i2["room"])
        sg_det = {t: r for t, (w, r) in best_t.items()}
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
    if ZX is not None:
        anames = list(ZX["anch"])
        aroom_i = {k: sm["static"][a]["room"] for k, a in enumerate(anames)
                   if a in sm["static"]}
        XS = ZX["s"][:, list(aroom_i)]
        Xrooms = [aroom_i[k] for k in aroom_i]
        XSc = XS - np.median(XS, axis=0, keepdims=True)
        XPp = ZX["p"][:, list(aroom_i)]
        xy_ = np.stack([XPp // pw, XPp % pw], -1)
    arm = np.array([live[t]["room"] for t in ts])
    _rp = os.path.expanduser(os.environ.get("ROOM_PREFIX", ""))
    if _rp:
        # ⚠️ 방 라벨을 CLIP 노드+Viterbi 예측으로 통째 교체 — GT 방 잔재 제거.
        _rz = np.load(_rp + hn + ".npz", allow_pickle=True)
        assert list(_rz["ts"]) == list(ts), "프레임 정렬 불일치"
        arm = _rz["room"]
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]
        if SG_SRC == "owl":
            sg = sg_det.get(v0["type"])
            if sg is None: continue          # 초기맵이 놓친 타겟 — 커버리지로 보고
        else:
            sg = v0["room"]
        TS = QS[:, j] + STx[:, j]
        base = float(np.median(TS))
        top = np.argsort(-TS)[:K]
        conf = float(TS[top].mean() - base)               # 검색 확신 (자기보정)
        # 검색이 답하는 방: 점수가중 앵커조합 + 이웃제한 (0.617 레시피, 오라클 없음)
        acc = {r: 0.0 for r in rids}
        for i in top:
            w = max(0.0, float(TS[i]) - base)
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                ww = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): sc[r] += ww
            nb = {arm[i]} | adj.get(arm[i], set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            if ZX is not None:
                # §78 결합 — 개체 앵커(방이 하나로 정해짐)의 표를 더한다 (θ=0.15)
                on = np.where(XSc[i] >= 0.15)[0]
                si_ = {r: 0.0 for r in rids}
                for k2 in on:
                    d = np.hypot(xy_[i, k2, 0]-cy, xy_[i, k2, 1]-cx)
                    si_[Xrooms[k2]] += float(XSc[i, k2]) / (1.0 + d/6.0)
                t2 = sum(si_.values())
                if t2 > 0:
                    for r in rids: sc[r] = sc[r]/(sum(sc.values())+1e-9) + si_[r]/t2
            tot = sum(sc.values()) + 1e-9
            for r in rids: acc[r] += w * sc[r] / tot
        find_room = max(acc, key=acc.get)
        # 부재 증거: **앵커 게이팅판** (849표본에서 AUC 0.781 로 확정된 기제).
        # 초기 구간에서 물체가 잘 보인 프레임의 정적 점수 = 자리 서명.
        # 이후 프레임 중 **그 자리가 화면에 있는 것만** 골라 점수 하락을 잰다 —
        # "안 보인다" 와 "그 자리를 안 봤다" 를 가른다(조건③). 시간 오라클 없음.
        inr = np.where(arm == sg)[0]
        drop = None
        if len(inr) >= 9:
            e = inr[:len(inr)//3]; l = inr[-len(inr)//3:]
            AS = S[:, nT:]
            k2 = max(3, len(e)//3)
            sig = AS[e[np.argsort(-TS[e])[:k2]]].mean(0)
            sig /= (np.linalg.norm(sig) + 1e-9)
            pe = AS[e] @ sig / (np.linalg.norm(AS[e], axis=1) + 1e-9)
            pl = AS[l] @ sig / (np.linalg.norm(AS[l], axis=1) + 1e-9)
            thp = np.quantile(pe, .7)
            ge = e[pe >= thp]; gl = l[pl >= thp]
            if len(ge) >= 3 and len(gl) >= 3:
                drop = float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9))
        # belief: 씬그래프 방 제외 argmax
        bel = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]], 1), r)
                   for r in rids if r != sg))[1]
        bel0 = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]], 1), r)
                    for r in rids))[1]
        rows.append(dict(sg=sg, tgt=tgt, moved=bool(mv), conf=conf, find=find_room,
                         drop=drop, bel=bel, bel0=bel0))

n = len(rows); mvd = [r for r in rows if r["moved"]]
print("=== 통합 4경우 · 타겟 %d개 (이동 %d) · 시간 오라클 없음 ===" % (n, len(mvd)))
for nm, f in (("정지 지도 (항상 씬그래프 방)", lambda r: r["sg"]),
              ("검색 단독", lambda r: r["find"]),
              ("belief 단독", lambda r: r["bel0"])):
    m, lo, hi = ci([f(r) == r["tgt"] for r in rows])
    mm = np.mean([f(r) == r["tgt"] for r in mvd]) if mvd else 0
    print("  %-26s 전체 %.3f [%.3f %.3f] · 이동만 %.3f" % (nm, m, lo, hi, mm))

confs = sorted(r["conf"] for r in rows)
drops = [r["drop"] for r in rows if r["drop"] is not None]
print("\n  θf\\θa   " + "   ".join("%4.2f" % q for q in (1.01, .95, .9, .8)))
best = (0, None)
for qf in (1.01, .95, .9, .8, .7, .5):
    tf = confs[min(int(len(confs)*qf), len(confs)-1)] if qf <= 1 else 1e9
    line = []
    for qa in (1.01, .95, .9, .8):
        ta = np.quantile(drops, qa) if qa <= 1 and drops else 1e9
        def ans(r):
            if r["conf"] >= tf: return r["find"]                 # 경우 0
            if r["drop"] is not None and r["drop"] >= ta: return r["bel"]   # 경우 2
            return r["sg"]                                       # 경우 1/3
        a = np.mean([ans(r) == r["tgt"] for r in rows])
        line.append(a)
        if a > best[0]: best = (a, (qf, qa, tf, ta))
    print("  %5.2f   " % qf + "   ".join("%.3f" % x for x in line))

# ── 합의 게이트: 두 약한 신호가 **서로 맞장구칠 때만** 뒤집는다 ──
# 검색이 다른 방을 가리키고(find≠sg) + 그 방에서 부재 하락도 보이면 = "옮겨졌다" 는
# 일관된 이야기. 한쪽 신호만으로 뒤집는 것보다 오탐이 줄어야 한다.
print("\n  ── 합의 게이트: find≠sg AND 부재 AND 확신 ──")
for qf2 in (0.8, 0.5, 0.0):
    tf2 = confs[min(int(len(confs)*qf2), len(confs)-1)] if qf2 > 0 else -1e9
    for qa2 in (0.8, 0.5):
        ta2 = np.quantile(drops, qa2) if drops else 1e9
        def ans2(r, use_bel):
            fire = (r["find"] != r["sg"] and r["conf"] >= tf2
                    and r["drop"] is not None and r["drop"] >= ta2)
            if fire: return r["bel"] if use_bel else r["find"]
            return r["sg"]
        af = np.mean([ans2(r, False) == r["tgt"] for r in rows])
        ab = np.mean([ans2(r, True) == r["tgt"] for r in rows])
        nf = sum(r["find"] != r["sg"] and r["conf"] >= tf2
                 and r["drop"] is not None and r["drop"] >= ta2 for r in rows)
        mf = np.mean([ans2(r, False) == r["tgt"] for r in mvd])
        print("  θf=q%.1f θa=q%.1f  발동 %2d · 답=검색 **%.3f** (이동만 %.3f) · 답=belief %.3f"
              % (qf2, qa2, nf, af, mf, ab))

a, (qf, qa, tf, ta) = best
def ans(r):
    if r["conf"] >= tf: return r["find"]
    if r["drop"] is not None and r["drop"] >= ta: return r["bel"]
    return r["sg"]
m, lo, hi = ci([ans(r) == r["tgt"] for r in rows])
c0 = sum(r["conf"] >= tf for r in rows)
c2 = sum(r["conf"] < tf and r["drop"] is not None and r["drop"] >= ta for r in rows)
mm = np.mean([ans(r) == r["tgt"] for r in mvd]) if mvd else 0
print("\n  최적 (θf=q%.2f, θa=q%.2f): **%.3f** [%.3f %.3f]" % (qf, qa, m, lo, hi))
print("  경우0 발동 %d · 경우2 발동 %d · 이동만 정답률 **%.3f** (정지 지도는 %.3f)"
      % (c0, c2, mm, np.mean([r["sg"] == r["tgt"] for r in mvd])))
