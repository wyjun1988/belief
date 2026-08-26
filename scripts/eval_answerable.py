#!/usr/bin/env python3
"""**답 가능성 계층별 채점** — 답을 맞출 수 있는 문제에서 얼마나 맞추는가.

    THOR_ROOT=data/thor4 python scripts/eval_answerable.py

균등 채점은 "원리적으로 답 불가능한 문제" 를 섞어 시스템 실력을 가린다. 관측
기록으로 계층을 가른다 (§51 의 --only-seen 원칙을 통합 파이프라인에 적용):

  T1 발견 가능      이동했고 새 방에서 3프레임+ 목격됨 → 검색이 답할 수 있다
  T2 부재추론 가능   이동했고 원래 방 재방문(목격은 없음) → 부재+belief 경로만
  T3 관측 불가      이동했고 아무 관측 없음 → 어떤 시스템도 belief 뿐
  T4 기록 유지      안 움직임 → 기록만으로 정답 (재확인 여부로 세분)
"""
import json, glob, os, numpy as np
from collections import Counter

ROOT = os.environ.get("THOR_ROOT", "data/thor4")
A3P = os.environ.get("A3_PREFIX", "/tmp/h4/cache/a3_")
QCP = os.environ.get("QC_PREFIX", "/tmp/h4/cache/qc_")
AXP = os.environ.get("AX_PREFIX", "/tmp/h4/cache/ax_")
PR = json.load(open("data/thor_prior.json"))
# 조건①′: 혼동 타입(실측 FP 해부에서 50회+ 쌍)이 같은 집에 있으면 타겟에서 제외.
# 사용자 지정 — "노트북처럼 비슷한 물체나 중복 물체가 없는 것"이 우리 시나리오 타겟.
CLEAN = os.environ.get("CLEAN", "0") == "1"
CONF = {}
if CLEAN:
    for a, b in json.load(open("data/thor_confusable.json")):
        CONF.setdefault(a, set()).add(b); CONF.setdefault(b, set()).add(a)

rows = []
for hd in sorted(glob.glob(ROOT + "/house_*")):
    hn = os.path.basename(os.path.realpath(hd))
    fa, fq = A3P + hn + ".npz", QCP + hn + ".npz"
    if not (os.path.exists(fa) and os.path.exists(fq)): continue
    za = np.load(fa, allow_pickle=True); zq = np.load(fq, allow_pickle=True)
    S, P, ph, pw, ts = za["s"], za["p"], int(za["ph"]), int(za["pw"]), za["ts"]
    vocab, nT = list(za["vocab"]), int(za["nT"])
    QT, QS, STx = list(zq["tg"]), zq["si"], zq["st"]
    ZX = np.load(AXP + hn + ".npz", allow_pickle=True) if os.path.exists(AXP + hn + ".npz") else None
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
    arm = np.array([live[t]["room"] for t in ts])
    if ZX is not None:
        an_ = list(ZX["anch"])
        ai = {k: sm["static"][a]["room"] for k, a in enumerate(an_) if a in sm["static"]}
        XS = ZX["s"][:, list(ai)]; Xr = [ai[k] for k in ai]
        XSc = XS - np.median(XS, axis=0, keepdims=True)
        XPp = ZX["p"][:, list(ai)]
        xy_ = np.stack([XPp // pw, XPp % pw], -1)
    AS = S[:, nT:]
    moves = sorted(g["moves"], key=lambda m: m["t"])
    cnt = Counter(v["type"] for v in g["gt0"].values())
    for j, oid in enumerate(QT):
        v0 = g["gt0"][oid]
        if not v0["room"] or cnt[v0["type"]] > 1 or v0["type"] not in vocab: continue
        if CLEAN:
            house_types = set(cnt) | {v["type"] for v in sm["static"].values()}
            if CONF.get(v0["type"], set()) & house_types: continue
        ti = vocab.index(v0["type"])
        mv = [x for x in moves if x["oid"] == oid]
        tgt = mv[-1]["to"] if mv else v0["room"]; sg = v0["room"]
        t0 = mv[-1]["t"] if mv else 0
        seen_new = sum(oid in live[t].get("vis", []) for t in ts if t > t0) >= 3 if mv else False
        revis = bool((arm[ts > t0] == sg).any()) if mv else bool((arm[len(arm)//2:] == sg).any())
        tier = ("T1" if mv and seen_new else "T2" if mv and revis
                else "T3" if mv else "T4r" if revis else "T4u")
        TS = QS[:, j] + STx[:, j]
        vis = np.array([oid in live[t].get("vis", []) for t in ts])
        base = float(np.median(TS)); top = np.argsort(-TS)[:10]
        acc = {r: 0.0 for r in rids}
        for i in top:
            w = max(0.0, float(TS[i]) - base)
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for c in range(nT, len(vocab)):
                t = vocab[c]
                if t not in idf or S[i, c] < .05: continue
                d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                w2 = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                for r in rids:
                    if t in rtypes.get(r, ()): sc[r] += w2
            if ZX is not None:
                on = np.where(XSc[i] >= 0.15)[0]
                si_ = {r: 0.0 for r in rids}
                for k2 in on:
                    d = np.hypot(xy_[i, k2, 0]-cy, xy_[i, k2, 1]-cx)
                    si_[Xr[k2]] += float(XSc[i, k2]) / (1.0 + d/6.0)
                t2 = sum(si_.values())
                if t2 > 0:
                    for r in rids: sc[r] = sc[r]/(sum(sc.values())+1e-9) + si_[r]/t2
            nb = {arm[i]} | adj.get(arm[i], set())
            for r in rids:
                if r not in nb: sc[r] *= .25
            t3 = sum(sc.values()) + 1e-9
            for r in rids: acc[r] += w * sc[r] / t3
        find = max(acc, key=acc.get)
        # ── T1 공략: "충분히 확실한 검출 중 **가장 최근**" — 이동 전 기간이 길어
        # 상위권이 옛 자리 목격으로 도배되는 문제를 시간으로 가른다 (§eval_temporal
        # 의 실패와 달리, 문턱을 먼저 걸고 그 안에서 최신을 본다)
        def loc_of(idxs):
            a2 = {r: 0.0 for r in rids}
            for i in idxs:
                w = max(0.0, float(TS[i]) - base)
                cy, cx = py[i, ti], px[i, ti]
                sc = {r: 0.0 for r in rids}
                for c in range(nT, len(vocab)):
                    t = vocab[c]
                    if t not in idf or S[i, c] < .05: continue
                    d = np.hypot(py[i, c]-cy, px[i, c]-cx)
                    w2 = float(S[i, c]) / (1.0 + d/6.0) * idf[t]
                    for r in rids:
                        if t in rtypes.get(r, ()): sc[r] += w2
                if ZX is not None:
                    on = np.where(XSc[i] >= 0.15)[0]
                    si_ = {r: 0.0 for r in rids}
                    for k2 in on:
                        d = np.hypot(xy_[i, k2, 0]-cy, xy_[i, k2, 1]-cx)
                        si_[Xr[k2]] += float(XSc[i, k2]) / (1.0 + d/6.0)
                    t2 = sum(si_.values())
                    if t2 > 0:
                        for r in rids: sc[r] = sc[r]/(sum(sc.values())+1e-9) + si_[r]/t2
                nb = {arm[i]} | adj.get(arm[i], set())
                for r in rids:
                    if r not in nb: sc[r] *= .25
                t3 = sum(sc.values()) + 1e-9
                for r in rids: a2[r] += (w + 1e-6) * sc[r] / t3
            return max(a2, key=a2.get) if a2 else None
        top50 = np.argsort(-TS)[:50]
        recent = sorted(top50, key=lambda i: -ts[i])[:5]
        find_recent = loc_of(recent)
        th2 = np.quantile(TS, 0.98)
        okm = np.ones(len(TS), bool)
        for c in range(nT):
            if c == ti: continue
            okm &= ~((P[:, c] == P[:, ti]) & (S[:, c] > S[:, ti]))
        cand_c = [i for i in np.where((TS >= th2) & okm)[0]]
        recent_c = sorted(cand_c, key=lambda i: -ts[i])[:5]
        find_ctr = loc_of(recent_c) if recent_c else find_recent
        # ── 군집(사건) 단위: 진짜 목격은 몰려 다니고 오검출은 고립된다 ──
        # 문턱 통과 프레임을 시간 간격 90초 이내로 묶고, 크기 2+ 군집만 남긴 뒤
        # **가장 최신 군집**의 방을 답한다. (사용자 제안: 1차로 거르고 시간 반영)
        th_ = np.quantile(TS, 0.98)

        hits = sorted(np.where(TS >= th_)[0], key=lambda i: ts[i])
        evs = []
        for i in hits:
            if evs and ts[i] - ts[evs[-1][-1]] <= 90:
                evs[-1].append(i)
            else:
                evs.append([i])
        evs = [e for e in evs if len(e) >= 2]
        find_event = loc_of(evs[-1][-5:]) if evs else find_recent
        # ── 시간 분포 전체를 읽는다 (사용자 제안의 확장) ──
        # 문턱 통과 목격들을 프레임별로 국소화해 방-시간 수열을 만들고,
        #   이동의 서명   = 어떤 방이 **중간부터 새로 나타나** 끝까지 이어짐
        #   오검출의 서명 = 처음부터 끝까지 계속 있음 (혼동 물체도 정지해 있으므로)
        # → 마지막 1/3 에 목격이 있는 방 중 **첫 등장이 가장 늦은** 방을 답한다.
        hr = [(ts[i], loc_of([i])) for i in hits]
        hr = [(t, r) for t, r in hr if r]
        cnt_r = Counter(r for _, r in hr)
        hr2 = [(t, r) for t, r in hr if cnt_r[r] >= 2]      # 총 2회 미만 방은 잡음
        find_onset = find_recent
        if hr2:
            tcut = hr2[0][0] + (hr2[-1][0] - hr2[0][0]) * 2 / 3
            tail_rooms = {r for t, r in hr2 if t >= tcut}
            if tail_rooms:
                first = {}
                for t, r in hr2:
                    if r in tail_rooms and r not in first: first[r] = t
                find_onset = max(first, key=first.get)
        inr = np.where(arm == sg)[0]
        drop = None
        if len(inr) >= 9:
            e = inr[:len(inr)//3]; l = inr[-len(inr)//3:]
            k2 = max(3, len(e)//3)
            sig = AS[e[np.argsort(-TS[e])[:k2]]].mean(0); sig /= (np.linalg.norm(sig)+1e-9)
            pe = AS[e] @ sig/(np.linalg.norm(AS[e],axis=1)+1e-9)
            pl = AS[l] @ sig/(np.linalg.norm(AS[l],axis=1)+1e-9)
            thp = np.quantile(pe, .7); ge = e[pe >= thp]; gl = l[pl >= thp]
            if len(ge) >= 3 and len(gl) >= 3:
                drop = float(np.quantile(TS[ge], .9) - np.quantile(TS[gl], .9))
        bel = max(((PR.get(v0["type"], {}).get(rt[r], .25)/max(nrt[rt[r]],1), r)
                   for r in rids if r != sg))[1]
        # ── 오라클 검증기 상한: 낮은 문턱(q0.80, 회수 0.986) + 완벽 검증 + 최신 3장 ──
        # VLM 이 시뮬 렌더에서 필터 급이 안 되므로(최고 0.60/0.70), 검증기 슬롯의
        # 상한을 GT 로 잰다. 이 값이 실촬영 검증기에게 요구할 스펙이다.
        th3 = np.quantile(TS, 0.80)
        cands = sorted(np.where(TS >= th3)[0], key=lambda i: -ts[i])
        ver = [i for i in cands if vis[i]][:3]
        find_ov = loc_of(ver) if ver else find_recent
        # ── 불완전 검증기 모의: 9B 실측 수준 (진짜 수용 0.75 · 가짜 수용 p_fa) ──
        _vr = np.random.default_rng(hash(oid) % 2**31)
        def simver(p_acc, p_fa):
            out = []
            for i in cands:
                ok = (_vr.random() < p_acc) if vis[i] else (_vr.random() < p_fa)
                if ok:
                    out.append(i)
                    if len(out) == 3: break
            return loc_of(out) if out else find_recent
        find_v75 = simver(0.42, 0.01)     # 독립 FA 모형 (기각 0.99)
        # ── FA 상관 모형: 오검출 = 같은 혼동물의 반복. 후보 프레임의 패치 위치에서
        # 가장 가까운 GT 개체를 정체로 삼고, **개체 단위로** 한 번 속으면 그 개체의
        # 모든 목격을 수용한다 (비관 모형) ──
        def ident(i):
            m = live[ts[i]]
            cx = (P[i, ti] % pw + .5) / pw * 384
            cy = (P[i, ti] // pw + .5) / ph * 384
            best = (1e9, None)
            for o2, c in list((m.get("ctr") or {}).items()) +                          list((m.get("anch") or {}).items()):
                if not c: continue
                d = np.hypot(c[0]-cx, c[1]-cy)
                if d < best[0]: best = (d, o2)
            return best[1] if best[0] < 40 else "bg%d" % (i // 20)
        def simcorr(p_acc, p_id):
            coin = {}
            out = []
            for i in cands:
                if vis[i]:
                    ok = _vr.random() < p_acc
                else:
                    k = ident(i)
                    if k not in coin: coin[k] = _vr.random() < p_id
                    ok = coin[k]
                if ok:
                    out.append(i)
                    if len(out) == 3: break
            return loc_of(out) if out else find_recent
        find_v75f = simcorr(0.42, 0.05)   # 개체당 5% 로 속음 (상관 비관)
        rows.append(dict(tier=tier, sg=sg, tgt=tgt, find=find, find_recent=find_recent,
                         find_ov=find_ov, find_v75=find_v75, find_v75f=find_v75f,
                         find_event=find_event, find_onset=find_onset, find_ctr=find_ctr, drop=drop, bel=bel))

drops = [r["drop"] for r in rows if r["drop"] is not None]
ta = float(np.quantile(drops, .95))
def sys_ans(r):
    if r["drop"] is not None and r["drop"] >= ta: return r["bel"]
    return r["sg"]

lab = {"T1": "T1 발견 가능 (새 방 목격)", "T2": "T2 부재추론만 가능",
       "T3": "T3 관측 불가 (belief 뿐)", "T4r": "T4 안 움직임·재확인됨",
       "T4u": "T4 안 움직임·미확인"}
print("=== 답 가능성 계층별 채점 · %s · n=%d ===" % (ROOT, len(rows)))
print("%-26s %-5s %-8s %-8s %-8s %s" % ("계층", "건수", "현시스템", "검색만", "이상적", "이상적의 정의"))
for t in ("T1", "T2", "T3", "T4r", "T4u"):
    rs = [r for r in rows if r["tier"] == t]
    if not rs: continue
    cur = np.mean([sys_ans(r) == r["tgt"] for r in rs])
    fnd = np.mean([r["find"] == r["tgt"] for r in rs])
    fr_ = np.mean([r["find_recent"] == r["tgt"] for r in rs])
    fe_ = np.mean([r["find_event"] == r["tgt"] for r in rs])
    fo_ = np.mean([r["find_onset"] == r["tgt"] for r in rs])
    fc_ = np.mean([r["find_ctr"] == r["tgt"] for r in rs])
    fv_ = np.mean([r["find_ov"] == r["tgt"] for r in rs])
    f75 = np.mean([r["find_v75"] == r["tgt"] for r in rs])
    f75f = np.mean([r["find_v75f"] == r["tgt"] for r in rs])
    ideal = {"T1": 1.0, "T2": np.mean([r["bel"] == r["tgt"] for r in rs]),
             "T3": np.mean([r["bel"] == r["tgt"] for r in rs]),
             "T4r": 1.0, "T4u": 1.0}[t]
    idef = {"T1": "목격 프레임을 찾으면 1.0", "T2": "완벽 부재 + belief",
            "T3": "belief 가 상한", "T4r": "기록 유지 = 1.0", "T4u": "기록 유지 = 1.0"}[t]
    print("%-26s %-5d 시스템 %.3f | 상위10 %.3f 최신 %.3f 오라클 %.3f | 검증기0.75/기각1.0 **%.3f** · /기각0.8 %.3f | 이상 %.3f"
          % (lab[t], len(rs), cur, fnd, fr_, fv_, f75, f75f, ideal))
ans = [r for r in rows if r["tier"] != "T3"]
print("\n**답 가능 문제만(T3 제외) 현시스템: %.3f** (n=%d)"
      % (np.mean([sys_ans(r) == r["tgt"] for r in ans]), len(ans)))
print("전체(기존 방식):              %.3f (n=%d)"
      % (np.mean([sys_ans(r) == r["tgt"] for r in rows]), len(rows)))
