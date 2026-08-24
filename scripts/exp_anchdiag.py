# **완전 실전**: 프레임 선택도 앵커 검출도 전부 OWL. 최초 맵(방→정적타입)만 GT
# — 사용자 설계상 맵은 한 번만 무겁게 만든다(depth+다시점).
import json, glob, os, numpy as np
from collections import Counter
import prior
from ai2thor.controller import Controller
ds = prior.load_dataset("procthor-10k")["train"]

def inside(x, z, pts):
    c = False; n = len(pts)
    for i in range(n):
        x1, z1 = pts[i]; x2, z2 = pts[(i+1) % n]
        if (z1 > z) != (z2 > z) and x < (x2-x1)*(z-z1)/(z2-z1+1e-12) + x1: c = not c
    return c
def room_of(x, z, polys):
    for rid, pts in polys.items():
        if inside(x, z, pts): return rid
    return None

ctrl = None
res = {}
for hd in sorted(glob.glob("data/thor2y/house_*")):
    hn = os.path.basename(hd)
    z = np.load("/tmp/anchcache_%s.npz" % hn, allow_pickle=True)
    S, P, ph, pw, ts = z["s"], z["p"], int(z["ph"]), int(z["pw"]), z["ts"]
    vocab = list(z["vocab"]); nT = int(z["nT"])
    tgt_v = vocab[:nT]; st_v = vocab[nT:]
    g = json.load(open(os.path.join(hd, "gt.json"))); h = ds[g["house"]]
    polys = {r["id"]: [(c["x"], c["z"]) for c in r["floorPolygon"]] for r in h["rooms"]}
    ctrl = Controller(scene=h, width=64, height=64, quality="Low") if ctrl is None else ctrl
    ctrl.reset(scene=h)
    rtypes = {}
    for o in ctrl.last_event.metadata["objects"]:
        if o.get("pickupable"): continue
        r = room_of(o["position"]["x"], o["position"]["z"], polys)
        if r: rtypes.setdefault(r, Counter())[o["objectType"]] += 1
    rids = sorted(rtypes)
    idf = {t: 1.0/sum(t in rtypes[r] for r in rids) for t in set().union(*[set(rtypes[r]) for r in rids])}
    adj = {r: set() for r in rids}
    for d in h.get("doors", []):
        a, b = d.get("room0"), d.get("room1")
        if a in adj and b in adj: adj[a].add(b); adj[b].add(a)
    live = {m["t"]: m for m in g["live"]}
    arm = np.array([live[t]["room"] if t in live else "" for t in ts])
    moves = sorted(g["moves"], key=lambda m: m["t"])
    py, px = P // pw, P % pw            # 패치 인덱스 → 화면 격자 위치
    KS = (5, 10, 25, 50)
    for oid, v in g["gt0"].items():
        if not v["room"]: continue
        mv = [x for x in moves if x["oid"] == oid]
        tg = mv[-1]["to"] if mv else v["room"]; t0 = mv[-1]["t"] if mv else 0
        ti = vocab.index(v["type"]) if v["type"] in vocab else -1
        if ti < 0: continue
        ok = np.where(ts > t0)[0]
        if len(ok) < 50: continue
        gtvis = np.array([oid in live[t].get("vis", []) for t in ts[ok]])
        order = ok[np.argsort(-S[ok, ti])]
        gtf = ok[gtvis][:25] if gtvis.sum() >= 3 else None
        def rscore(i):
            """프레임 i 에서 타겟 주변 정적 조합으로 방별 점수."""
            cy, cx = py[i, ti], px[i, ti]
            sc = {r: 0.0 for r in rids}
            for j in range(nT, len(vocab)):
                t = vocab[j]
                if t not in idf or S[i, j] < .05: continue
                d = np.hypot(py[i, j]-cy, px[i, j]-cx)
                w = float(S[i, j]) / (1.0 + d/6.0) * idf[t]
                for r in rids:
                    if t in rtypes[r]: sc[r] += w
            if arm[i]:
                nb = {arm[i]} | adj.get(arm[i], set())
                for r in rids:
                    if r not in nb: sc[r] *= 0.25      # 이웃 밖은 강하게 깎는다
            tot = sum(sc.values()) + 1e-9
            return {r: sc[r]/tot for r in rids}        # 프레임 안에서 정규화

        RS = {}
        def hard(idx):
            out = [max(rids, key=lambda r: RS.setdefault(i, rscore(i))[r]) for i in idx]
            return max(set(out), key=out.count) if out else None
        def soft(idx):
            base = float(np.median(S[ok, ti]))
            acc = {r: 0.0 for r in rids}
            for i in idx:
                w = max(0.0, float(S[i, ti]) - base)   # 물체별 자기보정 가중
                rs = RS.setdefault(i, rscore(i))
                for r in rids: acc[r] += w * rs[r]
            return max(rids, key=lambda r: acc[r])
        vs = {t: (oid in live[t].get("vis", [])) for t in ts[ok]}
        sh = order[:25]
        good = [i for i in sh if vs[ts[i]]]; bad = [i for i in sh if not vs[ts[i]]]
        res.setdefault("all25", []).append(hard(sh) == tg)
        if len(good) >= 2:
            res.setdefault("good", []).append(hard(good) == tg)
            res.setdefault("nroom", []).append(len(rids))
        if len(bad) >= 2:
            r = hard(bad)
            res.setdefault("bad_true", []).append(r == tg)
            # 오답 프레임들이 **한 방으로 몰리나** (우연이면 1/방수)
            votes = [max(rids, key=lambda q: RS.setdefault(i, rscore(i))[q]) for i in bad]
            res.setdefault("bad_conc", []).append(max(Counter(votes).values())/len(votes))
            res.setdefault("bad_nr", []).append(1.0/len(rids))
lab = {"all25": "OWL 상위25 전부", "good": "**그 중 실제로 보이는 것만**",
       "bad_true": "그 중 안 보이는 것만 → 정답방 적중",
       "bad_conc": "안 보이는 프레임들이 한 방에 몰린 비율",
       "bad_nr": "  (우연이면 1/방수)"}
print("=== 오답 프레임이 무엇을 하고 있나 ===")
for k in ("all25", "good", "bad_true", "bad_conc", "bad_nr"):
    a = np.array(res[k]); n = len(a)
    if n == 0: continue
    lo, hi = np.percentile([a[np.random.randint(0, n, n)].mean() for _ in range(2000)], [2.5, 97.5])
    print("  %-28s %.3f [%.3f %.3f]  n=%d" % (lab[k], a.mean(), lo, hi, n))
