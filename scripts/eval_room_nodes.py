#!/usr/bin/env python3
"""방 인지 — CLIP 노드 방출 (§57 이식) vs 정적점수 방출.

    THOR_ROOT=data/thor5 CLIP_PREFIX=~/khcache/c5_5_ python scripts/eval_room_nodes.py
"""
import glob, json, os
import numpy as np

ROOT = os.environ.get("THOR_ROOT", "data/thor5")
CP = os.path.expanduser(os.environ.get("CLIP_PREFIX", "/tmp/c5_"))
res = {k: [0, 0] for k in ("argmax", "viterbi")}
for hd in sorted(glob.glob(ROOT + "/house_*")):
    rd = os.path.realpath(hd); hn = os.path.basename(rd)
    f = CP + hn + ".npz"
    if not os.path.exists(f): continue
    z = np.load(f, allow_pickle=True)
    ME, MR, LE, ts = z["me"], list(z["mr"]), z["le"], z["ts"]
    g = json.load(open(os.path.join(rd, "gt.json")))
    live = {m["t"]: m for m in g["live"]}
    arm = np.array([live[t]["room"] for t in ts])
    rids = sorted(set(MR))
    sim = LE @ ME.T                                   # (배회, 노드)
    em = np.zeros((len(LE), len(rids)))
    for k, r in enumerate(rids):
        m = [i for i, rr in enumerate(MR) if rr == r]
        em[:, k] = sim[:, m].max(1)                   # 노드-최대 방출 (§57)
    tau = 0.01
    p = np.exp((em - em.max(1, keepdims=True)) / tau)
    p /= p.sum(1, keepdims=True)
    pred = np.array(rids, object)[em.argmax(1)]
    res["argmax"][0] += int((pred == arm).sum()); res["argmax"][1] += len(arm)
    sm = g.get("scene_meta", {})
    adj = {}
    for a, b in sm.get("doors", []):
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    K = len(rids)
    tr = np.full((K, K), 1e-4)
    for a2, r in enumerate(rids):
        tr[a2, a2] = 0.95
        nbr = [b2 for b2, r2 in enumerate(rids) if r2 in adj.get(r, ())]
        for b2 in nbr: tr[a2, b2] = 0.05 / max(len(nbr), 1)
    tr /= tr.sum(1, keepdims=True)
    lt = np.log(tr); le_ = np.log(p + 1e-9)
    dp = le_[0].copy(); bk = np.zeros((len(ts), K), int)
    for i in range(1, len(ts)):
        c = dp[:, None] + lt
        bk[i] = c.argmax(0); dp = c.max(0) + le_[i]
    path = np.zeros(len(ts), int); path[-1] = dp.argmax()
    for i in range(len(ts) - 2, -1, -1): path[i] = bk[i + 1, path[i + 1]]
    predv = np.array(rids, object)[path]
    res["viterbi"][0] += int((predv == arm).sum()); res["viterbi"][1] += len(arm)
    sp = os.path.expanduser(os.environ.get("SAVE_PREFIX", ""))
    if sp:
        np.savez_compressed(sp + hn + ".npz", room=predv, ts=ts)

print("=== 방 인지 · CLIP 노드 방출 (%s) ===" % ROOT)
for k, (ok, n) in res.items():
    print("  %-8s **%.3f**  (n=%d)" % (k, ok / max(n, 1), n))
