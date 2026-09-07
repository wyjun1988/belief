"""재구성의 live 카메라 중심을 GT live 위치에 직접 RANSAC sim3 로 맞춰 본다 — 지도 정렬을 거치지 않으므로
'live 모델 자체가 접혔는가(인라이어 낮음)' 와 '지도만 잘못 얹혔는가(live 인라이어 높음)' 를 가른다."""
import sys, os, json, numpy as np, pycolmap
sys.path.insert(0, os.path.expanduser("~/work/khronos"))
from kx.depth.pose_stitch import umeyama
rec_root, house_dir = sys.argv[1], sys.argv[2]
subs = [d for d in sorted(os.listdir(rec_root)) if os.path.isdir(os.path.join(rec_root, d))]
recs = {d: pycolmap.Reconstruction(os.path.join(rec_root, d)) for d in subs}
d = max(recs, key=lambda k: recs[k].num_reg_images()); rec = recs[d]
g = json.load(open(os.path.join(house_dir, "gt.json"))); live = {m["t"]: m for m in g["live"]}; gm = g["map"]
def sim3_ransac(X, Y, th=0.5, iters=500, seed=0):
    rng = np.random.default_rng(seed); best = None
    for _ in range(iters):
        idx = rng.choice(len(X), 3, replace=False)
        try: s, R, t = umeyama(X[idx], Y[idx])
        except Exception: continue
        e = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1); inl = e < th
        if best is None or inl.sum() > best.sum(): best = inl
    s, R, t = umeyama(X[best], Y[best]); e = np.linalg.norm((s * (R @ X.T)).T + t - Y, axis=1)
    return float(best.mean()), float(np.median(e)), float((e < 0.5).mean()), s
def run(tag, names_fn):
    X, Y = [], []
    for im in rec.images.values():
        if not im.has_pose: continue
        y = names_fn(im.name)
        if y is None: continue
        X.append(np.asarray(im.projection_center(), float)); Y.append(y)
    X, Y = np.array(X), np.array(Y)
    if len(X) < 10: print("%-6s n=%d (부족)" % (tag, len(X))); return
    out = []
    for mirror in (False, True):
        M = np.diag([-1.0, 1, 1]) if mirror else np.eye(3)
        out.append((sim3_ransac(X @ M.T, Y), mirror))
    (inl, med, lt05, s), mirror = max(out, key=lambda o: o[0][0])
    print("%-6s n=%4d · RANSAC 인라이어 %.2f · ATE 중앙 %.2fm · <0.5m %.2f · 스케일 %.3f · 미러 %s" % (tag, len(X), inl, med, lt05, s, mirror))
print("모델 %s/%s · 등록 %d · live %d · map %d" % (os.path.basename(rec_root), d, rec.num_reg_images(),
      sum(1 for im in rec.images.values() if im.has_pose and im.name.startswith("live/")), sum(1 for im in rec.images.values() if im.has_pose and im.name.startswith("map/"))))
run("live", lambda n: (lambda t: np.array([live[t]["apos"][0], 1.5, live[t]["apos"][1]]) if t in live else None)(int(os.path.basename(n)[:-4])) if n.startswith("live/") else None)
maps = sorted(f for f in os.listdir(os.path.join(house_dir, "map")) if f.endswith(".jpg")); mi = {"map/" + f: k for k, f in enumerate(maps)}
run("map", lambda n: np.array([gm[mi[n]]["apos"][0], 1.5, gm[mi[n]]["apos"][1]]) if n in mi else None)
run("both", lambda n: (np.array([gm[mi[n]]["apos"][0], 1.5, gm[mi[n]]["apos"][1]]) if n in mi else None) if n.startswith("map/") else (lambda t: np.array([live[t]["apos"][0], 1.5, live[t]["apos"][1]]) if t in live else None)(int(os.path.basename(n)[:-4])))
