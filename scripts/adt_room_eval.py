#!/usr/bin/env python3
"""**방 단위**로 네 가지를 한 번에 잰다 — 우리 설계 기준에 맞춘 측정.

    $P scripts/adt_room_eval.py --seq <name>

⚠️ **왜 다시 재는가**(2026-08-21, 사용자 지적): 그동안 장소 식별을 **가구 구획**
단위로 쟀다 — HD-EPIC 의 `counter.005` vs `counter.006` 같은 것이다. 그런데 우리
설계 기준은 **방 단위**다(belief 를 방 단위로 통일하기로 했다). "노트북 어디 있어"
의 답은 **"부엌"** 이지 "조리대 6번" 이 아니다.

게다가 **HD-EPIC 은 방이 하나다**(전 참가자가 부엌에서만 촬영) — 방 단위 측정이
아예 불가능한 데이터였다. ADT 는 아파트라 방이 여러 개이고, GT 물체 위치와 포즈가
모두 있어 넷을 한 데이터에서 잴 수 있다.

네 가지:
  ① **장소 식별** — 프레임 → 방. 방 키(latent)와 대조
  ② **증거 검색** — 물체 이름으로 검색 → 그 물체가 **있는 방**의 프레임이 오는가
  ③ **마지막 목격** — 검색 상위에서 가장 늦은 프레임의 방 = 물체의 현재 방인가
  ④ **부재** — 물체가 방 A 를 떠났을 때, A 를 다시 봤을 때 없음을 아는가

방 라벨은 `regions_*.npz` 의 바닥 격자(`rooms`)에서 온다 — 우리가 씬그래프 구축 때
만든 것이라 실사용과 같은 출처다.
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="Apartment_release_decoration_seq137_M1292")
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--regions", default="regions_full.npz")
    ap.add_argument("--min-disp", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sd = os.path.join(args.root, args.seq)
    z = np.load(os.path.join(sd, "clip_frames.npz"))
    E = z["emb"].astype(np.float32); fidx = z["idx"]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    T = np.loadtxt(os.path.join(sd, "pose", "poses.txt")).reshape(-1, 4, 4)
    gt = json.load(open(os.path.join(sd, "gt", "objects.json")))["instances"]
    rz = np.load(os.path.join(sd, args.regions))
    rooms, lo, res = rz["rooms"], rz["lo"], float(rz["res"])

    def room_of(xy):
        """세계좌표(수평 2D) → 방 라벨. 격자 밖·미분류는 -1."""
        g = np.floor((np.asarray(xy) - lo) / res).astype(int)
        if g.ndim == 1:
            g = g[None]
        out = np.full(len(g), -1)
        ok = ((g[:, 0] >= 0) & (g[:, 0] < rooms.shape[1]) &
              (g[:, 1] >= 0) & (g[:, 1] < rooms.shape[0]))
        out[ok] = rooms[g[ok, 1], g[ok, 0]]
        return out

    # ADT 는 y 가 위(up) — 수평면은 (x, z)
    cam = T[fidx][:, :3, 3]
    cam_room = room_of(cam[:, [0, 2]])
    labs = sorted({int(r) for r in cam_room if r > 0})
    print("프레임 %d · 방 라벨 %s · 분포 %s"
          % (len(fidx), labs, dict(Counter(int(r) for r in cam_room))))
    if len(labs) < 2:
        print("방이 2개 미만 — 방 단위 측정 불가")
        return

    from scripts.absence_evidence import clip_text

    # ── ① 장소 식별: 앞 40% 로 방 키를 만들고 뒤에서 채점
    n = len(fidx); split = int(n * 0.4)
    keys, kl = [], []
    for L in labs:
        idx = np.nonzero((cam_room == L) & (np.arange(n) < split))[0]
        if len(idx) < 5:
            continue
        v = E[idx].mean(0); v /= np.linalg.norm(v) + 1e-9
        keys.append(v); kl.append(L)
    res1 = None
    if len(keys) >= 2:
        K = np.stack(keys)
        te = np.nonzero((cam_room > 0) & (np.arange(n) >= split))[0]
        if len(te):
            pred = np.array([kl[int(np.argmax(K @ E[i]))] for i in te])
            acc = float((pred == cam_room[te]).mean())
            maj = Counter(int(r) for r in cam_room[te]).most_common(1)[0][1] / len(te)
            res1 = (acc, maj, len(te))
            print("\n① **장소 식별(방)** %.3f · 최빈방 %.3f · 우연 %.3f (n=%d)"
                  % (acc, maj, 1.0 / len(kl), len(te)))

    # ── 물체별 방 이력
    obj_room = {}
    for k, r in gt.items():
        c = (r.get("category") or "").strip()
        pos = r.get("positions") or []
        if not c or not pos:
            continue
        P = np.array(pos)
        if len(P) == 1:
            P = np.repeat(P, len(T), 0)
        rr = room_of(P[fidx][:, [0, 2]]) if len(P) >= len(T) else room_of(P[:, [0, 2]])
        obj_room[k] = (c, rr)

    cats = sorted({c for c, _ in obj_room.values()})
    Q = clip_text(["a photo of a " + c for c in cats], args.device)
    ci = {c: i for i, c in enumerate(cats)}

    # ── ②③ 증거 검색 · 마지막 목격 (방 단위 채점)
    hit, last_ok, nq = [], [], 0
    for k, (c, rr) in obj_room.items():
        val = np.nonzero(rr > 0)[0]
        if len(val) < 5:
            continue
        sim = E @ Q[ci[c]]
        top = np.argsort(-sim)[:args.topk]
        # ② 상위 k 중 하나라도 그 물체가 **그때 있던 방**과 맞는가
        hit.append(bool(any(rr[i] > 0 and rr[i] == cam_room[i] for i in top)))
        # ③ 상위 중 가장 늦은 프레임의 방 = 물체의 마지막 방인가
        latest = int(max(top))
        last_ok.append(bool(cam_room[latest] == rr[val[-1]]))
        nq += 1
    if nq:
        print("② **증거 검색(방 일치)** hit@%d %.3f (n=%d)"
              % (args.topk, float(np.mean(hit)), nq))
        print("③ **마지막 목격(방)** %.3f (n=%d)" % (float(np.mean(last_ok)), nq))

    # ── ④ 부재: 방이 바뀐 물체 vs 안 바뀐 물체
    mv, st = [], []
    for k, (c, rr) in obj_room.items():
        r_ = gt[k]
        big = [m for m in (r_.get("moves") or []) if m["displacement_m"] >= args.min_disp]
        val = np.nonzero(rr > 0)[0]
        if len(val) < 10:
            continue
        changed = len({int(x) for x in rr[val]}) > 1
        sim = E @ Q[ci[c]]
        # 원래 방의 프레임에서, 후반부 키워드 점수
        r0 = int(rr[val[0]])
        inr = np.nonzero((cam_room == r0) & (np.arange(n) >= split))[0]
        if len(inr) < 5:
            continue
        v = float(np.median(sim[inr]))
        (mv if (changed or big) else st).append(v)
    if len(mv) >= 3 and len(st) >= 3:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(st, mv, alternative="greater")
        print("④ **부재(방 이탈)** 이동 %d · 정적 %d · 분리 AUC **%.3f** (p=%.3f)"
              % (len(mv), len(st), u / (len(mv) * len(st)), p))
    else:
        print("④ 부재 — 표본 부족(이동 %d · 정적 %d)" % (len(mv), len(st)))

    if args.out:
        json.dump(dict(place=res1, hit=float(np.mean(hit)) if hit else None,
                       last=float(np.mean(last_ok)) if last_ok else None,
                       n=nq), open(args.out, "w"))
        print("→ %s" % args.out)


if __name__ == "__main__":
    main()
