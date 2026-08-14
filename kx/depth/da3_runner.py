"""Depth Anything 3 추론 — 겹치는 슬라이딩 윈도우 + 포즈 조건부 (GPU 쪽에서 실행).

DA3 의 `inference()` 는 extrinsics(w2c) / intrinsics 를 받으면 **포즈 조건부 다중뷰**
모드로 동작한다(`api.py:_normalize_extrinsics` 가 첫 뷰를 기준으로 정규화하고,
`align_to_input_ext_scale=True` 가 입력 포즈의 미터 스케일로 되돌린다). ADT 는 GT 포즈를
주므로 이 경로를 쓰는 게 자명하다 — 윈도우 **내부** 일관성은 모델이 책임진다.

윈도우 **사이**는 여전히 어긋난다. 그건 여기서 고치지 않고 `kx/depth/align.py` 가
전역 앵커로 절대 정렬한다. 그래서 이 모듈의 출력은 "정합 전 원시 예측"이다.

모드:
    mono     프레임 1장씩, 포즈 없음 — 애블레이션의 바닥선
    window   K프레임 윈도우 + 주어진 포즈 (기본)
    nopose   K프레임 윈도우, **포즈도 DA3 가 추정** — 기기 없는 현실 조건
    anchor   nopose + **앞 윈도우 결과를 앵커로 넣어** 순차적으로 푼다

`nopose` 는 윈도우마다 좌표계·스케일이 제각각이라 사후에 sim3 로 이어붙여야 하고,
그 체이닝이 76개 윈도우를 지나며 드리프트를 쌓는다(실측: 전역 자세 산포 11.35°,
ATE 1.43m, 그 결과 물체 위치오차 2.70m — 씬그래프가 무너진다).

`anchor` 는 그 원인을 없앤다. 윈도우 i+1 을 풀 때 **겹치는 프레임의 이미 확정된 전역
포즈를 입력 extrinsics 로 넣어** 모델이 처음부터 앵커 좌표계에서 기하를 만들게 한다.
새 프레임의 포즈는 등속 외삽으로 채워 넣는다(조건화용 초기값).

⚠️ 이때 `align_to_input_ext_scale` 을 반드시 **False** 로 둬야 한다. True 면 DA3 가
`prediction.extrinsics = extrinsics` 로 **입력을 그대로 되돌려주고** 뎁스만 재조정한다
(`api.py:_align_to_input_extrinsics_intrinsics`) — 외삽값이 그대로 나와 추정이 안 된다.
False 면 모델의 **예측 포즈**를 입력 포즈에 Umeyama(+RANSAC) 정합해서 돌려준다.
"""
import json
import os

import numpy as np


def _load_seq(seq_dir):
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    rgb = sorted(os.listdir(os.path.join(seq_dir, "rgb")))
    assert len(rgb) == len(poses), "rgb %d != poses %d" % (len(rgb), len(poses))
    return cam, poses, [os.path.join(seq_dir, "rgb", f) for f in rgb]


def _windows(n, size, stride):
    """겹치는 윈도우와, 각 프레임을 '가장 중앙에 놓인' 윈도우에 배정한 결과."""
    starts = list(range(0, max(n - size, 0) + 1, stride))
    if not starts or starts[-1] + size < n:
        starts.append(max(n - size, 0))
    owner = np.full(n, -1, dtype=np.int32)
    best = np.full(n, np.inf)
    for wi, s in enumerate(starts):
        e = min(s + size, n)
        idx = np.arange(s, e)
        centrality = np.abs(idx - (s + e - 1) / 2.0)
        take = centrality < best[idx]
        owner[idx[take]] = wi
        best[idx[take]] = centrality[take]
    return starts, owner


def run(seq_dir, out_dir, model_name="da3metric-large", mode="window",
        window=24, stride=12, process_res=504, device="cuda",
        limit=None, raw_suffix=""):
    import torch
    from depth_anything_3.api import DepthAnything3

    cam, poses, files = _load_seq(seq_dir)
    if limit:
        files, poses = files[:limit], poses[:limit]
    n = len(files)
    W, H = cam["width"], cam["height"]
    K = np.array(cam["intrinsics"], dtype=np.float32)

    model = DepthAnything3.from_pretrained("depth-anything/%s" % model_name)
    model = model.to(device).eval()
    model.device = device

    raw_dir = os.path.join(out_dir, "depth_raw" + raw_suffix)
    conf_dir = os.path.join(out_dir, "conf_raw" + raw_suffix)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    if mode in ("nopose", "anchor"):
        os.makedirs(os.path.join(out_dir, "poses_raw" + raw_suffix), exist_ok=True)

    if mode == "mono":
        starts, owner = list(range(n)), np.arange(n, dtype=np.int32)
        window = 1
    else:
        starts, owner = _windows(n, window, stride)

    # anchor 모드: 확정된 전역 포즈를 프레임 단위로 들고 간다
    n_fallback = [0]
    solved = np.zeros(n, bool)
    gpose = np.tile(np.eye(4), (n, 1, 1))

    def _extrapolate(target_idx):
        """확정된 마지막 두 포즈에서 등속으로 밀어 초기값을 만든다."""
        done = np.flatnonzero(solved)
        if len(done) == 0:
            return None
        last = done[-1]
        if len(done) >= 2:
            prev = done[-2]
            step = np.linalg.inv(gpose[prev]) @ gpose[last]
            gap = max(last - prev, 1)
        else:
            step, gap = np.eye(4), 1
        out = {}
        for i in target_idx:
            T = gpose[last].copy()
            # ⚠️ 반복 횟수를 8로 캡했다가 그 뒤 프레임이 전부 같은 포즈가 되어
            # DA3 내부 Umeyama 가 "Degenerate covariance rank" 로 죽었다. 캡하지 않는다.
            k = max(int(round((i - last) / gap)), 0)
            for _ in range(k):
                T = T @ step
            out[i] = T
        return out

    import cv2
    meta = {"mode": mode, "model": model_name, "window": window, "stride": stride,
            "process_res": process_res, "frames": n, "windows": len(starts),
            "owner": owner.tolist()}

    for wi, s in enumerate(starts):
        idx = np.arange(s, min(s + window, n))
        mine = idx[owner[idx] == wi]
        if len(mine) == 0:
            continue
        imgs = [files[i] for i in idx]
        if mode in ("mono", "nopose"):
            # 포즈를 주지 않는다 — DA3 가 extrinsics/intrinsics 를 직접 추정한다.
            pred = model.inference(imgs, process_res=process_res)
        elif mode == "anchor":
            known = [i for i in idx if solved[i]]
            if len(known) < 4:
                pred = model.inference(imgs, process_res=process_res)   # 첫 윈도우
            else:
                ext = np.tile(np.eye(4), (len(idx), 1, 1))
                extra = _extrapolate([i for i in idx if not solved[i]]) or {}
                for k, i in enumerate(idx):
                    ext[k] = gpose[i] if solved[i] else extra.get(i, gpose[known[-1]])
                ixt = np.repeat(K[None], len(idx), axis=0)
                try:
                    pred = model.inference(
                        imgs, extrinsics=np.linalg.inv(ext).astype(np.float32),
                        intrinsics=ixt, align_to_input_ext_scale=False,
                        process_res=process_res)
                except Exception as e:
                    # 앵커 포즈 집합이 퇴화하면(직선 궤적 등) 포즈-프리로 물러선다.
                    # 그 윈도우는 사후 sim3 로 붙는다 — 드리프트는 그만큼만 늘어난다.
                    print("   (anchor 실패, pose-free 폴백: %s)" % type(e).__name__, flush=True)
                    pred = model.inference(imgs, process_res=process_res)
                    n_fallback[0] += 1
            # 예측 포즈(앵커 좌표계로 정합된 것)를 확정 포즈로 채택
            E = np.asarray(pred.extrinsics, np.float64)
            if E.shape[-2:] == (3, 4):
                B = np.tile(np.eye(4), (len(E), 1, 1)); B[:, :3, :4] = E; E = B
            c2w = np.linalg.inv(E)
            for k, i in enumerate(idx):
                if owner[i] == wi or not solved[i]:
                    gpose[i] = c2w[k]
                    solved[i] = True
        else:
            # DA3 는 w2c 를 받는다 (api.py 가 affine_inverse 로 c2w 를 만든다).
            w2c = np.linalg.inv(poses[idx]).astype(np.float32)
            ixt = np.repeat(K[None], len(idx), axis=0)
            pred = model.inference(imgs, extrinsics=w2c, intrinsics=ixt,
                                   align_to_input_ext_scale=True, process_res=process_res)

        if mode in ("nopose", "anchor"):
            # 윈도우 고유 좌표계·스케일. 이어붙이기는 pose_stitch 가 한다.
            np.savez_compressed(os.path.join(out_dir, "poses_raw%s" % raw_suffix,
                                             "w%04d.npz" % wi),
                                frames=idx, owner=(owner[idx] == wi),
                                extrinsics=np.asarray(pred.extrinsics, np.float64),
                                intrinsics=np.asarray(pred.intrinsics, np.float64),
                                is_metric=np.array([str(pred.is_metric)]))
        depth = np.asarray(pred.depth, dtype=np.float32)                 # (N,h,w)
        conf = None if pred.conf is None else np.asarray(pred.conf, dtype=np.float32)
        for k, i in enumerate(idx):
            if owner[i] != wi:
                continue
            d = depth[k]
            if d.shape != (H, W):
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
            np.save(os.path.join(raw_dir, "%06d.npy" % i), d.astype(np.float16))
            if conf is not None:
                c = conf[k]
                if c.shape != (H, W):
                    c = cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR)
                np.save(os.path.join(conf_dir, "%06d.npy" % i), c.astype(np.float16))
        print("[%d/%d] window %d..%d  owned %d  (metric=%s)"
              % (wi + 1, len(starts), idx[0], idx[-1], len(mine), pred.is_metric), flush=True)
        del pred
        torch.cuda.empty_cache()

    if mode == "anchor":
        np.savetxt(os.path.join(out_dir, "pose", "poses_da3_anchor%s.txt" % raw_suffix),
                   gpose.reshape(n, 16), fmt="%.9g")
        meta["anchor_solved"] = int(solved.sum())
        meta["anchor_fallbacks"] = n_fallback[0]
    meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    meta["raw_dir"] = os.path.basename(raw_dir)
    with open(os.path.join(out_dir, "da3_meta%s.json" % raw_suffix), "w") as f:
        json.dump(meta, f)
    return meta
