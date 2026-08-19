"""4D 씬그래프 빌더 — 관측 스트림에서 물체 노드와 그 **배치 이력**을 만든다.

DAAAM/Khronos 가 하는 일의 후처리 판이다. Khronos 스택이 돌기 전에도(그리고 돌고 난
뒤에도 대조군으로) 같은 질문에 답할 수 있어야 하므로 순수 파이썬으로 따로 세운다.

핵심 자료구조는 물체당 **placement 목록**이다:

    object → [ {frames: [a,b], position, region}, ... ]

"지금 어디 있나(belief)"에 답하려면 좌표 하나가 아니라 *언제부터 언제까지 거기 있었나*가
필요하다. 그리고 **관측이 끊긴 사이에 일어난 이동**이 이 문제의 본질이라, 이동을
'언제 실제로 일어났는지'가 아니라 **'그래프가 언제 알아챘는지'**(`detected_at`)로
기록한다 — 둘의 차이가 곧 감지 지연이고, home-jepa 가 묻는 미관측 이동의 크기다.

세그멘테이션은 GT 마스크를 쓰므로 데이터 연관(어느 관측이 어느 물체인지)은 공짜다.
따라서 이 모듈이 재는 것은 **기하(뎁스)와 변화 감지**의 품질이지 연관 성능이 아니다.
"""
import json
import os

import numpy as np
from PIL import Image

MIN_AREA = 250          # 픽셀. 이보다 작은 마스크는 3D 중심이 신뢰할 수 없다
MIN_VALID = 80          # 마스크 안에서 유효 뎁스 픽셀 최소 개수
DEPTH_TRIM = 0.6        # 마스크 뎁스 중앙값에서 이만큼(m) 벗어난 픽셀은 경계 누출로 버린다
CONFIRM = 2             # 새 위치를 이만큼 연속 관측해야 '이동'으로 인정
MOVE_MIN = 0.15         # m. 물체 크기에 비례해 커진다
VOXEL = 0.10            # m. 점유 비교 격자
OVERLAP_MIN = 0.25      # 새 관측 복셀의 이 비율 이상이 기존 점유와 겹치면 '그대로 있다'
MAX_PTS = 400           # 관측당 보관할 점 수 (복셀 집합용 서브샘플)


def _instance_points(depth, seg, K, T_wc, wanted, min_area=MIN_AREA, rng=None):
    """한 프레임 → {local_id: (world points [M,3], n_valid)}"""
    out = {}
    ids, counts = np.unique(seg, return_counts=True)
    for lid, cnt in zip(ids, counts):
        if lid == 0 or cnt < min_area or (wanted is not None and int(lid) not in wanted):
            continue
        m = (seg == lid) & (depth > 0)
        if m.sum() < MIN_VALID:
            continue
        v, u = np.nonzero(m)
        d = depth[v, u]
        med = np.median(d)
        keep = np.abs(d - med) < DEPTH_TRIM
        if keep.sum() < MIN_VALID:
            continue
        u, v, d = u[keep], v[keep], d[keep]
        if len(d) > MAX_PTS:
            sel = rng.choice(len(d), MAX_PTS, replace=False)
            u, v, d = u[sel], v[sel], d[sel]
        x = (u - K[0, 2]) / K[0, 0] * d
        y = (v - K[1, 2]) / K[1, 1] * d
        pc = np.stack([x, y, d], axis=1)
        out[int(lid)] = (pc @ T_wc[:3, :3].T + T_wc[:3, 3], int(keep.sum()))
    return out


def _vox(P, size=VOXEL):
    return set(map(tuple, np.floor(np.asarray(P) / size).astype(np.int64)))


def _segment_track(frames, pts, thresh, confirm=CONFIRM, overlap_min=OVERLAP_MIN):
    """관측열 → placement 구간 + 변화 목록.

    ⚠️ 판정 기준은 **중심점 거리가 아니라 복셀 점유 겹침**이다. 처음엔 중심점을
    썼는데 소파(1.28m)·침대(1.44m)·TV장(0.96m) 같은 대형 정적 가구가 전부 '이동'으로
    잡혔다. 이유는 단순하다 — 시점이 바뀌면 *보이는 표면*이 달라져서 관측 중심이
    물체 위를 미끄러진다. 물체가 움직인 게 아니라 우리가 움직인 것이다.

    누적 점유(복셀 집합)와 새 관측의 겹침 비율로 보면 이 문제가 사라진다: 같은
    자리에 있는 한, 어느 면을 보든 새 관측은 기존 점유 안에 떨어진다. 겹침이
    무너지고 중심 거리도 문턱을 넘을 때만 이동으로 인정한다(두 조건 AND).

    관측이 드문드문하므로 속도는 쓸 수 없다. 대신 `confirm` 번 연속으로 어긋나야
    새 배치를 연다 — 한 프레임짜리 마스크 누출에 흔들리지 않는다.
    """
    placements, changes = [], []

    def new_state(f, P):
        return {"f0": f, "f1": f, "vox": _vox(P), "cent": [np.median(P, axis=0)], "n": 1}

    def close(st):
        # 위치 = **프레임별 관측 중심의 중앙값**.
        # ⚠️ 누적 복셀 집합의 중심으로 바꿔봤다가 되돌렸다(belief 수용체 정확도
        # decoration 0.900 → 0.700). 복셀은 가까이서·오래 본 면에 개수가 몰려서
        # 가중치가 시점에 비례해버린다. 프레임 중심의 중앙값은 시점을 고르게
        # 평균하므로 편향이 작다. 복셀은 '같은 자리인가' 판정에만 쓴다.
        # (남은 편향 — 보이는 면 쪽으로 치우침 — 은 미해결. 가구 위치를 GT 로
        #  바꾸면 0.900 → 0.943 이라 이 편향이 다음 개선 지점이다.)
        C = np.asarray(st["cent"])
        V = np.asarray(sorted(st["vox"]), dtype=np.float64) * VOXEL + VOXEL / 2.0
        out = {"start_frame": int(st["f0"]), "end_frame": int(st["f1"]),
               "position": np.median(C, axis=0).round(4).tolist(),
               "n_obs": st["n"], "n_voxels": len(st["vox"])}
        # 위치 추정량 3종을 함께 남긴다 — 어느 쪽이 나은지는 데이터가 정한다.
        # (belief 수용체 정확도로 비교: scripts/_archive/eval_belief.py --furn-pos)
        if len(V):
            lo, hi = V.min(axis=0), V.max(axis=0)
            out["vox_centroid"] = V.mean(axis=0).round(4).tolist()
            out["vox_bbox_center"] = ((lo + hi) / 2).round(4).tolist()
            out["vox_extent"] = (hi - lo + VOXEL).round(4).tolist()
        return out

    cur = new_state(frames[0], pts[0])
    pending = []
    for f, P in zip(frames[1:], pts[1:]):
        vq = _vox(P)
        overlap = len(vq & cur["vox"]) / max(len(vq), 1)
        cent = np.median(P, axis=0)
        moved = overlap < overlap_min and np.linalg.norm(cent - np.median(np.asarray(cur["cent"]), axis=0)) > thresh
        if moved:
            pending.append((f, P, cent))
            if len(pending) >= confirm:
                done = close(cur)
                placements.append(done)
                new_c = np.median(np.asarray([c for _, _, c in pending]), axis=0)
                changes.append({
                    "detected_at_frame": int(pending[0][0]),   # 처음 어긋난 관측
                    "confirmed_at_frame": int(pending[-1][0]),
                    "from": done["position"],
                    "to": new_c.round(4).tolist(),
                    "distance_m": round(float(np.linalg.norm(new_c - np.array(done["position"]))), 4),
                    "unobserved_gap_frames": int(pending[0][0] - done["end_frame"]),
                })
                cur = new_state(pending[0][0], pending[0][1])
                for pf, pp, pc in pending[1:]:
                    cur["vox"] |= _vox(pp)
                    cur["cent"].append(pc)
                    cur["n"] += 1
                    cur["f1"] = pf
                pending = []
        else:
            pending = []
            cur["vox"] |= vq
            cur["cent"].append(cent)
            cur["n"] += 1
            cur["f1"] = f
    placements.append(close(cur))
    return placements, changes


def build_graph(seq_dir, depth_dir="depth", every=1, max_extent=None, min_area=MIN_AREA,
                pose_file="pose/poses.txt", seg_dir="gt/seg", seg_ids="gt/seg_ids.json"):
    cam = json.load(open(os.path.join(seq_dir, "camera_info.json")))
    K = np.array(cam["intrinsics"])
    poses = np.loadtxt(os.path.join(seq_dir, pose_file)).reshape(-1, 4, 4)
    ids = json.load(open(os.path.join(seq_dir, seg_ids)))
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]

    meta = {}
    for local, m in ids.items():
        rec = gt.get(str(m["instance_id"]))
        ext = rec.get("extent_m") if rec else None
        if rec is None and m.get("category") is None:
            ext = None            # SAM 트랙: GT 메타가 없으면 크기는 관측에서 정한다
        if max_extent and ext and max(ext) > max_extent:
            continue                       # 벽·바닥 같은 구조물은 물체 노드로 올리지 않는다
        meta[int(local)] = {**m, "extent_m": ext}
    wanted = set(meta)

    dep_dir = os.path.join(seq_dir, depth_dir)
    seg_dir = os.path.join(seq_dir, seg_dir)
    tracks = {lid: {"f": [], "P": [], "npx": []} for lid in wanted}
    rng = np.random.default_rng(0)
    n_frames = 0
    for i in range(0, len(poses), every):
        dp = os.path.join(dep_dir, "%06d.png" % i)
        sp = os.path.join(seg_dir, "%06d.png" % i)
        if not (os.path.exists(dp) and os.path.exists(sp)):
            continue
        depth = np.array(Image.open(dp)).astype(np.float32) / 1000.0
        seg = np.array(Image.open(sp))
        n_frames += 1
        for lid, (P, npx) in _instance_points(depth, seg, K, poses[i], wanted,
                                              min_area, rng).items():
            tracks[lid]["f"].append(i)
            tracks[lid]["P"].append(P)
            tracks[lid]["npx"].append(npx)

    objects, agents = {}, {}
    for lid, t in tracks.items():
        if len(t["f"]) < 3:
            continue
        m = meta[lid]
        ext = m.get("extent_m")
        thresh = max(MOVE_MIN, 0.5 * (max(ext) if ext else 0.3))
        pl, ch = _segment_track(np.array(t["f"]), t["P"], thresh)
        rec = {
            "local_id": lid,
            "instance_id": m["instance_id"],
            "name": m.get("name"),
            "category": m.get("category"),
            "gt_motion_type": m.get("motion_type"),
            "gt_instance": m.get("gt_instance"),      # SAM 트랙일 때 채점·대조용 (GT 는 None)
            "extent_m": ext,
            "move_threshold_m": round(thresh, 3),
            "n_obs": len(t["f"]),
            "first_frame": int(t["f"][0]),
            "last_frame": int(t["f"][-1]),
            "placements": pl,
            "changes": ch,
        }
        # 사람은 물체가 아니다. 끊임없이 움직이므로 배치 이력 대신 **궤적**으로 따로 둔다
        # (Khronos 의 단기 동역학 층에 해당). 물체 변화 통계를 오염시키지 않는다.
        if (m.get("instance_type") or "").upper() == "HUMAN" or (m.get("category") or "") == "person":
            rec["trajectory"] = [[int(f), *np.median(P, axis=0).round(3).tolist()]
                                 for f, P in zip(t["f"], t["P"])]
            rec.pop("placements"), rec.pop("changes")
            agents[str(m["instance_id"])] = rec
        else:
            objects[str(m["instance_id"])] = rec

    return {
        "sequence": os.path.basename(seq_dir.rstrip("/")),
        "depth_dir": depth_dir,
        "frames_processed": n_frames,
        "n_frames": len(poses),
        "pose_file": pose_file,
        "params": {"min_area": min_area, "confirm": CONFIRM, "move_min_m": MOVE_MIN,
                   "depth_trim_m": DEPTH_TRIM, "max_extent_m": max_extent, "every": every,
                   "voxel_m": VOXEL, "overlap_min": OVERLAP_MIN},
        "objects": objects,
        "agents": agents,
    }
