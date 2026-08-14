"""ADT GT → 채점 기준 타임라인 (`gt/objects.json`).

P3 의 모든 지표(노드 위치오차·변화감지 P/R·지연·정적물체 오탐율)가 이 파일 하나를
기준으로 매겨진다. 그래서 여기서 **"물체가 어디에 있고 언제 움직였는가"를 한 번만** 정의한다.

두 가지 결정:

1. **위치 = AABB 중심(world)**. `scene_objects.csv` 의 `t_wo` 는 물체 원점이고 메시에 따라
   중심에서 한참 떨어져 있다. 씬그래프 노드는 관측된 표면의 중심에 생기므로, 원점으로
   채점하면 계통 오차가 그대로 지표에 실린다.
2. **이동 판정은 속도 기반**. 모캡이라 노이즈는 없지만 사람이 물건을 들었다 제자리에
   놓는 구간이 많아 "첫↔마지막 변위"만 보면 이동을 통째로 놓친다.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

V_MIN = 0.03        # m/s. 이보다 느리면 정지로 본다 (GT 는 모캡이라 임계가 낮아도 안전)
GAP_S = 1.0         # 이보다 짧은 정지는 한 번의 이동으로 잇는다
D_MIN = 0.10        # 구간 순변위가 이보다 작으면 '들었다 제자리' 로 보고 버린다


def _runs(mask: np.ndarray):
    """불리언 마스크의 True 구간을 (start, end_inclusive) 로 반환."""
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0])
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask) - 1)
    return list(zip(starts, ends))


def _detect_moves(pos: np.ndarray, dt: float):
    """(F,3) 위치열 → 이동 구간 목록."""
    if len(pos) < 3:
        return []
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt      # 길이 F-1
    moving = speed > V_MIN
    gap = max(int(round(GAP_S / dt)), 1)

    filled = moving.copy()                          # 짧은 정지 메우기 (closing)
    for s, e in _runs(~moving):
        if s > 0 and e < len(moving) - 1 and (e - s + 1) <= gap:
            filled[s:e + 1] = True

    moves = []
    for s, e in _runs(filled):
        i0, i1 = s, min(e + 1, len(pos) - 1)        # speed 인덱스 → 위치 인덱스
        disp = float(np.linalg.norm(pos[i1] - pos[i0]))
        if disp < D_MIN:
            continue
        moves.append({
            "start_idx": int(i0),
            "end_idx": int(i1),
            "from": np.round(pos[i0], 4).tolist(),
            "to": np.round(pos[i1], 4).tolist(),
            "displacement_m": round(disp, 4),
            "path_length_m": round(float(
                np.linalg.norm(np.diff(pos[i0:i1 + 1], axis=0), axis=1).sum()), 4),
        })
    return moves


def build_timeline(seq_dir: str, out_dir: str) -> dict:
    frames = json.load(open(os.path.join(out_dir, "frames.json")))
    t_frames = np.array([f["t_ns"] for f in frames["frames"]], dtype=np.int64)
    dt = 1.0 / float(frames["fps"])
    F = len(t_frames)

    instances = json.load(open(os.path.join(seq_dir, "instances.json")))
    obj = pd.read_csv(os.path.join(seq_dir, "scene_objects.csv"))
    bb = pd.read_csv(os.path.join(seq_dir, "3d_bounding_box.csv"))

    aabb = {}                    # uid → (local center, extent)
    for uid, g in bb.groupby("object_uid"):
        r = g.iloc[0]
        lo = np.array([r["p_local_obj_xmin[m]"], r["p_local_obj_ymin[m]"], r["p_local_obj_zmin[m]"]])
        hi = np.array([r["p_local_obj_xmax[m]"], r["p_local_obj_ymax[m]"], r["p_local_obj_zmax[m]"]])
        aabb[int(uid)] = (0.5 * (lo + hi), hi - lo)

    out, n_moved = {}, 0
    for uid, g in obj.groupby("object_uid"):
        uid = int(uid)
        info = instances.get(str(uid))
        if info is None:
            continue
        ts = g["timestamp[ns]"].to_numpy(dtype=np.int64)
        xyz = g[["t_wo_x[m]", "t_wo_y[m]", "t_wo_z[m]"]].to_numpy(dtype=np.float64)
        quat = g[["q_wo_x", "q_wo_y", "q_wo_z", "q_wo_w"]].to_numpy(dtype=np.float64)

        if (ts < 0).all():
            # 정적 물체는 t=-1 한 행뿐 — 시퀀스 내내 그 자리.
            sel = np.zeros(F, dtype=int)
        else:
            m = ts >= 0
            ts, xyz, quat = ts[m], xyz[m], quat[m]
            order = np.argsort(ts)
            ts, xyz, quat = ts[order], xyz[order], quat[order]
            # GT 는 30Hz 이상이라 10Hz 격자에 최근접이면 오차 ≤16ms. 회전 보간보다 안전하다.
            j = np.searchsorted(ts, t_frames)
            j = np.clip(j, 1, len(ts) - 1)
            sel = np.where(np.abs(ts[j] - t_frames) < np.abs(ts[j - 1] - t_frames), j, j - 1)

        t_wo = xyz[sel]
        R = Rotation.from_quat(quat[sel]).as_matrix()               # (F,3,3)
        c_local, extent = aabb.get(uid, (np.zeros(3), None))
        centers = t_wo + np.einsum("fij,j->fi", R, c_local)

        moves = _detect_moves(centers, dt)
        if moves:
            n_moved += 1
        out[str(uid)] = {
            "instance_id": uid,
            "name": info.get("instance_name"),
            "category": info.get("category"),
            "motion_type": info.get("motion_type"),
            "extent_m": None if extent is None else np.round(extent, 4).tolist(),
            "positions": np.round(centers, 4).tolist(),            # world AABB 중심
            "moves": moves,
            "total_displacement_m": round(float(np.linalg.norm(centers[-1] - centers[0])), 4),
        }

    doc = {
        "sequence": os.path.basename(seq_dir.rstrip("/")),
        "n_frames": F,
        "fps": frames["fps"],
        "params": {"v_min_mps": V_MIN, "gap_s": GAP_S, "d_min_m": D_MIN},
        "n_instances": len(out),
        "n_dynamic": sum(1 for v in out.values() if v["motion_type"] == "dynamic"),
        "n_with_moves": n_moved,
        "instances": out,
    }
    os.makedirs(os.path.join(out_dir, "gt"), exist_ok=True)
    with open(os.path.join(out_dir, "gt", "objects.json"), "w") as f:
        json.dump(doc, f)
    return {k: v for k, v in doc.items() if k != "instances"}
