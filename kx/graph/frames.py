"""좌표 관례 — ADT 의 world 프레임에서 '위'가 어느 축인가.

ADT Apartment 씬은 **y 축이 위**다(중력 = (0, −9.81, 0)). 바닥 평면은 x–z.
하드코딩하지 않고 `aria_trajectory.csv` 의 gravity 벡터에서 읽는다 — 다른 씬이나
다른 데이터셋으로 넓힐 때 조용히 틀리는 것을 막는다.
"""
import os

import numpy as np
import pandas as pd


def up_vector(adt_seq_dir):
    """world 좌표계의 '위' 단위벡터."""
    p = os.path.join(adt_seq_dir, "aria_trajectory.csv")
    g = pd.read_csv(p, usecols=["gravity_x_world", "gravity_y_world", "gravity_z_world"],
                    nrows=200).to_numpy(np.float64)
    g = np.median(g, axis=0)
    n = np.linalg.norm(g)
    if n < 1e-6:
        return np.array([0.0, 1.0, 0.0])
    return -g / n                      # 중력의 반대가 위


def floor_basis(up):
    """(e1, e2, up) 정규직교 기저 — e1,e2 가 바닥 평면을 편다."""
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, up)) > 0.9:
        a = np.array([0.0, 0.0, 1.0])
    e1 = a - np.dot(a, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2, up


def to_floor(points, basis):
    """world 점 → (u, v, h). u,v 는 바닥 좌표, h 는 높이."""
    e1, e2, up = basis
    P = np.atleast_2d(points)
    return np.stack([P @ e1, P @ e2, P @ up], axis=1)
