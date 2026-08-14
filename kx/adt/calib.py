"""Aria 어안(fisheye624) → 선형(pinhole) 카메라 변환.

왜 필요한가: Depth Anything 3 도 Khronos 볼류메트릭 매핑도 **핀홀 모델을 가정**한다.
Aria RGB 는 1408×1408 fisheye624(대각 110°)이고 센서가 90° 눕혀 있어, 원본을 그대로
먹이면 (a) 뎁스 모델이 학습 분포 밖의 왜곡을 보고 (b) Khronos 의 광선 투사가 틀어진다.

여기서 한 번에 처리하는 것:
  1. fisheye624 → LINEAR 리샘플 (`distort_by_calibration`)
  2. cw90 회전으로 정립(upright) — 이미지와 캘리브를 **같이** 돌린다
  3. RGB/뎁스/라벨에 각각 맞는 보간(bilinear / bilinear-depth / nearest)

`rotate_camera_calib_cw90deg` 는 `T_device_camera` 까지 함께 회전시키므로, 회전 후
캘리브에서 꺼낸 외부 파라미터를 그대로 포즈 계산에 쓰면 된다.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from projectaria_tools.core import calibration as C

# 704×704, f=350 → 수평 화각 ≈ 90°. 원본 대각 110° 를 전부 담으려면 선형 모델에서
# 가장자리 배율이 폭발하므로, 화각을 조금 잘라 리샘플 품질을 지키는 쪽을 택했다.
DEFAULT_SIZE = 704
DEFAULT_FOCAL = 350.0


@dataclass
class LinearRig:
    """원본 어안 캘리브와, 정립된 선형 목표 캘리브의 쌍."""

    src: C.CameraCalibration          # fisheye624 (원본 VRS 프레임)
    dst_flat: C.CameraCalibration     # 선형, 회전 전 (리샘플 대상)
    dst: C.CameraCalibration          # 선형 + cw90 (최종 출력 프레임)
    width: int
    height: int
    focal: float

    @property
    def K(self) -> np.ndarray:
        fx, fy = self.dst.get_focal_lengths()
        cx, cy = self.dst.get_principal_point()
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def T_device_camera(self) -> np.ndarray:
        """회전까지 반영된 4×4 device←camera."""
        return self.dst.get_transform_device_camera().to_matrix()

    def hfov_deg(self) -> float:
        return float(2.0 * np.degrees(np.arctan(0.5 * self.width / self.focal)))

    # --- 리샘플 -----------------------------------------------------------------
    def _rot(self, a: np.ndarray) -> np.ndarray:
        # k=3 (= cw90). rotate_camera_calib_cw90deg 와 방향이 맞아야 한다.
        return np.ascontiguousarray(np.rot90(a, k=3))

    def rgb(self, img: np.ndarray) -> np.ndarray:
        return self._rot(C.distort_by_calibration(img, self.dst_flat, self.src))

    def depth(self, img: np.ndarray) -> np.ndarray:
        """uint16 mm 뎁스 — 경계에서 값이 섞이지 않는 뎁스 전용 보간."""
        return self._rot(C.distort_depth_by_calibration(img, self.dst_flat, self.src))

    def labels(self, img: np.ndarray) -> np.ndarray:
        """인스턴스 id 맵 — 최근접 이웃(값을 섞으면 없는 id 가 생긴다)."""
        return self._rot(C.distort_label_by_calibration(img, self.dst_flat, self.src))


def make_rig(
    src_calib: C.CameraCalibration,
    size: int = DEFAULT_SIZE,
    focal: float = DEFAULT_FOCAL,
) -> LinearRig:
    """어안 캘리브에서 정립 선형 리그를 만든다."""
    flat = C.get_linear_camera_calibration(
        size, size, focal, src_calib.get_label(), src_calib.get_transform_device_camera()
    )
    return LinearRig(
        src=src_calib,
        dst_flat=flat,
        dst=C.rotate_camera_calib_cw90deg(flat),
        width=size,
        height=size,
        focal=focal,
    )


def camera_info(rig: LinearRig, fps: float) -> dict:
    """DAAAM `ImageSequenceDataset` 가 읽는 camera_info.json 내용."""
    K = rig.K
    return {
        "width": rig.width,
        "height": rig.height,
        "fps": fps,
        "intrinsics": K.tolist(),
        "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "distortion_model": "none",
        "distortion_coefficients": [0.0] * 5,
        "source": "aria_rgb_fisheye624 -> linear(cw90)",
        "hfov_deg": rig.hfov_deg(),
        "T_device_camera": rig.T_device_camera.tolist(),
    }


def project(K: np.ndarray, pts_cam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """카메라 좌표 3D 점 → 픽셀. (uv, z) 반환. z<=0 은 호출측에서 걸러야 한다."""
    z = pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pts_cam[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    return uv, z
