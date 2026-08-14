"""ADT 시퀀스를 DAAAM 데이터셋으로 등록한다 — 포크 없이, 얇게.

`kx/adt/export.py` 가 이미 DAAAM 의 `image_sequence` 규약(`rgb/ depth/ pose/
camera_info.json`)대로 써 놓았으므로 사실상 그대로 읽힌다. 여기서 덧붙이는 것은 둘뿐:

  * **진짜 타임스탬프** — 기본 로더는 `timestamp = idx / fps` 로 균일 가정을 한다.
    ADT 는 프레임 드롭이 있어 실제 간격이 흔들리고, Khronos 의 액티브 윈도우는
    시간으로 잘리므로 `frames.json` 의 디바이스 타임스탬프를 그대로 쓴다.
  * **뎁스 단위** — 우리는 uint16 mm 로 쓰므로 `depth_scale=1000`.
"""
import json
import os

from daaam.datasets import ImageSequenceDataset
from daaam.datasets.factory import DatasetFactory


class AdtDataset(ImageSequenceDataset):
    def __init__(self, data_path, config=None):
        cfg = dict(config or {})
        cam_p = os.path.join(str(data_path), "camera_info.json")
        if os.path.exists(cam_p):
            cfg.setdefault("fps", json.load(open(cam_p)).get("fps", 10.0))
        super().__init__(data_path, cfg, depth_scale=1000.0)

        self._t0 = None
        self._ts = None
        fp = os.path.join(str(data_path), "frames.json")
        if os.path.exists(fp):
            fr = json.load(open(fp))["frames"]
            self._ts = [f["t_ns"] for f in fr]
            self._t0 = self._ts[0]

    def __getitem__(self, idx):
        frame = super().__getitem__(idx)
        if self._ts is not None and idx < len(self._ts):
            frame.timestamp = (self._ts[idx] - self._t0) / 1e9
        return frame


def register():
    DatasetFactory.register_dataset_type("adt", AdtDataset)
    return AdtDataset
