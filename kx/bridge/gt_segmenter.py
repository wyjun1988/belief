"""ADT GT 인스턴스 마스크를 DAAAM 세그멘테이션 자리에 끼워 넣는다.

1차에서 SAM 을 쓰지 않는 이유: 지각 오차와 기하/동역학 오차가 섞이면 "윈도우 간
뎁스 정합이 씬그래프에 얼마나 기여했는가"를 분리할 수 없다. GT 마스크로 세그멘테이션을
고정해 놓고 뎁스만 바꿔 재는 것이 이번 실험의 설계다. (SAM 교체는 P4.)

DAAAM 포크는 하지 않는다. `SegmenterInterface` 만 구현하고, 실행 스크립트가
`SegmentationService._initialize_segmenter` 를 이 구현으로 갈아끼운다.

**프레임 식별**: 인터페이스가 이미지 한 장만 넘겨주므로(`__call__(source)`) 어떤
프레임인지 알 수 없다. 호출 순서를 세는 방식은 워밍업의 더미 이미지 한 번에 전부
어긋난다. 그래서 32×32 썸네일 해시로 내용을 보고 프레임을 찾는다 — 못 찾으면
빈 결과를 돌려주므로 워밍업은 조용히 통과한다.
"""
import json
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    from daaam.segmentation.interfaces import SegmenterInterface
except Exception:                       # 로컬(맥)에서 임포트만 확인할 때
    class SegmenterInterface:           # type: ignore
        pass


def thumb_hash(img: np.ndarray) -> bytes:
    a = np.asarray(Image.fromarray(img).convert("L").resize((32, 32), Image.BILINEAR))
    return a.tobytes()


class GTMaskSegmenter(SegmenterInterface):
    def __init__(self, seq_dir: str, min_area: int = 200, max_extent_m: float = None,
                 mode: str = "masks_only", logger=None):
        self.seq_dir = seq_dir
        self.min_area = min_area
        self.mode = mode
        self.logger = logger
        self.seg_dir = os.path.join(seq_dir, "gt", "seg")
        self.ids = json.load(open(os.path.join(seq_dir, "gt", "seg_ids.json")))

        # 벽·바닥 같은 구조물은 물체 노드로 올리지 않는다 (Khronos 는 배경을
        # 볼류메트릭 맵으로 따로 다룬다). GT extent 로 거른다.
        self.skip = set()
        if max_extent_m:
            gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
            for local, meta in self.ids.items():
                rec = gt.get(str(meta["instance_id"]))
                if rec and rec.get("extent_m") and max(rec["extent_m"]) > max_extent_m:
                    self.skip.add(int(local))

        self._index = {}                # 썸네일 해시 → 프레임 번호
        rgb_dir = os.path.join(seq_dir, "rgb")
        for f in sorted(os.listdir(rgb_dir)):
            i = int(os.path.splitext(f)[0])
            self._index[thumb_hash(np.array(Image.open(os.path.join(rgb_dir, f))))] = i
        self.misses = 0

    def initialize(self, model_path: str = None, config_path: str = None, device: str = None):
        return None

    def __call__(self, source: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        i = self._index.get(thumb_hash(source))
        if i is None:
            self.misses += 1
            return np.empty((0, 6), dtype=np.float32), []
        p = os.path.join(self.seg_dir, "%06d.png" % i)
        if not os.path.exists(p):
            return np.empty((0, 6), dtype=np.float32), []

        seg = np.array(Image.open(p))
        dets, masks = [], []
        for lid in np.unique(seg):
            if lid == 0 or int(lid) in self.skip:
                continue
            m = seg == lid
            if m.sum() < self.min_area:
                continue
            ys, xs = np.nonzero(m)
            # cls 자리에 GT 인스턴스 로컬 id 를 넣는다. masks_only 모드에서 BotSort 는
            # 이걸 클래스로만 쓰고 연관은 스스로 하므로 추적 성능은 GT 가 아니다.
            dets.append([xs.min(), ys.min(), xs.max(), ys.max(), 1.0, float(lid)])
            masks.append(m)
        return np.asarray(dets, dtype=np.float32).reshape(-1, 6), masks


def patch_daaam(seq_dir: str, **kw):
    """`SegmentationService` 가 GT 마스크를 쓰도록 갈아끼운다. 실행 스크립트에서 호출."""
    from daaam.segmentation import services as S

    seg = GTMaskSegmenter(seq_dir, **kw)

    def _init(self):
        self.segmenter = seg
        self.logger.info("GTMaskSegmenter 주입: %s (%d 인스턴스, %d 제외)"
                         % (seq_dir, len(seg.ids), len(seg.skip)))

    S.SegmentationService._initialize_segmenter = _init
    return seg
