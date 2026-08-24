# AI2-THOR / ProcTHOR 셋업 — 재구축 가이드

MBP(M1 Pro, arm64) 에서 쓰던 구성. **매각 대비로 남긴다.**
Unity 빌드(`~/.ai2thor`, 1 GB)와 venv 는 **아키텍처별 바이너리라 옮겨봐야 못 쓴다.**
아래대로 새 기기에서 다시 만든다.

## 1. 환경

| | 값 |
|---|---|
| Python | 3.9.6 (시스템 python3 그대로) |
| venv | `~/thor-venv` (생성·시뮬레이션 전용), `~/khronos-venv` (OWL/CLIP 추론) |

**venv 를 둘로 나눈 이유.** `ai2thor` 가 `numpy` 를 강하게 묶어서 torch 계열과
같이 두면 충돌한다. 생성과 추론을 분리하면 각각 최신을 쓸 수 있다.

```bash
python3 -m venv ~/thor-venv
~/thor-venv/bin/pip install ai2thor==5.0.0 prior==1.0.3 numpy==2.0.2 pillow==11.3.0

python3 -m venv ~/khronos-venv
~/khronos-venv/bin/pip install torch==2.8.0 torchvision==0.23.0 \
    transformers==4.57.6 numpy==2.0.2 pillow==11.3.0 scipy==1.13.1
```

첫 `Controller()` 호출에서 Unity 빌드를 자동 내려받는다(1 GB, `~/.ai2thor/releases`).
빌드 이름에 아키텍처가 박힌다 — 실측 `thor-OSXIntel64-f0825767…`.

## 2. ProcTHOR-10K 집 데이터

```python
import prior
ds = prior.load_dataset("procthor-10k")["train"]   # 10,000채, 첫 호출에 캐시
h = ds[hi]                                          # dict 를 Controller(scene=h) 로
```

⚠️ `ai2thor` 5.0+ 경고가 뜨지만 동작한다. 구버전을 쓰려면
`prior.load_dataset("procthor-10k", revision="ab3cacd0fc17754d4c080a3fd50b18395fae8647")`.

**방 폴리곤은 `gt.json` 이 아니라 집 dict 에 있다** — `h["rooms"][i]["floorPolygon"]`.
`ds[hi]` 를 다시 불러야 얻는다(우리 `gt.json` 은 인덱스만 저장한다).
문 연결은 `h["doors"]` 의 `room0`/`room1`.

## 3. **반드시 설정해야 하는 것 — 이걸로 한 번 크게 물렸다**

```python
Controller(scene=h, width=384, height=384, quality="Low",
           visibilityDistance=20.0,          # ★ 기본 1.5m 는 '보인다'를 거리로 자른다
           renderInstanceSegmentation=True)  # ★ instance_detections2D (bbox) 에 필요
```

### `visibilityDistance` (기본 1.5 m)

`object["visible"]` 은 시야각·가림뿐 아니라 **거리로도 잘린다.** 실측:

```
1.5  → pickupable 52개 중 visible=True **0개** (가장 가까운 것이 1.6m)
20.0 → 13개 (최대 5.7m)
```

기본값으로 GT 를 만들면 방 건너편에 뻔히 보이는 물체가 전부 "안 보임" 이 되고,
검출기가 그걸 잡을 때마다 오검출로 세어진다. 물체당 "보인" 프레임이 6장 대 122장
으로 20배 차이났다. **§58 참조 — 이 오염으로 여러 결론이 뒤집혔다.**

### `renderInstanceSegmentation`

`event.instance_detections2D[objectId] → [x0,y0,x1,y1]`. 렌더 비용이 붙지만
bbox 없이는 exemplar crop(§63)도 앵커 화면위치(§60)도 못 만든다.

## 4. 자주 쓰는 호출

```python
e = ctrl.step("Teleport", position=p, rotation=dict(x=0,y=yaw,z=0), horizon=10)
e.metadata["lastActionSuccess"]                      # 실패하면 그 프레임은 버린다
ctrl.step("GetReachablePositions").metadata["actionReturn"]
ctrl.step("PlaceObjectAtPoint", objectId=oid, position=dict(x=..,y=y+0.6,z=..))
e.metadata["objects"]      # objectId/objectType/position/visible/pickupable
e.metadata["agent"]["position"]
e.frame                    # numpy RGB
```

`pickupable` 로 타겟(움직이는 물체)과 앵커(정적 물체)를 가른다.

## 5. 방 판정 — shapely 없이

`shapely` 가 없는 환경이 많다. ray casting 으로 충분하다:

```python
def inside(x, z, pts):
    c = False; n = len(pts)
    for i in range(n):
        x1, z1 = pts[i]; x2, z2 = pts[(i+1) % n]
        if (z1 > z) != (z2 > z) and x < (x2-x1)*(z-z1)/(z2-z1+1e-12) + x1:
            c = not c
    return c
```

⚠️ **오픈플랜이라 "에이전트가 선 방" 은 "물체의 방" 이 아니다**(일치 0.257).
물체 위치를 폴리곤에 넣어야 한다. §59 참조.

## 6. 한국 주거에 맞춘 집 선별

ProcTHOR 에는 방 10개 넘는 집이 섞여 있고, 그런 집이 들어가면 후보 방이 많아져
국소화 천장이 무너진다(3주택 0.897 → 20주택 0.621). `thor_gen2.py` 플래그:

```
--min-rooms 4 --max-rooms 8 --max-nonbath 6 --min-nonbath 2
```

한국 아파트 = 방 3개 + 거실 + 주방 ≈ 5개.

## 7. 리눅스 GPU 로 옮길 때 (4090/H100)

헤드리스에는 Unity 창이 없다. **CloudRendering 플랫폼**이 필요하다:

```python
from ai2thor.platform import CloudRendering
Controller(platform=CloudRendering, scene=h, ...)
```

`ai2thor` 가 CloudRendering 전용 빌드를 따로 내려받는다. Vulkan 드라이버가 필요하니
`nvidia-smi` 만 되는 게 아니라 `vulkaninfo` 도 확인할 것.

속도 비교(실측/추정): 20주택 × 1시간 × 1fps 생성이 MBP 53분, 4090 ≈ 25분.
OWL 캐시는 MBP 62분 대 4090 ≈ 5분으로 **10배 이상** 차이가 크다.

## 8. 우리 스크립트

| 스크립트 | 역할 |
|---|---|
| `thor_gen2.py` | 집 선별 → 맵 촬영 → 1fps 배회 → 미관측 이동 |
| `thor_prior_llm.py` | Qwen 으로 **배치** 사전확률 (`thor_prior.json`) |
| `thor_move_llm.py` | Qwen 으로 **움직임** 사전확률 (`thor_move.json`) |
| `thor2_owl.py` | OWLv2/CLIP 프레임 캐시 |

⚠️ 사전확률을 넣었으면 **생성된 데이터에서 그 분포를 되재라.** 유형 확률을 방
인스턴스마다 그대로 주면 유형 분포가 깨진다(거실 0.40 의도 → 실측 0.22). §67 참조.
