#!/usr/bin/env python3
"""CUT3R 재국소화 어댑터 — **하나의 세션**에 매핑워크 → live 를 순서대로 흘려 넣는다.
CUT3R 은 영속 상태(persistent state)에 프레임을 하나씩 더하며 모든 프레임을 **같은 월드 좌표계**로 낸다.
따라서 VGGT 처럼 부분 재구성을 이어 붙일 필요가 없다 — 우리 구조(지도 1회 + live 프레임별 등록)와 원래 맞는 모델.

    python scripts/cut3r_reloc.py <house> --cut3r-root /path/to/CUT3R --model-path /path/to/cut3r_512_dpt_4_64.pth \
        --out raw_<house>.jsonl --live-step 4
    python scripts/sfm_reloc.py <house> --from-poses raw_<house>.jsonl --scale da --align sites --out pose_<house>.jsonl

출력: {"name": "map/….jpg"|"live/….jpg", "c":[x,y,z], "f":[x,y,z]} jsonl (COLMAP·VGGT 와 같은 형식 → 같은 표).
좌표계: 첫 프레임 카메라 기준 월드(CUT3R 규약). 척도: CUT3R 은 메트릭을 지향하지만 렌더 도메인에선 어긋날 수 있어
`--da-scale` 로 DA-V2 metric 깊이 비율(§146 원리)을 곱한다(기본 ON).
⚠️ CUT3R 리포지토리의 함수 이름은 버전에 따라 다르다. `_load_cut3r()` 안의 세 import 와 `_run_stream()` 의 결과 키
(`camera_pose`, `pts3d_in_other_view`) 만 그 버전의 demo.py 에 맞추면 된다 — 나머지는 건드릴 필요 없다.
"""
import argparse, json, os, sys, time
import numpy as np, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("house"); ap.add_argument("--out", default=None)
ap.add_argument("--cut3r-root", required=True, help="git clone 한 CUT3R 디렉터리 (src/ 가 있는 곳)")
ap.add_argument("--model-path", required=True, help="체크포인트 .pth (cut3r_512_dpt_4_64.pth 권장)")
ap.add_argument("--size", type=int, default=512, help="입력 크기(512 또는 224)")
ap.add_argument("--query-frames", default=None, help="json {house:[t,…]} — 이 프레임들만 세션 뒤에 추가(scripts/query_frames.py)")
ap.add_argument("--live-step", type=int, default=4)
ap.add_argument("--map-max", type=int, default=0, help="0=지도 전부")
ap.add_argument("--max-frames", type=int, default=2000, help="한 세션에 넣는 총 프레임 상한(메모리 보호)")
ap.add_argument("--da-scale", type=int, default=1, help="1=DA-V2 metric 깊이 비율로 척도 보정")
ap.add_argument("--da-model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
ap.add_argument("--selfcheck", action="store_true", help="지도만 넣은 세션 vs 지도+live 세션에서 지도 포즈가 같은가(상태 안정성)")
a = ap.parse_args()
hd = a.house.rstrip("/"); hn = os.path.basename(hd); T0 = time.time()
def log(*x): print("[%6.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

sys.path.insert(0, a.cut3r_root); sys.path.insert(0, os.path.join(a.cut3r_root, "src"))
def _load_cut3r():
    """버전별 import 지점 — 실패하면 리포지토리 demo.py 의 import 세 줄로 바꿔 달라."""
    try:
        from dust3r.model import ARCroco3DStereo          # CUT3R 의 모델 클래스
        from dust3r.inference import inference             # 순차 추론(영속 상태)
        from dust3r.utils.image import load_images         # 전처리(리사이즈·정규화)
    except ImportError:
        from src.dust3r.model import ARCroco3DStereo
        from src.dust3r.inference import inference
        from src.dust3r.utils.image import load_images
    return ARCroco3DStereo, inference, load_images
ARCroco3DStereo, inference, load_images = _load_cut3r()
model = ARCroco3DStereo.from_pretrained(a.model_path).to(DEV).eval()
log("CUT3R 적재 · %s · size %d" % (DEV, a.size))

maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg"))
lives = sorted(f for f in os.listdir(os.path.join(hd, "live")) if f.endswith(".jpg"))
n_live0 = len(lives)
if a.map_max and len(maps) > a.map_max: maps = maps[:: int(np.ceil(len(maps) / a.map_max))][:a.map_max]
if a.query_frames:
    _q = set("%06d.jpg" % t for t in json.load(open(a.query_frames)).get(hn, [])); lives = [f for f in lives if f in _q]
    log("질의 프레임 모드: %d/%d" % (len(lives), n_live0))
elif a.live_step > 1: lives = lives[::a.live_step]
names = ["map/" + f for f in maps] + ["live/" + f for f in lives]
if len(names) > a.max_frames:
    log("⚠️ 프레임 %d > 상한 %d — live 를 더 성기게(--live-step) 하거나 --max-frames 를 올려라" % (len(names), a.max_frames)); names = names[:a.max_frames]
paths = [os.path.join(hd, n) for n in names]
log("세션 프레임 %d (지도 %d + live %d/%d)" % (len(paths), len(maps), len(paths) - len(maps), n_live0))

def _run_stream(plist):
    """한 세션: 프레임을 순서대로 넣고 (중심, 전방, 대표깊이) 를 돌려준다. 결과 키는 버전에 맞출 것."""
    views = load_images(plist, size=a.size, verbose=False)
    for k, v in enumerate(views): v["idx"] = k; v["instance"] = str(k)
    with torch.no_grad():
        out = inference(views, model, DEV, dtype=torch.float32, verbose=False)
    preds = out["pred"] if isinstance(out, dict) and "pred" in out else out
    C, F, Dm, U_ = [], [], [], []
    for p_ in preds:
        pose = np.asarray((p_["camera_pose"] if "camera_pose" in p_ else p_["cam_pose"]).squeeze().detach().cpu().numpy() if torch.is_tensor(p_.get("camera_pose", p_.get("cam_pose"))) else p_.get("camera_pose", p_.get("cam_pose"))).reshape(4, 4)
        C.append(pose[:3, 3]); F.append(pose[:3, :3] @ np.array([0, 0, 1.0])); U_.append(-(pose[:3, :3] @ np.array([0, 1.0, 0])))
        pts = p_.get("pts3d_in_other_view", p_.get("pts3d"))
        if pts is not None:
            pts = pts.detach().cpu().numpy() if torch.is_tensor(pts) else np.asarray(pts)
            Dm.append(float(np.nanmedian(np.linalg.norm(pts.reshape(-1, 3) - pose[:3, 3], axis=1))))
        else: Dm.append(np.nan)
    _run_stream.last_U = np.array(U_)
    return np.array(C), np.array(F), np.array(Dm)

if a.selfcheck:
    Cmo, _, _ = _run_stream(paths[:len(maps)])
    Call, _, _ = _run_stream(paths[:len(maps)] + paths[len(maps):len(maps) + 16])
    d = np.linalg.norm(Cmo - Call[:len(maps)], axis=1)
    log("자가검사(상태 안정성): 지도 %d장 포즈가 live 16장 추가 뒤에도 같은가 — 이동 중앙 %.4f m · 최대 %.4f m (0 에 가까워야 정상)" % (len(maps), float(np.median(d)), float(d.max())))
    log("→ 크면 세션이 live 를 넣을 때 지도까지 다시 쓴다는 뜻 — 그 경우 CUT3R 의 상태 고정(freeze) 옵션을 찾아 켜야 한다.")
    sys.exit(0)

C, F, Dm = _run_stream(paths); U = _run_stream.last_U
log("세션 완료 · 중심 산포 %.2f · 대표깊이 중앙 %.2f" % (float(np.std(C)), float(np.nanmedian(Dm))))
scale = 1.0
if a.da_scale and np.isfinite(Dm).any():
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    dp = AutoImageProcessor.from_pretrained(a.da_model); dm = AutoModelForDepthEstimation.from_pretrained(a.da_model).to(DEV).eval()
    rat = []
    for k in range(0, len(maps), max(1, len(maps) // 24)):
        if not np.isfinite(Dm[k]): continue
        img = Image.open(paths[k]).convert("RGB")
        with torch.no_grad(): dd = dm(**dp(images=img, return_tensors="pt").to(DEV)).predicted_depth
        rat.append(float(np.median(dd.float().cpu().numpy())) / Dm[k])
    if rat: scale = float(np.median(rat)); log("DA 척도 비율 %.3f (프레임 %d · 산포 %.2f) — CUT3R 이 메트릭이면 ≈1" % (scale, len(rat), float(np.std(rat) / (np.median(rat) + 1e-9))))
out = a.out or os.path.join(os.path.dirname(hd), "raw_cut3r_%s.jsonl" % hn)
with open(out, "w") as fo:
    for nm, c, f, u in zip(names, C * scale, F, U):
        fo.write(json.dumps(dict(name=nm, c=[round(float(v), 4) for v in c], f=[round(float(v), 4) for v in f], u=[round(float(v), 4) for v in u])) + "\n")
log("→ %s · 지도 %d · live %d/%d" % (out, len(maps), len(names) - len(maps), n_live0))
