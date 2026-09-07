#!/usr/bin/env python3
"""스트리밍 재구성 모델(LingBot-Map · LoGeR) 어댑터 — 지도 프레임 → live 표본 순서로 한 스트림을 넣고 프레임별 카메라 포즈를
공통 raw 형식 {name, c:[x,y,z], f:[전방 단위벡터], u:[위 단위벡터]} 으로 저장한다 → `sfm_reloc.py --from-poses` 로 GT 정렬·평가(같은 표).
    python scripts/stream_recon_reloc.py <house> --backend lingbot --repo /path/lingbot-map --ckpt <ckpt> --live-step 10 --out ~/khcache/lingbot/raw_<h>.jsonl
    python scripts/stream_recon_reloc.py <house> --backend loger   --repo /path/LoGeR --ckpt ckpts/LoGeR/latest.pt --live-step 10 --out ...
모델별 포즈 추출 함수(`run_lingbot`, `run_loger`)는 각 리포지토리의 demo.py 를 보고 **표시된 두 곳만** 맞추면 된다 — 나머지(프레임 목록·표본·척도·출력·평가)는 공통.
포즈 규약: cam-to-world 4×4 (열 = 카메라 x,y,z 축, 마지막 열 = 카메라 중심). 모델이 world-to-cam 을 주면 역행렬로 바꿀 것(아래 `_c2w`)."""
import argparse, json, os, sys, time, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("house"); ap.add_argument("--backend", required=True, choices=["lingbot", "loger"]); ap.add_argument("--repo", required=True); ap.add_argument("--ckpt", default=None)
ap.add_argument("--out", default=None); ap.add_argument("--live-step", type=int, default=10); ap.add_argument("--map-max", type=int, default=0); ap.add_argument("--max-frames", type=int, default=4000)
ap.add_argument("--size", type=int, default=518, help="입력 긴 변(LingBot 518×378 · LoGeR 리포 기본값)"); ap.add_argument("--da-scale", type=int, default=0, help="1: DA 메트릭 깊이로 척도 보정(모델 출력이 메트릭이 아닐 때)")
ap.add_argument("--da-k", type=float, default=0.468); ap.add_argument("--da-model", default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
a = ap.parse_args(); hd = a.house.rstrip("/"); hn = os.path.basename(hd); T0 = time.time()
def log(*x): print("[%5.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)
sys.path.insert(0, a.repo); sys.path.insert(0, os.path.join(a.repo, "src"))
import torch; DEV = "cuda" if torch.cuda.is_available() else "cpu"
maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg")); lives = sorted(f for f in os.listdir(os.path.join(hd, "live")) if f.endswith(".jpg")); n_live0 = len(lives)
if a.map_max and len(maps) > a.map_max: maps = maps[:: int(np.ceil(len(maps) / a.map_max))][:a.map_max]
if a.live_step > 1: lives = lives[::a.live_step]
names = ["map/" + f for f in maps] + ["live/" + f for f in lives]
if len(names) > a.max_frames: log("⚠️ 프레임 %d > 상한 %d — 잘라냄" % (len(names), a.max_frames)); names = names[:a.max_frames]
paths = [os.path.join(hd, n) for n in names]; log("스트림 %d (지도 %d + live %d/%d) · %s" % (len(paths), len(maps), len(paths) - len(maps), n_live0, a.backend))
def _c2w(M):
    M = np.asarray(M, float).reshape(4, 4); return M
def _pack(c2w_list):
    C, F, U = [], [], []
    for M in c2w_list:
        M = _c2w(M); C.append(M[:3, 3]); F.append(M[:3, :3] @ np.array([0, 0, 1.0])); U.append(-(M[:3, :3] @ np.array([0, 1.0, 0])))
    return np.array(C), np.array(F), np.array(U)
def run_lingbot(plist):
    """LingBot-Map (github.com/Robbyant/lingbot-map): 스트리밍 추론 — 프레임을 순서대로 넣고 프레임별 cam-to-world 4×4 를 모은다.
    ★ 맞출 곳 1: 리포지토리 demo.py 의 모델 적재/스트리밍 호출 이름. 아래는 예상 형태이며 실제 API 에 맞춰 두 줄만 바꾼다."""
    from PIL import Image
    try:
        from gct import GCT  # noqa — 예상 모듈명(demo.py 참조)
        model = GCT.from_pretrained(a.ckpt or "robbyant/lingbot-map").to(DEV).eval()
    except Exception as e:
        raise SystemExit("LingBot-Map 적재 실패 — demo.py 의 import/적재 두 줄을 run_lingbot 에 옮겨라: %s" % e)
    poses = []
    with torch.no_grad():
        state = model.init_state() if hasattr(model, "init_state") else None           # ★ 맞출 곳 2: 스트리밍 상태·프레임 입력·포즈 키
        for p in plist:
            out = model.stream(Image.open(p).convert("RGB"), state=state) if state is not None else model.infer_frame(Image.open(p).convert("RGB"))
            pose = out["camera_pose"] if isinstance(out, dict) else out
            poses.append(np.asarray(pose.detach().cpu().numpy() if torch.is_tensor(pose) else pose).reshape(4, 4))
    return poses
def run_loger(plist):
    """LoGeR (github.com/Junyi42/LoGeR): 청크 단위 스트리밍 + 하이브리드 메모리 — 리포지토리 demo 의 '이미지 목록 → per-frame pose' 경로를 그대로 쓴다.
    ★ 맞출 곳: 적재 함수와 추론 호출 이름(예: load_model(ckpt), inference(images) → dict(camera_poses=(N,4,4)))."""
    try:
        from loger.model import load_model   # noqa — 예상 모듈명
        model = load_model(a.ckpt).to(DEV).eval()
        from loger.inference import run_sequence  # noqa
    except Exception as e:
        raise SystemExit("LoGeR 적재 실패 — 리포지토리 demo 의 import/적재를 run_loger 에 옮겨라: %s" % e)
    with torch.no_grad(): out = run_sequence(model, plist, size=a.size)
    P = out["camera_poses"] if isinstance(out, dict) else out
    P = P.detach().cpu().numpy() if torch.is_tensor(P) else np.asarray(P)
    return [P[k].reshape(4, 4) for k in range(len(plist))]
poses = run_lingbot(paths) if a.backend == "lingbot" else run_loger(paths)
C, F, U = _pack(poses); log("스트림 완료 · 포즈 %d · 중심 산포 %.2f" % (len(C), float(np.std(C))))
scale = 1.0
if a.da_scale:
    # 모델 출력이 메트릭이 아니면 DA 메트릭 깊이와의 비율로 척도를 맞춘다(CUT3R 어댑터와 같은 방식). 대표깊이가 없으니 지도 프레임 간 기선을 쓴다: DA 는 프레임별 절대 깊이 → 여기서는 생략 가능
    log("da-scale: 스트리밍 모델은 대표깊이를 안 주므로 척도는 정렬(--scale gt) 진단으로 읽는다")
out = a.out or os.path.join(os.path.dirname(hd), "raw_%s_%s.jsonl" % (a.backend, hn)); os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as fo:
    for nm, c, f, u in zip(names, C * scale, F, U):
        fo.write(json.dumps(dict(name=nm, c=[round(float(v), 4) for v in c], f=[round(float(v), 4) for v in f], u=[round(float(v), 4) for v in u])) + "\n")
log("→ %s · 지도 %d · live %d/%d" % (out, len(maps), len(names) - len(maps), n_live0))
