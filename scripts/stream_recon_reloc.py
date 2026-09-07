#!/usr/bin/env python3
"""스트리밍 재구성 모델(LingBot-Map · LoGeR) 어댑터 — 지도 프레임 → live 표본 순서로 한 스트림을 넣고 프레임별 카메라 포즈를
공통 raw 형식 {name, c:[x,y,z], f:[전방 단위벡터], u:[위 단위벡터]} 으로 저장 → `sfm_reloc.py --from-poses` 로 GT 정렬·평가(VGGT·CUT3R 와 같은 표).
    python scripts/stream_recon_reloc.py <house> --backend lingbot --repo /path/lingbot-map --ckpt /path/lingbot-map.pt --live-step 10 --out ~/khcache/stream/raw_lingbot_<h>.jsonl
    python scripts/stream_recon_reloc.py <house> --backend loger   --repo /path/LoGeR       --ckpt /path/LoGeR/ckpts/LoGeR_star/latest.pt --live-step 10 --out ...
API 는 두 리포지토리의 demo(2026-09 main)를 그대로 옮겼다(2026-09-07, M2 에서 코드만 읽고 작성 — CUDA 없어 미실행):
  · LingBot-Map: demo.py load_model/load_images/postprocess — GCTStream(...) + load_and_preprocess_images(mode="crop") + inference_streaming(...) → pose_enc → extri(w2c) → 역행렬 = c2w
  · LoGeR: demo_viser.py load_pi3_model/load_images_from_paths — Pi3(**config.model 중 __init__ 인자) + latest.pt → model(imgs[None], window_size, overlap_size, se3, ...) → out["camera_poses"] = Twc (N,4,4)
포즈 규약: cam-to-world 4×4, 카메라 x=우 y=하 z=전방(OpenCV). f = R[:, 2], u = −R[:, 1]."""
import argparse, json, os, sys, time, math, inspect, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("house"); ap.add_argument("--backend", required=True, choices=["lingbot", "loger"]); ap.add_argument("--repo", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default=None); ap.add_argument("--live-step", type=int, default=10); ap.add_argument("--map-max", type=int, default=0); ap.add_argument("--max-frames", type=int, default=4000)
ap.add_argument("--size", type=int, default=518, help="LingBot 입력 긴 변(demo 기본 518)"); ap.add_argument("--lb-mode", default="streaming", choices=["streaming", "windowed"], help="LingBot: 프레임 >1024 면 windowed")
ap.add_argument("--lb-keyframe", type=int, default=1, help="LingBot keyframe_interval (긴 스트림이면 2~4 로 KV 절약)"); ap.add_argument("--lb-sdpa", action="store_true", help="FlashInfer 없이 SDPA")
ap.add_argument("--lg-config", default=None, help="LoGeR original_config.yaml (기본: ckpt 옆)"); ap.add_argument("--lg-window", type=int, default=32); ap.add_argument("--lg-overlap", type=int, default=3)
ap.add_argument("--lg-res", default="auto", help="LoGeR 입력 해상도 'W,H'(14 배수) 또는 auto(픽셀 25.5만 상한)"); ap.add_argument("--fp32", action="store_true")
a = ap.parse_args(); hd = a.house.rstrip("/"); hn = os.path.basename(hd); T0 = time.time()
def log(*x): print("[%5.0fs] " % (time.time() - T0) + " ".join(str(v) for v in x), flush=True)
sys.path.insert(0, a.repo)
import torch; DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32 if (a.fp32 or DEV == "cpu") else (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
maps = sorted(f for f in os.listdir(os.path.join(hd, "map")) if f.endswith(".jpg")); lives = sorted(f for f in os.listdir(os.path.join(hd, "live")) if f.endswith(".jpg")); n_live0 = len(lives)
if a.map_max and len(maps) > a.map_max: maps = maps[:: int(np.ceil(len(maps) / a.map_max))][:a.map_max]
if a.live_step > 1: lives = lives[::a.live_step]
names = ["map/" + f for f in maps] + ["live/" + f for f in lives]
if len(names) > a.max_frames: log("⚠️ 프레임 %d > 상한 %d — 잘라냄" % (len(names), a.max_frames)); names = names[:a.max_frames]
paths = [os.path.join(hd, n) for n in names]; log("스트림 %d (지도 %d + live %d/%d) · %s · %s %s" % (len(paths), len(maps), len(paths) - len(maps), n_live0, a.backend, DEV, DT))

def run_lingbot(plist):
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    if a.lb_mode == "windowed": from lingbot_map.models.gct_stream_window import GCTStream
    else: from lingbot_map.models.gct_stream import GCTStream
    model = GCTStream(img_size=a.size, patch_size=14, enable_3d_rope=True, max_frame_num=max(1024, len(plist) + 16), kv_cache_sliding_window=64, kv_cache_scale_frames=8,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True, use_sdpa=a.lb_sdpa, camera_num_iterations=4)
    ckpt = torch.load(a.ckpt, map_location=DEV, weights_only=False); sd = ckpt.get("model", ckpt); miss, unexp = model.load_state_dict(sd, strict=False)
    log("LingBot-Map 적재 · missing %d · unexpected %d" % (len(miss), len(unexp))); model = model.to(DEV).eval()
    if DT != torch.float32 and getattr(model, "aggregator", None) is not None: model.aggregator = model.aggregator.to(dtype=DT)   # demo 와 같이: 트렁크만 저정밀
    images = load_and_preprocess_images(plist, mode="crop", image_size=a.size, patch_size=14).to(DEV); log("전처리 %s" % (tuple(images.shape),))
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DT, enabled=(DEV == "cuda")):
        if a.lb_mode == "streaming": preds = model.inference_streaming(images, num_scale_frames=8, keyframe_interval=a.lb_keyframe, output_device=torch.device("cpu"))
        else: preds = model.inference_windowed(images, window_size=64, overlap_size=16, overlap_keyframes=None, num_scale_frames=8, keyframe_interval=a.lb_keyframe, output_device=torch.device("cpu"))
    pe = preds["pose_enc"]; pe = pe if pe.dim() == 3 else pe[None]                       # [B,S,9]
    extri, intri = pose_encoding_to_extri_intri(pe.float(), images.shape[-2:])       # w2c [B,S,3,4]
    E4 = torch.zeros((*extri.shape[:-2], 4, 4), dtype=extri.dtype); E4[..., :3, :4] = extri; E4[..., 3, 3] = 1.0
    c2w = closed_form_inverse_se3_general(E4)[0].cpu().numpy()                          # [S,4,4]
    log("LingBot 스트림 완료 · 프레임 유형 %s" % (np.bincount(preds["frame_type"].reshape(-1).cpu().numpy().astype(int)).tolist() if "frame_type" in preds else "—"))
    return [c2w[k] for k in range(len(plist))]

def run_loger(plist):
    import yaml
    from PIL import Image
    from torchvision import transforms
    from loger.models.pi3 import Pi3
    cfgp = a.lg_config or os.path.join(os.path.dirname(a.ckpt), "original_config.yaml"); cfg = yaml.safe_load(open(cfgp)) if os.path.exists(cfgp) else {}
    mcfg = cfg.get("model", {}) or {}; sig = inspect.signature(Pi3.__init__).parameters
    kwargs = {k: mcfg[k] for k in sig if k in mcfg}; model = Pi3(**kwargs)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); sd = ck.get("model_state_dict", ck); sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    try: model.load_state_dict(sd, strict=True)
    except Exception as e: log("strict 적재 실패 → strict=False: %s" % str(e)[:120]); model.load_state_dict(sd, strict=False)
    model = model.to(DEV).eval(); log("LoGeR 적재 · %s · model 인자 %s" % (os.path.basename(cfgp), list(kwargs)))
    # 이미지: demo load_images_from_paths 와 동일 (14 배수, 픽셀 25.5만 상한 또는 지정)
    if a.lg_res != "auto": TW, TH = [int(v) for v in a.lg_res.split(",")]
    else:
        W0, H0 = Image.open(plist[0]).size; sc = math.sqrt(255000 / (W0 * H0)); k, m = round(W0 * sc / 14), round(H0 * sc / 14)
        while (k * 14) * (m * 14) > 255000:
            if k / m > W0 / H0: k -= 1
            else: m -= 1
        TW, TH = max(1, k) * 14, max(1, m) * 14
    tt = transforms.ToTensor(); imgs = torch.empty((len(plist), 3, TH, TW), dtype=torch.float32)
    for i, p in enumerate(plist): imgs[i].copy_(tt(Image.open(p).convert("RGB").resize((TW, TH), Image.Resampling.LANCZOS)))
    log("전처리 %s" % (tuple(imgs.shape),))
    ts_ = cfg.get("training_settings", {}) or {}
    fk = dict(window_size=a.lg_window, overlap_size=a.lg_overlap, reset_every=ts_.get("reset_every", 0), num_iterations=cfg.get("num_iterations", 1), sim3=bool(cfg.get("sim3", False)),
              sim3_scale_mode="median", se3=bool(mcfg.get("se3", cfg.get("se3", False))), turn_off_ttt=False, turn_off_swa=False)
    log("forward 인자 %s" % fk)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=DT, enabled=(DEV == "cuda")):
        out = model(imgs[None].to(DEV), **fk)
    P = out["camera_poses"]; P = (P[0] if P.dim() == 4 else P).float().cpu().numpy()      # Twc (N,4,4)
    log("LoGeR 스트림 완료 · 포즈 %s" % (P.shape,))
    return [P[k] for k in range(len(plist))]

poses = run_lingbot(paths) if a.backend == "lingbot" else run_loger(paths)
C, F, U = [], [], []
for M in poses:
    M = np.asarray(M, float).reshape(4, 4); R = M[:3, :3]; C.append(M[:3, 3]); F.append(R @ np.array([0, 0, 1.0])); U.append(-(R @ np.array([0, 1.0, 0])))
C, F, U = np.array(C), np.array(F), np.array(U); log("중심 산포 %.2f · 지도 궤적 크기 %.1f" % (float(np.std(C)), float(np.ptp(C[:len(maps)], axis=0).max())))
out = a.out or os.path.join(os.path.dirname(hd), "raw_%s_%s.jsonl" % (a.backend, hn)); os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as fo:
    for nm, c, f, u in zip(names, C, F, U):
        fo.write(json.dumps(dict(name=nm, c=[round(float(v), 4) for v in c], f=[round(float(v), 4) for v in f], u=[round(float(v), 4) for v in u])) + "\n")
log("→ %s · 지도 %d · live %d/%d" % (out, len(maps), len(names) - len(maps), n_live0))
