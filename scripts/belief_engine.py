#!/usr/bin/env python3
"""v3 belief 엔진 — "지금 안 보이는 그 물건, 어디 있나" 를 확률 순위로.

    $P scripts/belief_engine.py --seq <adt-seq> --obj Clock          # 단건 질의
    $P scripts/belief_engine.py --seq <adt-seq> --eval               # 이동물체 채점

구성(v3 확정 설계):
  신호 A  home-jepa `two_head_v5` 분포 — last-known 앵커 + 루틴 + 게이트 readout 을
          학습한 모델의 receptacle 확률 (원 belief 설계의 본체를 그대로 재사용)
  신호 B  부재 증거 — 후보 가구의 **배제 질의**(가구·방 이름만, keyword 제외)로 최근
          프레임을 찾고, 봤는데 keyword 가 없으면 그 후보를 감쇠(γ), 보이면 부스트(1/γ)
  출력    후보(가구, 방) 확률 순위 + 근거 태그

부재 게이트는 '봤다' 가 전제다(문맥 z ≥ gate, 실측 교훈: 없으면 미방문을 부재로 오판).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HJ = os.path.expanduser("~/work/home-jepa")
sys.path.insert(0, HJ)
sys.path.insert(0, os.path.join(HJ, "scripts"))

from kx.eval.homejepa_export import build_episode      # noqa: E402
from kx.eval.room_belief import load_regions           # noqa: E402
from kx.graph.regions import assign                    # noqa: E402

FPS = 10          # ADT 내보내기 10Hz
TICK_S = 5        # 에피소드 틱 = 5초


def camel_words(s):
    return " ".join(re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?![a-z])", s)).lower()


def clip_scores(sd, texts, device="mps"):
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    z = np.load(os.path.join(sd, "clip_frames.npz"))
    E, fidx = z["emb"].astype(np.float32), z["idx"]
    nm = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(nm)
    txt = CLIPTextModelWithProjection.from_pretrained(nm, use_safetensors=True).eval().to(device)
    with torch.no_grad():
        tt = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        V = torch.nn.functional.normalize(txt(**tt).text_embeds, dim=-1).cpu().numpy()
    S = V @ E.T
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    return Z, fidx


def absence_gate(Z, fidx, kw_i, ctx_i, t_frame, win_s=60, gate=1.0, topf=8,
                 min_frames=3, present_z=0.5):
    """질의 직전 win_s 초에서 배제 질의로 후보 장소를 보고, keyword 존재를 판정.

    반환: +1(목격) · -1(봤는데 없음=부재) · 0(미방문=판정 보류)"""
    lo = t_frame - win_s * FPS
    m = (fidx >= lo) & (fidx <= t_frame)
    fr = np.nonzero(m)[0]
    if len(fr) < min_frames:
        return 0, None
    cs = Z[ctx_i][:, fr].max(0)
    ok = fr[cs >= gate]
    if len(ok) < min_frames:
        return 0, None
    sel = ok[np.argsort(-Z[ctx_i][:, ok].max(0))[:topf]]
    pk = float(np.median(Z[kw_i][:, sel].max(0)))
    return (1 if pk >= present_z else -1), pk


def extend_class_vocab(model, new_names, device):
    """**열린 어휘의 정공법** — CLIP 임베딩을 클래스 속성으로 주입한다.

    home-jepa 의 e_cls 는 10클래스+미지의 룩업 테이블이라 새 물체가 오면 사전을
    잃는다(미지 슬롯). 여기서는 CLIP 텍스트 공간 → 모델 클래스 임베딩 공간의
    선형 사상 W 를 **알려진 10클래스를 앵커로** 최소자승으로 맞추고, 새 이름에
    W·clip(name) 을 계산해 임베딩 행을 **덧붙인다**. 재학습 없이 어휘가 열린다.

    반환: {새 이름: 새 클래스 인덱스}
    """
    from homejepa.model import CLASS_NAMES
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    W_old = model.e_cls.weight.data                       # (NCLS+1, d)
    d = W_old.shape[1]
    nm = "openai/clip-vit-base-patch16"
    tok = CLIPTokenizer.from_pretrained(nm)
    txt = CLIPTextModelWithProjection.from_pretrained(nm, use_safetensors=True).eval()
    def clip_emb(names):
        with torch.no_grad():
            tt = tok(["a photo of a " + n.replace("_", " ") for n in names],
                     padding=True, truncation=True, return_tensors="pt")
            e = txt(**tt).text_embeds
            return torch.nn.functional.normalize(e, dim=-1)
    A = clip_emb(list(CLASS_NAMES)).to(torch.float32)     # (NCLS, 512)
    B = W_old[:len(CLASS_NAMES)].to(torch.float32).cpu()  # (NCLS, d)
    # 릿지 최소자승: A W ≈ B  (앵커가 10개뿐이라 정칙화 필수)
    lam = 1e-2
    AtA = A.T @ A + lam * torch.eye(A.shape[1])
    Wmap = torch.linalg.solve(AtA, A.T @ B)               # (512, d)
    new = clip_emb(new_names).to(torch.float32) @ Wmap    # (M, d)
    # 스케일 정합 — 기존 행들의 노름에 맞춘다
    new = new / (new.norm(dim=1, keepdim=True) + 1e-9) * B.norm(dim=1).mean()
    W_new = torch.cat([W_old.cpu(), new], 0)
    emb = torch.nn.Embedding(W_new.shape[0], d)
    emb.weight.data = W_new
    model.e_cls = emb.to(device)
    base = W_old.shape[0]
    return {n: base + i for i, n in enumerate(new_names)}


def load_model(model_name, device):
    from reeval import build_jepa_probe, build_supervised
    ck = torch.load(os.path.join(HJ, "results", model_name + ".pt"),
                    map_location=device, weights_only=False)
    builder = build_jepa_probe if model_name.startswith("jepa_") else build_supervised
    model, _ = builder(ck, device)
    return model.to(device).eval()


def rank_for(ep, E_t, model, dev, oid, qi_sel, Z, fidx, kw_text, gamma=0.3,
             gate=1.0, device="mps"):
    """단일 질의 → 후보 확률 순위 (신호 A × 신호 B)."""
    from homejepa.model import make_batch
    rec = {r["id"]: r for r in ep["home"]["recepts"]}
    room = {r["id"]: r for r in ep["home"]["rooms"]}
    with torch.no_grad():
        b = make_batch([E_t], [(0, qi_sel)], dev)
        p = model.log_prob(b).exp().cpu().numpy()[0]
    loc_ids = [r["id"] for r in E_t.recepts]
    q = E_t.queries[qi_sel]["meta"]
    t_frame = q["qt"] * TICK_S * FPS

    # keyword 텍스트 임베딩 행: Z 마지막 행에 미리 넣어 두었다(호출측 규약)
    kw_i = [Z.shape[0] - 1]
    rows = []
    for k, lid in enumerate(loc_ids):
        r = rec[lid]
        name = camel_words(r.get("name") or r["type"])
        rm = room[r["room"]]["type"].replace("_", " ")
        ctx_i = [Z.shape[0] - 2 - k]           # 후보별 문맥 행(호출측 규약)
        sig, pk = absence_gate(Z, fidx, kw_i, ctx_i, t_frame, gate=gate)
        w = 1.0
        tag = "미방문"
        if sig < 0:
            w = gamma
            tag = "부재확인(z=%.2f)" % pk
        elif sig > 0:
            w = 1.0 / gamma
            tag = "목격(z=%.2f)" % pk
        rows.append(dict(recept=r.get("name") or r["type"], room=rm,
                         p_model=float(p[k]), w_abs=w, tag=tag))
    tot = sum(r["p_model"] * r["w_abs"] for r in rows) + 1e-12
    for r in rows:
        r["p"] = r["p_model"] * r["w_abs"] / tot
    rows.sort(key=lambda r: -r["p"])
    return rows, q


def prepare(seq_dir, args):
    from homejepa.model import CLASS_NAMES, EpTensors
    gt = json.load(open(os.path.join(seq_dir, "gt", "objects.json")))["instances"]
    meta = json.load(open(os.path.join(seq_dir, "graph_%s.json" % args.ref)))["regions"]
    ref = load_regions(np.load(os.path.join(seq_dir, "regions_%s.npz" % args.ref)),
                       meta["zone_names"], meta["up"])
    g = json.load(open(os.path.join(seq_dir, args.graph + ".json")))
    poses = np.loadtxt(os.path.join(seq_dir, "pose", "poses.txt")).reshape(-1, 4, 4)
    ep = build_episode(g, gt, lambda p: assign(ref, p)[1], poses=poses, extra_cls={},
                       open_vocab=getattr(args, "open_vocab", False))
    E_t = EpTensors(ep, 256, noid=True)
    unk_cat = {}
    NC = len(CLASS_NAMES)
    if getattr(args, "open_vocab", False):
        # 어휘 밖 물체는 이름표만 빌렸으므로 cls 축을 미지 슬롯으로 되돌린다.
        # (--clip-class 가 켜지면 상위에서 CLIP 사상 인덱스로 다시 덮어쓴다)
        unk_cat = {o["id"]: (o.get("src_category") or "object")
                   for o in ep["home"]["objects"] if o.get("cls_unknown")}
        for i in range(len(E_t.cls)):
            if int(E_t.ev_obj[i]) in unk_cat:
                E_t.cls[i] = NC
        for q in E_t.queries:
            if q["meta"]["obj"] in unk_cat:
                q["qcls"] = NC
    if getattr(args, "unknown_class", False):
        # 새 물체 모사: 클래스 축을 전부 미지로
        E_t.cls[:] = NC
        for q in E_t.queries:
            q["qcls"] = NC
    return ep, E_t, g, gt, unk_cat


def build_Z(seq_dir, ep, E_t, kw_text, device):
    """후보 문맥 행들 + keyword 행을 한 번에 임베딩. 규약: [ctx_K..ctx_0, kw]"""
    rec = {r["id"]: r for r in ep["home"]["recepts"]}
    room = {r["id"]: r for r in ep["home"]["rooms"]}
    loc_ids = [r["id"] for r in E_t.recepts]
    texts = []
    for lid in reversed(loc_ids):              # ctx 행: 뒤에서 k 번째 = 후보 k
        r = rec[lid]
        texts.append("a photo of the %s in the %s"
                     % (camel_words(r.get("name") or r["type"]),
                        room[r["room"]]["type"].replace("_", " ")))
    texts.append("a photo of a " + kw_text)
    return clip_scores(seq_dir, texts, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--graph", default="graph_gtdepth")
    ap.add_argument("--ref", default="gtdepth")
    ap.add_argument("--model", default="supervised_two_head_v5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--obj", default=None)
    ap.add_argument("--eval", action="store_true", help="이동물체 전체 채점")
    ap.add_argument("--gamma", type=float, default=0.3)
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--clip-class", action="store_true",
                    help="미지 대신 **CLIP 임베딩을 클래스 속성으로** 주입(열린 어휘 정공법)."
                         " 알려진 10클래스를 앵커로 CLIP→클래스공간 선형사상을 맞춘 뒤"
                         " 새 물체 이름의 임베딩 행을 덧붙인다. 재학습 불필요")
    ap.add_argument("--open-vocab", action="store_true",
                    help="어휘 밖 물체를 버리지 않고 **미지 클래스로 통과**시킨다."
                         " 실측 근거: 클래스를 미지로 돌려도 belief 성능 동일(0.67/1.00)")
    ap.add_argument("--unknown-class", action="store_true",
                    help="질의 물체의 클래스를 **미지**로 돌린다 — '학습 어휘에 없는 새 물건이"
                         " 왔을 때' 를 모사한다. home-jepa 는 NCLS 슬롯(미지)을 갖고 있어"
                         " 동작은 하되 클래스 사전을 잃는다. 그 손실 크기를 재는 스위치")
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    ep, E_t, g, gt, unk_cat = prepare(seq_dir, args)
    dev = torch.device(args.device)
    model = load_model(args.model, dev)
    if args.clip_class and unk_cat:
        names = sorted(set(unk_cat.values()))
        idx_of = extend_class_vocab(model, names, dev)
        for i in range(len(E_t.cls)):
            oid = int(E_t.ev_obj[i])
            if oid in unk_cat:
                E_t.cls[i] = idx_of[unk_cat[oid]]
        for q in E_t.queries:
            if q["meta"]["obj"] in unk_cat:
                q["qcls"] = idx_of[unk_cat[q["meta"]["obj"]]]
        print("CLIP 클래스 주입: 새 어휘 %d개 (%s...)" % (len(names), ", ".join(names[:4])))
    rec = {r["id"]: r for r in ep["home"]["recepts"]}

    name2oid = {}
    for o in ep["home"]["objects"]:
        src = g["objects"].get(o["src_instance"], {})
        name2oid.setdefault(src.get("name") or "", []).append(o["id"])

    def queries_of(oids):
        return [qi for qi in range(len(E_t.queries))
                if E_t.queries[qi]["meta"]["obj"] in oids]

    if args.eval:
        # 이동물체: 마지막 질의에서 top-1/top-3 (신호A 단독 vs A×B)
        movers = [n for n in name2oid
                  if any(gt.get(str(g["objects"].get(str(i), {}).get("instance_id") or ""), {})
                         for i in [])] # placeholder — 아래서 GT moves 로 거른다
        res = []
        for n, oids in name2oid.items():
            qs = queries_of(set(oids))
            if not qs:
                continue
            # GT 이동 여부
            moved = False
            for o in ep["home"]["objects"]:
                if o["id"] in oids:
                    src = g["objects"].get(o["src_instance"], {})
                    r = gt.get(str(src.get("instance_id") or ""), {})
                    if r.get("moves"):
                        moved = True
            if not moved:
                continue
            Zk, fidx = build_Z(seq_dir, ep, E_t, camel_words(n), args.device)
            rows, q = rank_for(ep, E_t, model, dev, oids, qs[-1], Zk, fidx,
                               camel_words(n), args.gamma, args.gate, args.device)
            rows0 = sorted(rows, key=lambda r: -r["p_model"])
            gt_name = rec[q["gt_recept"]].get("name") or rec[q["gt_recept"]]["type"]
            def rank_of(rr):
                for i, r in enumerate(rr):
                    if r["recept"] == gt_name:
                        return i + 1
                return 99
            res.append(dict(obj=n, gt=gt_name, rA=rank_of(rows0), rAB=rank_of(rows),
                            top=rows[0]["recept"], tag=rows[0]["tag"]))
            print("%-24s GT=%-22s A단독 rank=%d → A×부재 rank=%d · top=%s [%s]"
                  % (n[:24], gt_name[:22], res[-1]["rA"], res[-1]["rAB"],
                     res[-1]["top"][:20], res[-1]["tag"]))
        if res:
            for k in (1, 3):
                a = np.mean([r["rA"] <= k for r in res])
                b = np.mean([r["rAB"] <= k for r in res])
                print("top-%d: 모델 단독 %.2f → +부재게이트 %.2f (n=%d)" % (k, a, b, len(res)))
        return

    # 단건 질의
    hits = {n: v for n, v in name2oid.items() if args.obj.lower() in n.lower()}
    if not hits:
        sys.exit("물체 없음: %s" % sorted(name2oid)[:10])
    oids = {o for v in hits.values() for o in v}
    qs = queries_of(oids)
    Zk, fidx = build_Z(seq_dir, ep, E_t, camel_words(list(hits)[0]), args.device)
    rows, q = rank_for(ep, E_t, model, dev, oids, qs[-1], Zk, fidx,
                       camel_words(list(hits)[0]), args.gamma, args.gate, args.device)
    print("\n질의: '%s 지금 어디?' (틱 %d = %d초)" % (list(hits)[0], q["qt"], q["qt"] * TICK_S))
    for r in rows[:5]:
        print("  %5.1f%%  %-24s (%s)  [모델 %.2f · %s]"
              % (100 * r["p"], r["recept"][:24], r["room"], r["p_model"], r["tag"]))


if __name__ == "__main__":
    main()
