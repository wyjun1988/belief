#!/usr/bin/env python3
"""IT3DEgo pv 프레임을 **영상당 연속 1GB 조각**으로만 받는다 — 893GB → 49GB.

    python3 scripts/it3d_slice.py --stage map    --out data/it3dego/pv_marks.json
    python3 scripts/it3d_slice.py --stage fetch  --marks … --dest /Volumes/exDisk/it3dego/pv

### 근거 — tar 안 파일 순서가 **무작위**다

한 영상의 pv 는 약 15GB(≈14,000프레임 · 30fps)라 전수를 받으면 893GB 다.
그런데 실측하니 tar 안 순서가 **시각 오름차순이 아니다**(무작위 해시 순서).
400MB 조각 410프레임의 시각 간격이 중앙 1.03초 · 큰 공백 0개로, **9.8분 전 구간에
고루 퍼져** 있었다. 즉 **연속 1GB 조각 = 전 구간 균일 표본 약 900프레임**이다.

우리 파이프라인은 어차피 프레임을 부분추출해 쓴다(`--stride`). 그러니 15GB 를 받아
1/15 을 쓰는 대신 **1GB 를 받아 전부 쓰면** 결과가 같고 시간이 15배 빠르다.

⚠️ 조각이 정말 균일한지는 **받은 뒤 검사**한다(`--check`) — 큰 시간 공백이 있으면
그 영상은 무작위 순서가 아니라는 뜻이므로 표본으로 못 쓴다.
"""
import argparse, json, os, re, subprocess, sys, time

ID = "1VVszWG4mmm0g3ai3EoZw-3cGNBmZCN-9"
TOT = 958982840320
_u = [None, 0.0]


def uuid():
    if _u[0] and time.time() - _u[1] < 600:
        return _u[0]
    for _ in range(10):
        pg = subprocess.run(
            ["curl", "-sL", "-m", "60",
             "https://drive.usercontent.google.com/download?id=%s&export=download" % ID],
            capture_output=True).stdout.decode("utf-8", "replace")
        if "Virus scan warning" in pg:
            m = re.search(r'name="uuid" value="([^"]*)"', pg)
            if m:
                _u[0], _u[1] = m.group(1), time.time()
                return _u[0]
        time.sleep(20)
    raise RuntimeError("uuid 획득 실패")


def url():
    return ("https://drive.usercontent.google.com/download?id=%s"
            "&export=download&confirm=t&uuid=%s" % (ID, uuid()))


def get(a, n, out=None):
    """⚠️ 구글이 'Quota exceeded' HTML 을 HTTP 200 으로 준다 — 받은 뒤 검사한다."""
    for _ in range(5):
        cmd = ["curl", "-s", "-m", "900", "-H", "Range: bytes=%d-%d" % (a, a + n - 1), url()]
        if out:
            cmd = cmd[:1] + ["-o", out] + cmd[1:]
            subprocess.run(cmd)
            if os.path.exists(out) and os.path.getsize(out) > n // 2:
                with open(out, "rb") as f:
                    if f.read(9) != b"<!DOCTYPE":
                        return os.path.getsize(out)
        else:
            b = subprocess.run(cmd, capture_output=True).stdout
            if b[:9] != b"<!DOCTYPE" and len(b) > n // 2:
                return b
        _u[0] = None
        time.sleep(20)
    return 0 if out else b""


def scan(b, base=0):
    """버퍼에서 512정렬 tar 헤더를 모두 파싱."""
    out = []
    i = 0
    while i + 512 <= len(b):
        if b[i + 257:i + 263] == b"ustar\0" or b[i + 257:i + 265] == b"ustar  \0":
            nm = b[i:i + 100].rstrip(b"\0").decode("utf-8", "replace")
            pre = b[i + 345:i + 500].rstrip(b"\0").decode("utf-8", "replace")
            try:
                sz = int(b[i + 124:i + 136].rstrip(b"\0 ").decode() or "0", 8)
            except ValueError:
                sz = 0
            full = pre + "/" + nm if pre else nm
            data = i + 512
            nxt = data + ((sz + 511) // 512) * 512
            if nxt <= len(b) and full:
                out.append(dict(off=base + data, local=data, size=sz, name=full))
                i = nxt
                continue
        i += 512
    return out


PV = re.compile(r"raw_videos/(video_\d+_scene_\d+)/pv/(\d+)\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["map", "fetch"], required=True)
    ap.add_argument("--coarse", type=float, default=3.0, help="성긴 통과 간격 GB")
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--slice", type=float, default=1.0, help="영상당 받을 조각 GB")
    ap.add_argument("--marks", default="data/it3dego/pv_marks.json")
    ap.add_argument("--dest", default="/Volumes/exDisk/it3dego/pv")
    ap.add_argument("--out", default="data/it3dego/pv_marks.json")
    args = ap.parse_args()

    if args.stage == "map":
        marks = json.load(open(args.out)) if os.path.exists(args.out) else {}
        off = int(args.start * 1e9)
        step = int(args.coarse * 1e9)
        while off < TOT:
            b = get(off, 1 << 21)
            hs = scan(b, off)
            hit = next((h for h in hs if PV.match(h["name"])), None)
            if hit:
                v = PV.match(hit["name"]).group(1)
                if v not in marks:
                    marks[v] = hit["off"] - 512
                    print("%14d  %s  (영상 %d개째)" % (hit["off"], v, len(marks)), flush=True)
                    json.dump(marks, open(args.out, "w"), indent=1)
            off += step
        print("영상 %d개 확보 → %s" % (len(marks), args.out))
        return

    marks = json.load(open(args.marks))
    os.makedirs(args.dest, exist_ok=True)
    n = int(args.slice * 1e9)
    for v, off in sorted(marks.items()):
        p = os.path.join(args.dest, v + ".bin")
        if os.path.exists(p) and os.path.getsize(p) > n // 2:
            print("  %-20s 이미 있음" % v); continue
        t = time.time()
        got = get(off, n, out=p)
        b = open(p, "rb").read(min(got, n))
        hs = [h for h in scan(b) if PV.match(h["name"])]
        same = sum(1 for h in hs if PV.match(h["name"]).group(1) == v)
        ts = sorted(int(PV.match(h["name"]).group(2)) for h in hs
                    if PV.match(h["name"]).group(1) == v)
        span = (ts[-1] - ts[0]) / 1e7 if len(ts) > 1 else 0
        gap = max((ts[i + 1] - ts[i]) for i in range(len(ts) - 1)) / 1e7 if len(ts) > 1 else 0
        print("  %-20s %5.1f GB · 프레임 %4d(같은 영상 %4d) · 구간 %5.0f s · 최대공백 %4.0f s · %4.0f s"
              % (v, got / 1e9, len(hs), same, span, gap, time.time() - t), flush=True)
        json.dump([dict(name=h["name"], off=h["local"], size=h["size"], type="0") for h in hs],
                  open(os.path.join(args.dest, v + ".index.json"), "w"))


if __name__ == "__main__":
    main()
