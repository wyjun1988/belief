#!/usr/bin/env python3
"""IT3DEgo tar 에서 **pv 블록만** 찾아낸다 — 893GB 중 20GB 만 받으면 된다.

    python3 scripts/it3d_map.py --out data/it3dego/pv_blocks.json

### 왜

한 영상의 pv 는 410프레임 · 약 400MB(0.7fps · 9.8분)다. 49영상이면 **20GB**.
나머지 870GB 는 `depth_ahat`·`vlc_*`(45~30fps · 카메라 5대)라 우리는 안 쓴다.
tar 안에서 각 영상의 pv 가 **연속 블록**이라는 것을 400MB 조각 실측으로 확인했다
(한 조각에 `video_27_scene_5/pv` 410개가 통째로 들어 있었다).

그래서 전수 다운로드 15시간 대신, 블록 위치를 찾아 범위 요청으로 20분에 끝낸다.

### 방법

tar 는 색인이 없지만 헤더가 512 정렬 + `ustar` 매직이라 **임의 위치에서 재동기화**된다.
  ① 성긴 통과 — `--coarse` 간격으로 훑어 영상 블록 경계를 잡는다
  ② 영상 안에서 **이진 탐색** — 하위 디렉터리는 tar 안에서 사전순 연속이므로
     `pv` 의 시작·끝을 log 번의 프로브로 찾는다

⚠️ 구글이 간헐적으로 "Quota exceeded" HTML 을 **HTTP 200** 으로 준다 →
프로브가 헤더를 못 찾으면 재시도한다(`fetch.sh` 와 같은 함정).
"""
import argparse, json, re, subprocess, sys, time

ID = "1VVszWG4mmm0g3ai3EoZw-3cGNBmZCN-9"
TOT = 958982840320
_uu = [None, 0.0]


def uuid():
    if _uu[0] and time.time() - _uu[1] < 600:
        return _uu[0]
    for _ in range(10):
        pg = subprocess.run(
            ["curl", "-sL", "-m", "60",
             f"https://drive.usercontent.google.com/download?id={ID}&export=download"],
            capture_output=True).stdout.decode("utf-8", "replace")
        if "Virus scan warning" in pg:
            m = re.search(r'name="uuid" value="([^"]*)"', pg)
            if m:
                _uu[0], _uu[1] = m.group(1), time.time()
                return _uu[0]
        time.sleep(20)
    raise RuntimeError("uuid 획득 실패")


def get(a, n):
    for _ in range(4):
        r = subprocess.run(
            ["curl", "-s", "-m", "120", "-H", f"Range: bytes={a}-{a+n-1}",
             f"https://drive.usercontent.google.com/download?id={ID}"
             f"&export=download&confirm=t&uuid={uuid()}"], capture_output=True).stdout
        if r[:9] != b"<!DOCTYPE" and len(r) > n // 2:
            return r
        _uu[0] = None
        time.sleep(15)
    return b""


def hdr(b, i):
    nm = b[i:i + 100].rstrip(b"\0").decode("utf-8", "replace")
    pre = b[i + 345:i + 500].rstrip(b"\0").decode("utf-8", "replace")
    try:
        sz = int(b[i + 124:i + 136].rstrip(b"\0 ").decode() or "0", 8)
    except ValueError:
        sz = 0
    return (pre + "/" + nm if pre else nm), sz


def probe(off, win=1 << 21):
    """off 이후 첫 tar 헤더 → (절대오프셋, 이름, 크기)."""
    off = max(0, (off // 512) * 512)
    b = get(off, win)
    for i in range(0, max(0, len(b) - 512), 512):
        if b[i + 257:i + 263] == b"ustar\0" or b[i + 257:i + 265] == b"ustar  \0":
            nm, sz = hdr(b, i)
            if nm:
                return off + i, nm, sz
    return None


KEY = re.compile(r"raw_videos/(video_\d+_scene_\d+)/([^/]+)/")


def parts(name):
    m = KEY.match(name)
    return (m.group(1), m.group(2)) if m else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=float, default=2.0, help="성긴 통과 간격 GB")
    ap.add_argument("--start", type=float, default=0.0, help="탐색 시작 GB")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    step = int(args.coarse * 1e9)
    marks = []
    off = int(args.start * 1e9)
    while off < TOT:
        r = probe(off)
        if r:
            v, sub = parts(r[1])
            marks.append((r[0], v, sub, r[1]))
            print("%14d  %-20s %-12s %s" % (r[0], v or "-", sub or "-", r[1][-46:]), flush=True)
        off += step
    json.dump(marks, open(args.out + ".marks", "w"))

    # 영상별 성긴 범위 → 그 안에서 pv 시작·끝 이진 탐색
    seen = {}
    for i, (o, v, sub, _) in enumerate(marks):
        if v:
            seen.setdefault(v, [o, o])
            seen[v][0] = min(seen[v][0], o)
            seen[v][1] = max(seen[v][1], o)
    print("\n영상 %d개 발견" % len(seen))
    blocks = {}
    for v, (lo, hi) in sorted(seen.items()):
        lo = max(0, lo - step); hi = min(TOT, hi + step)

        def sub_at(x):
            r = probe(x)
            if not r:
                return None
            vv, ss = parts(r[1])
            return (vv, ss)

        def find(pred, a, b):
            """pred(x) 가 False→True 로 바뀌는 최소 x."""
            for _ in range(34):
                if b - a <= 1 << 20:
                    break
                m = (a + b) // 2
                if pred(m):
                    b = m
                else:
                    a = m
            return b

        st = find(lambda x: (sub_at(x) or (None, None)) >= (v, "pv"), lo, hi)
        en = find(lambda x: (sub_at(x) or (None, None)) > (v, "pv"), st, hi)
        blocks[v] = [st, en]
        print("  %-20s pv %14d ~ %14d  (%.0f MB)" % (v, st, en, (en - st) / 1e6), flush=True)
        json.dump(blocks, open(args.out, "w"), indent=1)
    print("\n합계 %.1f GB" % (sum(b - a for a, b in blocks.values()) / 1e9))


if __name__ == "__main__":
    main()
