#!/usr/bin/env python3
"""내려받는 중인 IT3DEgo tar 를 **로컬에서** 걸으며 색인을 만든다.

    python3 scripts/it3d_tarindex.py --tar <경로> --out <index.jsonl>

tar 는 순차 자기기술 포맷이라 **앞부분만 있어도** 거기까지 목록을 얻을 수 있다.
다운로드가 도는 동안 계속 돌려 진척을 확인하고, 동시에 **무결성 검사**가 된다 —
헤더가 512 경계에서 `ustar` 매직으로 안 나오면 그 지점에서 파일이 깨진 것이다
(구글이 HTML 을 섞어 넣는 사고를 잡는다).

색인이 있으면 나중에 프레임을 **PNG 로 풀지 않고** tar 에서 직접 바이트 범위로
읽어 검출할 수 있다 — 600GB 추출을 통째로 아낀다.
"""
import argparse, json, os


def walk(f, size, start=0):
    off = start
    while off + 512 <= size:
        f.seek(off)
        h = f.read(512)
        if len(h) < 512:
            break
        if h[:1] == b"\0":
            off += 512
            continue
        if h[257:263] != b"ustar\0" and h[257:265] != b"ustar  \0":
            raise ValueError("오프셋 %d 에서 tar 헤더가 아니다 — 파일 손상" % off)
        name = h[0:100].rstrip(b"\0").decode("utf-8", "replace")
        pre = h[345:500].rstrip(b"\0").decode("utf-8", "replace")
        try:
            sz = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        except ValueError:
            sz = 0
        typ = chr(h[156]) if h[156] else "0"
        full = pre + "/" + name if pre else name
        data = off + 512
        nxt = data + ((sz + 511) // 512) * 512
        if nxt > size:                      # 아직 다 안 받은 마지막 항목
            break
        yield dict(off=data, size=sz, type=typ, name=full)
        off = nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    size = os.path.getsize(args.tar)
    n, last, kinds = 0, 0, {}
    with open(args.tar, "rb") as f, open(args.out, "w") as o:
        for r in walk(f, size):
            o.write(json.dumps(r) + "\n")
            n += 1
            last = r["off"] + r["size"]
            if r["type"] == "0":
                k = r["name"].split("/")[2] if r["name"].count("/") >= 2 else r["name"]
                kinds[k] = kinds.get(k, 0) + r["size"]
    print("항목 %d · 검증된 바이트 %d / %d (%.2f%%)"
          % (n, last, size, last * 100.0 / max(size, 1)))
    for k, v in sorted(kinds.items(), key=lambda t: -t[1])[:10]:
        print("  %-14s %8.2f GB" % (k, v / 1e9))


if __name__ == "__main__":
    main()
