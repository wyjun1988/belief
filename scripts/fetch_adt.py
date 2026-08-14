#!/usr/bin/env python3
"""ADT 선별 다운로드 — 파일럿 시퀀스의 픽셀 데이터까지 받는다.

home-jepa 의 `scripts/adt_download.py` 는 GT CSV(main_groundtruth)만 받도록 되어 있고
비-GT 파트를 `seq_dir/<part>/` 하위로 풀어버린다. 이 프로젝트는 `video.vrs`,
`segmentations.vrs`, `depth_images.vrs` 를 **한 폴더에 평평하게** 둬야
`AriaDigitalTwinDataPathsProvider` 가 시퀀스를 인식하므로 별도 판을 쓴다.

사용:
    python3 scripts/fetch_adt.py --preset pilot                    # 코어(≈6.2GB)
    python3 scripts/fetch_adt.py --preset pilot --parts depth      # 뎁스 GT(≈11.4GB)
    python3 scripts/fetch_adt.py --list                            # 받을 목록/용량만 출력

서명 URL 은 발급 후 보통 14일이면 만료된다. 만료 시 projectaria.com/datasets/adt 에서
매니페스트를 재발급받아 data/adt/ADT_download_urls.json 을 교체할 것.
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "adt", "ADT_download_urls.json")
DEST = os.path.join(ROOT, "data", "adt", "gt")

# 파일럿 2시퀀스 — GT 스캔(변위>0.3m 물체 수, 방 전환 유무)으로 선정. docs/DESIGN 참조.
PRESETS = {
    "pilot": [
        "Apartment_release_multiskeleton_party_seq102_M1292",   # 2인, 7개 이동
        "Apartment_release_decoration_seq137_M1292",            # 1인, 4개 이동, 실제 방 전환
    ],
}

CORE_PARTS = [
    "main_vrs",                # → video.vrs (RGB/SLAM/IMU 원본)
    "segmentation",            # → segmentations*.vrs (GT 인스턴스 마스크)
    "mps_slam_points",         # → semidense_points.csv.gz (뎁스 정합 앵커)
    "mps_slam_trajectories",
    "mps_slam_calibration",
]

# main_vrs 는 zip 이 아니라 raw .vrs 이고, ADT SDK 는 'video.vrs' 라는 이름을 찾는다.
RENAME = {"main_vrs": "video.vrs"}
# MPS 산출물은 시퀀스 루트를 어지럽히지 않게 mps/slam/ 아래로 모은다.
SUBDIR = {
    "mps_slam_points": os.path.join("mps", "slam"),
    "mps_slam_trajectories": os.path.join("mps", "slam"),
    "mps_slam_calibration": os.path.join("mps", "slam"),
}


def human(n):
    return "%.2f GB" % (n / 1e9) if n >= 1e9 else "%.0f MB" % (n / 1e6)


def sha1_of(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst, size, label):
    """Range 헤더로 이어받는다 — 수 GB 파일이 중간에 끊겨도 처음부터 다시 받지 않는다."""
    have = os.path.getsize(dst) if os.path.exists(dst) else 0
    if have == size:
        return
    if have > size:                       # 손상 — 버리고 다시
        os.remove(dst)
        have = 0
    req = urllib.request.Request(url)
    mode = "wb"
    if have:
        req.add_header("Range", "bytes=%d-" % have)
        mode = "ab"
        print("      resume @ %s" % human(have), flush=True)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, mode) as f:
        if have and r.status != 206:      # 서버가 Range 를 무시했다
            f.close()
            os.remove(dst)
            raise RuntimeError("server ignored Range; rerun to restart %s" % label)
        got = have
        last = 0.0
        while True:
            buf = r.read(1 << 22)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            pct = 100.0 * got / size
            if pct - last >= 5.0:
                last = pct
                rate = (got - have) / max(time.time() - t0, 1e-6) / 1e6
                print("      %5.1f%%  %s  %.1f MB/s" % (pct, human(got), rate), flush=True)


def extract_flat(zip_path, out_dir):
    """zip 안에 단일 최상위 폴더가 있으면 벗겨내고 평평하게 푼다."""
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        tops = {n.split("/")[0] for n in names}
        strip = len(tops) == 1 and any("/" in n for n in names)
        for n in names:
            rel = n.split("/", 1)[1] if strip and "/" in n else n
            if not rel or rel.startswith("__MACOSX"):
                continue
            target = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(n) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return sorted(os.path.basename(n) for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--out", default=DEST)
    ap.add_argument("--preset", default="pilot", choices=sorted(PRESETS))
    ap.add_argument("--seqs", default=None, help="쉼표 구분 시퀀스명 (preset 대신)")
    ap.add_argument("--parts", default=",".join(CORE_PARTS))
    ap.add_argument("--list", action="store_true", help="목록·용량만 출력하고 종료")
    ap.add_argument("--keep-zips", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="sha1 검증 생략(빠름)")
    args = ap.parse_args()

    import json
    if not os.path.exists(args.manifest):
        sys.exit("매니페스트 없음: %s\nprojectaria.com/datasets/adt 에서 재발급 필요." % args.manifest)
    seqs_all = json.load(open(args.manifest))["sequences"]

    seqs = args.seqs.split(",") if args.seqs else PRESETS[args.preset]
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]

    jobs = []
    for s in seqs:
        if s not in seqs_all:
            sys.exit("매니페스트에 없는 시퀀스: %s" % s)
        for p in parts:
            e = seqs_all[s].get(p)
            if e is None:
                print("  (없음) %s / %s" % (s, p))
                continue
            jobs.append((s, p, e))

    total = sum(e["file_size_bytes"] for _, _, e in jobs)
    print("%d files, %s" % (len(jobs), human(total)))
    for s, p, e in jobs:
        print("   %-52s %-24s %s" % (s, p, human(e["file_size_bytes"])))
    if args.list:
        return

    t0 = time.time()
    for i, (s, p, e) in enumerate(jobs, 1):
        seq_dir = os.path.join(args.out, s)
        os.makedirs(seq_dir, exist_ok=True)
        marker = os.path.join(seq_dir, ".done_" + p)
        if os.path.exists(marker):
            print("[%d/%d] %s / %s  — 이미 완료" % (i, len(jobs), s, p), flush=True)
            continue
        print("[%d/%d] %s / %s  %s" % (i, len(jobs), s, p, human(e["file_size_bytes"])), flush=True)

        staging = os.path.join(seq_dir, "." + e["filename"])
        download(e["download_url"], staging, e["file_size_bytes"], "%s/%s" % (s, p))

        if not args.no_verify:
            got = sha1_of(staging)
            if got != e["sha1sum"]:
                os.remove(staging)
                sys.exit("sha1 불일치 %s/%s: %s != %s — 다시 실행하세요" % (s, p, got, e["sha1sum"]))

        if e["filename"].endswith(".zip"):
            out_dir = os.path.join(seq_dir, SUBDIR[p]) if p in SUBDIR else seq_dir
            got_files = extract_flat(staging, out_dir)
            print("      → %s" % ", ".join(got_files[:6]), flush=True)
            if not args.keep_zips:
                os.remove(staging)
        else:
            final = os.path.join(seq_dir, RENAME.get(p, e["filename"]))
            os.replace(staging, final)
            print("      → %s" % os.path.basename(final), flush=True)

        open(marker, "w").close()

    print("ADT_FETCH_DONE  %s  (%.0f분)" % (args.out, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
