"""dsg_web.html 템플릿 + 점구름/씬 데이터 → 자립형 HTML 한 장.

    $P kx/viz/build_web.py --seq <name> --out docs/dsg_viewer.html

아티팩트 CSP 가 외부 요청을 전부 막으므로 데이터는 전부 인라인한다.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--root", default=os.path.join(ROOT, "data", "seq"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "dsg_viewer.html"))
    args = ap.parse_args()

    seq_dir = args.seq if os.path.isdir(args.seq) else os.path.join(args.root, args.seq)
    cd = os.path.join(seq_dir, "cloud")
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsg_web.html")).read()
    html = tpl.replace("__SCENE__", open(os.path.join(cd, "scene.json")).read())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write(html)
    print("→ %s  (%.1f MB)" % (args.out, len(html) / 1e6))


if __name__ == "__main__":
    main()
