"""Download the low-light HDR+ bursts selected from index.csv (exposure x ISO >= --min-exp-iso): all payload DNGs,
lens-shading maps and sidecars, plus Google's results (final.jpg = label reference, merged.dng, reference_frame.txt).

    python -m scripts.journal.real.hdrplus_download --root /mnt/Data/akshay/real/hdrplus --min-exp-iso 4
Resumable: files already present with the right size are skipped.
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

API = "https://storage.googleapis.com/storage/v1/b/hdrplusdata/o"
GET = "https://storage.googleapis.com/hdrplusdata/"


def files(burst):
    out = []
    for pre in (f"20171106/bursts/{burst}/", f"20171106/results_20171023/{burst}/"):
        out += json.load(urllib.request.urlopen(f"{API}?prefix={pre}&fields=items(name,size)")).get("items", [])
    return out


def fetch(item, root):
    p = root / item["name"].replace("20171106/", "", 1)
    if p.exists() and p.stat().st_size == int(item["size"]):
        return 0
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".part")
    with urllib.request.urlopen(GET + item["name"], timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(p)
    return int(item["size"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-exp-iso", type=float, default=4.0)
    a = ap.parse_args()
    root = Path(a.root)
    idx = pd.read_csv(root / "index.csv")
    sel = idx[idx.exposure_x_iso >= a.min_exp_iso].burst.tolist()
    (root / "selected.txt").write_text("\n".join(sel) + "\n")
    print(len(sel), "bursts", flush=True)
    with ThreadPoolExecutor(16) as ex:
        items = [i for lst in ex.map(files, sel) for i in lst]
    print(len(items), "files", sum(int(i["size"]) for i in items) / 2**30, "GiB", flush=True)
    done = 0
    with ThreadPoolExecutor(16) as ex:
        for k, n in enumerate(ex.map(lambda i: fetch(i, root), items)):
            done += n
            if k % 500 == 0:
                print(f"{k}/{len(items)} files, {done / 2**30:.1f} GiB new", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
