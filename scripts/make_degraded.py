"""Build every degraded version of the POPE images and write manifest.csv.

Deterministic, so it is safe to re-run; existing files are overwritten with identical output.
    python -m scripts.make_degraded [--limit N] [--workers N]
"""
import argparse
import csv
from multiprocessing import Pool

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from scripts.download_pope import load_image_names
from src.degradations import KINDS, LEVELS, degrade, seed_for
from src.paths import POPE_MANIFEST, POPE_SOURCE_IMAGES, degraded_image_path

FIELDS = ["image", "kind", "level", "param", "seed", "psnr", "ssim", "path"]


def save(img: Image.Image, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")


def process(name: str) -> list[dict]:
    clean = Image.open(POPE_SOURCE_IMAGES / name).convert("RGB")
    clean_arr = np.asarray(clean)
    path = degraded_image_path(name, "clean")
    save(clean, path)
    rows = [dict(image=name, kind="clean", level=0, param="", seed="", psnr="", ssim="", path=str(path))]

    for kind in KINDS:
        for level, param in enumerate(LEVELS[kind], start=1):
            out = degrade(clean, kind, level, name)
            out_arr = np.asarray(out)
            path = degraded_image_path(name, kind, level)
            save(out, path)
            rows.append(dict(
                image=name, kind=kind, level=level, param=param,
                seed=seed_for(name, kind, level) if kind == "noise" else "",
                psnr=round(peak_signal_noise_ratio(clean_arr, out_arr, data_range=255), 3),
                ssim=round(structural_similarity(clean_arr, out_arr, channel_axis=2, data_range=255), 4),
                path=str(path),
            ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process the first N images")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    names = load_image_names()[: args.limit]
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(process, names), total=len(names)))

    rows = [row for image_rows in results for row in image_rows]
    with open(POPE_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Per-level summary so the severity ladder can be sanity-checked at a glance
    print(f"\n{len(rows)} images written, manifest: {POPE_MANIFEST}")
    print(f"{'kind':<6} {'lvl':>3} {'param':>6} {'PSNR':>7} {'SSIM':>6}")
    for kind in KINDS:
        for level, param in enumerate(LEVELS[kind], start=1):
            sel = [r for r in rows if r["kind"] == kind and r["level"] == level]
            psnr = np.mean([r["psnr"] for r in sel])
            ssim = np.mean([r["ssim"] for r in sel])
            print(f"{kind:<6} {level:>3} {param:>6} {psnr:>7.2f} {ssim:>6.3f}")


if __name__ == "__main__":
    main()
