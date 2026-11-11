"""Restore the cached degraded POPE images with a trained DAE (Phase 2, fix B).

    python -m scripts.make_restored --dae dae_v1 [--kinds noise] [--limit N]

Reads $VCD_CHECKPOINT_ROOT/dae/<dae>/best.pt. Writes pope/images/restored/<dae>/<kind>/s<level>/*.png
and pope/restored_<dae>_manifest.csv (PSNR/SSIM against the clean image, before and after).
Default kinds: noise only, since dae_v1 leaves blurred images unchanged (CLAUDE.md, Fix B).
Runs on GPU if there is one; fp32 with deterministic cuDNN, so reruns give the same PNGs.
"""
import argparse
import csv

import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from scripts.download_pope import load_image_names
from src.degradations import KINDS, LEVELS
from src.fixes.dae import load_dae, restore
from src.model import pick_device
from src.paths import CHECKPOINT_ROOT, POPE_ROOT, degraded_image_path, restored_image_path

FIELDS = ["image", "kind", "level", "psnr_in", "psnr_out", "ssim_in", "ssim_out", "path"]


def metrics(clean: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    return (round(peak_signal_noise_ratio(clean, img, data_range=255), 3),
            round(structural_similarity(clean, img, channel_axis=2, data_range=255), 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dae", default="dae_v1", help="DAE run name under $VCD_CHECKPOINT_ROOT/dae/")
    parser.add_argument("--kinds", nargs="+", default=["noise"], choices=KINDS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ckpt_path = CHECKPOINT_ROOT / "dae" / args.dae / "best.pt"
    step = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("step")
    model = load_dae(ckpt_path, pick_device())
    print(f"{ckpt_path} (step {step}) on {next(model.parameters()).device}")

    rows = []
    for name in tqdm(load_image_names()[: args.limit], mininterval=30):
        clean = np.asarray(Image.open(degraded_image_path(name, "clean")).convert("RGB"))
        for kind in args.kinds:
            for level in range(1, len(LEVELS[kind]) + 1):
                degraded = Image.open(degraded_image_path(name, kind, level)).convert("RGB")
                restored = restore(model, degraded)
                path = restored_image_path(name, args.dae, kind, level)
                path.parent.mkdir(parents=True, exist_ok=True)
                restored.save(path, format="PNG")
                (p_in, s_in), (p_out, s_out) = metrics(clean, np.asarray(degraded)), metrics(clean, np.asarray(restored))
                rows.append(dict(image=name, kind=kind, level=level, psnr_in=p_in, psnr_out=p_out,
                                 ssim_in=s_in, ssim_out=s_out, path=str(path)))

    manifest = POPE_ROOT / f"restored_{args.dae}_manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} images restored, manifest: {manifest}")
    print(f"{'kind':<6} {'lvl':>3} {'PSNR in':>8} {'PSNR out':>9} {'SSIM in':>8} {'SSIM out':>9}")
    for kind in args.kinds:
        for level in range(1, len(LEVELS[kind]) + 1):
            sel = [r for r in rows if r["kind"] == kind and r["level"] == level]
            mean = lambda k: np.mean([r[k] for r in sel])
            print(f"{kind:<6} {level:>3} {mean('psnr_in'):>8.2f} {mean('psnr_out'):>9.2f} {mean('ssim_in'):>8.3f} {mean('ssim_out'):>9.3f}")


if __name__ == "__main__":
    main()
