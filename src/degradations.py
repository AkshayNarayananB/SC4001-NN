"""Deterministic image degradations applied at native resolution, before model preprocessing."""
import zlib

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Parameter per severity level (index 0 = level 1). Values are in native-resolution pixel
# units; COCO images are ~640x480 and LLaVA downsizes to 336, which roughly halves the
# effective blur sigma and averages out some noise, so levels are set accordingly.
# Calibrated 2026-11-01 (configs/calib): greedy LLaVA-1.5 baseline accuracy drops of roughly
# -3/-7/-14/-23 points for blur and -2/-12/-14/-20 for noise (clean = 80.0 on the 100-question
# subset). The first ladder (blur 2/4/8/16, noise .05/.1/.2/.4) left noise s1-s3 with no effect.
LEVELS = {
    "blur": [5.0, 7.0, 10.0, 16.0],   # Gaussian sigma in pixels
    "noise": [0.3, 0.5, 0.7, 1.0],    # Gaussian std on a [0, 1] intensity scale
}

KINDS = tuple(LEVELS)


def seed_for(image_name: str, kind: str, level: int) -> int:
    return zlib.crc32(f"{image_name}|{kind}|{level}".encode())


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    out = gaussian_filter(arr, sigma=(sigma, sigma, 0), mode="reflect")
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def gaussian_noise(img: Image.Image, std: float, seed: int) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    out = arr + rng.normal(0.0, std, size=arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8))


def apply(img: Image.Image, kind: str, param: float, seed: int) -> Image.Image:
    """Apply `kind` with an explicit parameter (sigma for blur, std for noise)."""
    img = img.convert("RGB")
    if kind == "blur":
        return gaussian_blur(img, param)
    if kind == "noise":
        return gaussian_noise(img, param, seed)
    raise ValueError(f"unknown degradation kind: {kind}")


def degrade(img: Image.Image, kind: str, level: int, image_name: str) -> Image.Image:
    """Apply `kind` at severity `level` (1-based). `image_name` seeds stochastic degradations."""
    return apply(img, kind, LEVELS[kind][level - 1], seed_for(image_name, kind, level))
