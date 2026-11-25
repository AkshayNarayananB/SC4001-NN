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

# Sensor-realistic degradations (journal extension, 2026-11-25). Kept separate from LEVELS so the
# course-project dataset build (KINDS) is unchanged; levels are calibrated on DEV images only
# (scripts/journal/calibrate_levels.py) to give baseline accuracy drops comparable to blur/noise.
SENSOR_LEVELS = {
    # Calibrated 2026-11-25 on 600 DEV questions (results/journal/calibration.csv); greedy LLaVA-1.5
    # accuracy drops vs clean 85.8: lowlight ~-4/-9/-17/-25, motion -4/-8/-13/-17, jpeg -1/-5/-8/-11.
    "lowlight": [20.0, 10.0, 5.0, 3.0],     # peak signal in photo-electrons at white (lower = darker scene)
    "motion": [25.0, 40.0, 60.0, 90.0],     # motion-blur path length in native pixels
    "jpeg": [10.0, 5.0, 3.0, 1.0],          # JPEG quality factor
}
READ_NOISE_E = 3.0   # sensor read noise (electrons, rms), typical of small CMOS sensors

KINDS = tuple(LEVELS)
ALL_LEVELS = {**LEVELS, **SENSOR_LEVELS}
ALL_KINDS = tuple(ALL_LEVELS)


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


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def low_light(img: Image.Image, peak_electrons: float, seed: int) -> Image.Image:
    """Low-light capture with auto-gain: the scene (linear light) delivers `peak_electrons` at white;
    each pixel gets Poisson shot noise plus Gaussian read noise, then a digital gain of
    1/peak_electrons restores brightness and the sRGB curve is re-applied (8-bit output).
    Noise is therefore signal-dependent and strongest in the shadows, unlike additive Gaussian noise."""
    lin = srgb_to_linear(np.asarray(img, dtype=np.float64) / 255.0)
    rng = np.random.default_rng(seed)
    electrons = rng.poisson(lin * peak_electrons) + rng.normal(0.0, READ_NOISE_E, size=lin.shape)
    out = linear_to_srgb(np.clip(electrons / peak_electrons, 0.0, 1.0))
    return Image.fromarray(np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8))


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """Normalised straight-line motion-blur kernel, rendered with sub-pixel sampling."""
    size = int(np.ceil(length)) | 1
    k = np.zeros((size, size), dtype=np.float64)
    c = (size - 1) / 2
    t = np.linspace(-length / 2, length / 2, int(length * 8) + 1)
    a = np.deg2rad(angle_deg)
    xs, ys = c + t * np.cos(a), c - t * np.sin(a)
    for x, y in zip(xs, ys):   # bilinear splat
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if 0 <= x0 + dx < size and 0 <= y0 + dy < size:
                k[y0 + dy, x0 + dx] += w
    return k / k.sum()


def motion_blur(img: Image.Image, length: float, seed: int) -> Image.Image:
    """Camera-shake motion blur: a straight path of `length` px at a seeded random angle."""
    from scipy.signal import fftconvolve
    angle = np.random.default_rng(seed).uniform(0.0, 180.0)
    k = motion_kernel(length, angle)
    arr = np.asarray(img, dtype=np.float64)
    p = k.shape[0] // 2
    padded = np.pad(arr, ((p, p), (p, p), (0, 0)), mode="reflect")
    out = np.stack([fftconvolve(padded[..., ch], k, mode="valid") for ch in range(3)], axis=-1)
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def jpeg(img: Image.Image, quality: float) -> Image.Image:
    """JPEG round trip at `quality` (Pillow encoder, 4:2:0 chroma subsampling)."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply(img: Image.Image, kind: str, param: float, seed: int) -> Image.Image:
    """Apply `kind` with an explicit parameter (sigma for blur, std for noise, see ALL_LEVELS)."""
    img = img.convert("RGB")
    if kind == "blur":
        return gaussian_blur(img, param)
    if kind == "noise":
        return gaussian_noise(img, param, seed)
    if kind == "lowlight":
        return low_light(img, param, seed)
    if kind == "motion":
        return motion_blur(img, param, seed)
    if kind == "jpeg":
        return jpeg(img, param)
    raise ValueError(f"unknown degradation kind: {kind}")


def degrade(img: Image.Image, kind: str, level: int, image_name: str) -> Image.Image:
    """Apply `kind` at severity `level` (1-based). `image_name` seeds stochastic degradations."""
    return apply(img, kind, ALL_LEVELS[kind][level - 1], seed_for(image_name, kind, level))
