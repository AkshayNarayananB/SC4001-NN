"""Physically parameterised capture simulator for the sensing-vs-compute study ("Photons or FLOPs?").

A clean sRGB image is treated as the scene's radiance (linearised, white = 1). A capture is described by
  light    peak photo-electrons per millisecond of exposure at white (scene illuminance x optics x QE)
  speed    camera ego-motion in native-resolution pixels per millisecond (constant velocity, seeded direction)
  t_ms     exposure time of one frame
  n        number of back-to-back frames in a burst (frame k covers [k*t, (k+1)*t))
  fusion   "single" (n == 1), "naive" (plain average: motion between frames adds blur), or "aligned" (each frame
           registered to frame 0 by phase correlation on the noisy frames, then averaged), or "oracle"
           (registered with the true offset: upper bound of any alignment)

Per frame: motion blur of length speed*t along the trajectory, scene shifted by k*speed*t, then
electrons = Poisson(light * t * radiance) + N(0, READ_NOISE_E) (read noise is paid once PER FRAME, which is why
bursts of very short frames lose to one long frame in the dark). Fusion averages electrons in the linear (RAW-like)
domain, then a simple ISP applies auto-gain (1 / (light * t)), clips, applies the sRGB curve and quantises to 8 bit.

Determinism and prefix consistency: frame k of a burst depends only on (image, light, speed, t_ms, k), so the
first k frames of an n-frame burst are exactly a k-frame burst. A sequential capture policy can therefore be
evaluated offline from logged prefixes. Bayer mosaic / demosaicking are not modelled (covered by the real-RAW data).
"""
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.signal import fftconvolve

from src.degradations import READ_NOISE_E, linear_to_srgb, motion_kernel, srgb_to_linear

T_REF_MS = 33.3          # one video frame at 30 fps: the reference exposure


@dataclass(frozen=True)
class Capture:
    light: float          # e-/ms at white
    speed: float          # px/ms
    t_ms: float
    n: int = 1
    fusion: str = "single"

    @property
    def total_ms(self) -> float:
        return self.t_ms * self.n

    @property
    def name(self) -> str:
        return f"L{self.light:g}_v{self.speed:g}_t{self.t_ms:g}_n{self.n}_{self.fusion}"


def _seed(image_name: str, *parts) -> int:
    return zlib.crc32("|".join([image_name, *map(str, parts)]).encode())


def _direction(image_name: str) -> float:
    """Ego-motion direction in degrees, fixed per image (the same camera move for every capture setting)."""
    return float(np.random.default_rng(_seed(image_name, "dir")).uniform(0.0, 180.0))


def shift_img(x: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Bilinear translation by (dy, dx) px with reflect boundary (out[y, x] = x[y - dy, x - dx]).
    Same result as scipy.ndimage.shift(order=1, mode="reflect") up to boundary handling, ~10x faster."""
    if dy == 0 and dx == 0:
        return x
    H, W = x.shape[:2]
    P = int(np.ceil(max(abs(dy), abs(dx)))) + 2
    pad = np.pad(x, ((P, P), (P, P), (0, 0)), mode="symmetric")
    iy, ix = int(np.floor(dy)), int(np.floor(dx))
    fy, fx = dy - iy, dx - ix
    def win(oy, ox):
        return pad[P - oy:P - oy + H, P - ox:P - ox + W]
    return ((1 - fy) * (1 - fx) * win(iy, ix) + (1 - fy) * fx * win(iy, ix + 1)
            + fy * (1 - fx) * win(iy + 1, ix) + fy * fx * win(iy + 1, ix + 1))


def _blur_shift(lin: np.ndarray, length: float, angle: float, shift_px: float) -> np.ndarray:
    """Blur by a motion path of `length` px at `angle`, then translate the scene by `shift_px` along it."""
    out = lin
    if length >= 1.0:
        k = motion_kernel(length, angle)
        p = k.shape[0] // 2
        padded = np.pad(lin, ((p, p), (p, p), (0, 0)), mode="reflect")
        out = np.stack([fftconvolve(padded[..., c], k, mode="valid") for c in range(3)], axis=-1)
    if shift_px:
        a = np.deg2rad(angle)
        out = shift_img(out, -shift_px * np.sin(a), shift_px * np.cos(a))
    return out


def frames(img: Image.Image, image_name: str, light: float, speed: float, t_ms: float, n: int):
    """The n raw frames (electrons, float64 HxWx3) of a burst and their true offsets (dy, dx) in px."""
    lin = srgb_to_linear(np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0)
    angle = _direction(image_name)
    step = speed * t_ms
    blurred = _blur_shift(lin, step, angle, 0.0)     # within-frame blur is the same for every frame
    a = np.deg2rad(angle)
    out, offs = [], []
    for k in range(n):
        scene = _blur_shift(blurred, 0.0, angle, k * step) if k else blurred
        rng = np.random.default_rng(_seed(image_name, light, speed, t_ms, k))
        e = rng.poisson(np.clip(scene, 0, None) * light * t_ms) + rng.normal(0.0, READ_NOISE_E, size=scene.shape)
        out.append(e)
        offs.append((-k * step * np.sin(a), k * step * np.cos(a)))
    return out, offs


ALIGN_DOWN = 4           # box-downsampling before registration (as in coarse levels of burst pipelines)
ALIGN_MAX_PX = 96        # search window: |shift| <= 96 native px


def estimate_shift(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float]:
    """Translation of `mov` relative to `ref`: cross-correlation of box-downsampled, lightly smoothed luminance
    within a +-ALIGN_MAX_PX window, parabolic sub-pixel peak. Uses only the noisy frames, so it fails in the dark
    like a real burst pipeline would."""
    f = ALIGN_DOWN

    def prep(x):
        y = x.mean(-1)
        h, w = (y.shape[0] // f) * f, (y.shape[1] // f) * f
        y = y[:h, :w].reshape(h // f, f, w // f, f).mean((1, 3))
        y = ndimage.gaussian_filter(y, 1.0)
        y = y - y.mean()
        return y * np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1]))
    a, b = prep(ref), prep(mov)
    r = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    r = np.fft.fftshift(r)
    cy, cx = r.shape[0] // 2, r.shape[1] // 2
    m = ALIGN_MAX_PX // f
    win = r[cy - m:cy + m + 1, cx - m:cx + m + 1]
    iy, ix = np.unravel_index(np.argmax(win), win.shape)

    def para(lo, c, hi):
        den = lo - 2 * c + hi
        return 0.5 * (lo - hi) / den if den != 0 else 0.0
    dy = iy + (para(win[iy - 1, ix], win[iy, ix], win[iy + 1, ix]) if 0 < iy < win.shape[0] - 1 else 0.0) - m
    dx = ix + (para(win[iy, ix - 1], win[iy, ix], win[iy, ix + 1]) if 0 < ix < win.shape[1] - 1 else 0.0) - m
    return -dy * f, -dx * f     # offset of mov w.r.t. ref, same convention as `frames`


def fuse(fr: list, offs: list, fusion: str) -> np.ndarray:
    if fusion in ("single", "naive") or len(fr) == 1:
        return np.mean(fr, axis=0)
    acc = fr[0].copy()
    for e, true in zip(fr[1:], offs[1:]):
        dy, dx = true if fusion == "oracle" else estimate_shift(fr[0], e)
        acc += shift_img(e, -dy, -dx)
    return acc / len(fr)


def isp(electrons: np.ndarray, light: float, t_ms: float) -> Image.Image:
    """Auto-gain to the per-frame exposure, clip, sRGB, 8-bit."""
    out = linear_to_srgb(np.clip(electrons / (light * t_ms), 0.0, 1.0))
    return Image.fromarray(np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8))


def capture(img: Image.Image, image_name: str, c: Capture) -> Image.Image:
    fr, offs = frames(img, image_name, c.light, c.speed, c.t_ms, c.n)
    return isp(fuse(fr, offs, c.fusion), c.light, c.t_ms)


def capture_prefixes(img: Image.Image, image_name: str, light: float, speed: float, t_ms: float,
                     ns: tuple, fusions: tuple) -> dict:
    """All requested burst lengths and fusions from ONE set of frames (cheap sweep): {(n, fusion): image}."""
    fr, offs = frames(img, image_name, light, speed, t_ms, max(ns))
    out = {}
    for n in ns:
        for f in (("single",) if n == 1 else fusions):
            out[(n, f)] = isp(fuse(fr[:n], offs[:n], f), light, t_ms)
    return out
