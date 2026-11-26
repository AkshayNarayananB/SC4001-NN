"""Real RAW frames through the SAME fusion + ISP as the simulator (src/capture.py), so real and simulated captures
differ only in where the photons came from.

  decode    rawpy (LibRaw): black/white-level normalisation, camera white balance, AHD demosaic at half size
            (one RGB pixel per Bayer quad, no interpolation blur), camera -> linear sRGB primaries, NO gamma,
            NO auto-brightness. Output: float linear RGB, white = 1 at the frame's own exposure.
  fuse      capture.fuse: "single", "naive" (plain mean) or "aligned" (capture.estimate_shift on the noisy frames,
            same estimator as in simulation; the search window is scaled to the frame size).
  isp       digital gain (exposure ratio for SID, auto-exposure to a target mean for HDR+), clip, sRGB, 8 bit,
            area-downsample to LONG_SIDE px (COCO-like native size; done in linear light before the sRGB curve).
"""
from pathlib import Path

import numpy as np
from PIL import Image

from src import capture as C
from src.degradations import linear_to_srgb

LONG_SIDE = 640
AE_TARGET_MEAN = 0.18      # HDR+ auto-exposure: linear mean of the frame after gain (18% grey)


def decode(path) -> np.ndarray:
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True, gamma=(1, 1),
                              output_bps=16, output_color=rawpy.ColorSpace.sRGB, user_flip=0)
    return rgb.astype(np.float64) / 65535.0


def downsample(lin: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    h, w = lin.shape[:2]
    f = max(h, w) / long_side
    if f <= 1:
        return lin
    nh, nw = int(round(h / f)), int(round(w / f))
    chans = [np.asarray(Image.fromarray(lin[..., c].astype(np.float32), mode="F").resize((nw, nh), Image.BOX))
             for c in range(3)]
    return np.stack(chans, -1).astype(np.float64)


def fuse_frames(frames: list, fusion: str, align_scale: float = 1.0) -> np.ndarray:
    """Average linear frames (k >= 1). Alignment uses the simulator's estimator; its window and downsampling
    factor are scaled by `align_scale` (frame width / 640) so that it searches the same field-of-view fraction."""
    if fusion == "single" or len(frames) == 1:
        return frames[0]
    if fusion == "naive":
        return np.mean(frames, 0)
    old = C.ALIGN_DOWN, C.ALIGN_MAX_PX
    try:
        C.ALIGN_DOWN = max(1, int(round(old[0] * align_scale)))
        C.ALIGN_MAX_PX = int(round(old[1] * align_scale))
        acc = frames[0].copy()
        for f in frames[1:]:
            dy, dx = C.estimate_shift(frames[0], f)
            acc += C.shift_img(f, -dy, -dx)
        return acc / len(frames)
    finally:
        C.ALIGN_DOWN, C.ALIGN_MAX_PX = old


def isp(lin: np.ndarray, gain: float) -> Image.Image:
    small = downsample(np.clip(lin * gain, 0, None))
    out = linear_to_srgb(np.clip(small, 0.0, 1.0))
    return Image.fromarray(np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8))


def ae_gain(lin: np.ndarray) -> float:
    """Auto-exposure gain from one frame (robust mean of a 1/8-size copy), as a camera's AE would set it."""
    m = float(np.mean(lin[::8, ::8]))
    return AE_TARGET_MEAN / max(m, 1e-6)


def hdrplus_burst(burst_dir: Path) -> list:
    return sorted(Path(burst_dir).glob("payload_N*.dng"))
