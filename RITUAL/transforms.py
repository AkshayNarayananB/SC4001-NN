"""RITUAL's random image transformations (Woo et al. 2024, arXiv 2405.17821, Sec. 4.1 / App. E.1).

Six transforms; RITUAL applies ONE, chosen at random, per image:
    hflip, vflip
    rotate          angle ~ U(-180, 180) degrees
    color_jitter    brightness=1, contrast=1, saturation=1, hue=0.5 (torchvision ColorJitter semantics:
                    factors ~ U(0, 2), hue shift ~ U(-0.5, 0.5), applied in random order)
    gaussian_blur   kernel 13, sigma ~ U(1.5, 2.0)
    crop            random 336x336 crop
Applied to the (degraded) PIL image at native resolution, before LLaVA's processor, like the
reference pipeline. All randomness is seeded by (image name, transform, seed), so the same photo
gets the same transform and parameters at every degradation level and in every POPE split.
"""
import zlib

import torch
import torchvision.transforms.functional as F
from PIL import Image

TRANSFORMS = ("hflip", "vflip", "rotate", "color_jitter", "gaussian_blur", "crop")
CROP_SIZE = 336


def _gen(*parts) -> torch.Generator:
    return torch.Generator().manual_seed(zlib.crc32("|".join(map(str, parts)).encode()))


def _uniform(gen: torch.Generator, lo: float, hi: float) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=gen).item()


def choose_transform(image_name: str, seed: int = 0) -> str:
    """RITUAL's per-image random choice."""
    return TRANSFORMS[int(torch.randint(len(TRANSFORMS), (1,), generator=_gen(image_name, "choice", seed)))]


def apply_transform(img: Image.Image, name: str, image_name: str, seed: int = 0) -> Image.Image:
    gen = _gen(image_name, name, seed)
    if name == "hflip":
        return F.hflip(img)
    if name == "vflip":
        return F.vflip(img)
    if name == "rotate":
        return F.rotate(img, _uniform(gen, -180.0, 180.0))
    if name == "color_jitter":
        ops = [lambda x, f=_uniform(gen, 0.0, 2.0): F.adjust_brightness(x, f),
               lambda x, f=_uniform(gen, 0.0, 2.0): F.adjust_contrast(x, f),
               lambda x, f=_uniform(gen, 0.0, 2.0): F.adjust_saturation(x, f),
               lambda x, f=_uniform(gen, -0.5, 0.5): F.adjust_hue(x, f)]
        for i in torch.randperm(4, generator=gen).tolist():
            img = ops[i](img)
        return img
    if name == "gaussian_blur":
        return F.gaussian_blur(img, kernel_size=13, sigma=_uniform(gen, 1.5, 2.0))
    if name == "crop":
        w, h = img.size
        size = min(CROP_SIZE, w, h)
        top = int(torch.randint(h - size + 1, (1,), generator=gen))
        left = int(torch.randint(w - size + 1, (1,), generator=gen))
        return F.crop(img, top, left, size, size)
    raise ValueError(f"unknown transform {name!r}; choose from {TRANSFORMS}")
