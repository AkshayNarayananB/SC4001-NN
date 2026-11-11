"""Denoising autoencoder for Phase 2, fix B: a small U-Net that maps a degraded RGB image to a
restored one.

It runs once per image at native resolution, before LLaVA's preprocessing, so the VCD real
branch sees the restored image (and so does the contrast branch, which noises the real branch's
input). Trained by scripts/train_dae.py on COCO train2014 only; POPE uses val2014.

Design:
- U-Net with skip connections, not a bottleneck autoencoder: a plain bottleneck blurs away exactly
  the small-object detail POPE asks about.
- Residual output (x + f(x)) with a zero-initialised last layer, so the network starts as the
  identity and clean images pass through unchanged unless training says otherwise.
- No normalisation layers, so the output doesn't depend on batch composition.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True),
    )


class Up(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 1)
        self.block = conv_block(2 * cout, cout)

    def forward(self, z, skip):
        # bilinear upsampling + conv avoids the checkerboard artefacts of transposed convs
        z = F.interpolate(z, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([self.reduce(z), skip], dim=1))


class DAE(nn.Module):
    def __init__(self, base: int = 32, depth: int = 4):
        super().__init__()
        self.config = dict(base=base, depth=depth)
        chs = [base * 2**i for i in range(depth + 1)]
        self.inc = conv_block(3, chs[0])
        self.down = nn.ModuleList(nn.Sequential(nn.MaxPool2d(2), conv_block(chs[i], chs[i + 1])) for i in range(depth))
        self.up = nn.ModuleList(Up(chs[i + 1], chs[i]) for i in reversed(range(depth)))
        self.out = nn.Conv2d(chs[0], 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1], any H and W. Returns the restored image, unclamped."""
        h, w = x.shape[-2:]
        m = 2 ** self.config["depth"]
        z = F.pad(x, (0, (-w) % m, 0, (-h) % m), mode="reflect") - 0.5
        z = self.inc(z)
        skips = []
        for down in self.down:
            skips.append(z)
            z = down(z)
        for up in self.up:
            z = up(z, skips.pop())
        return x + self.out(z)[..., :h, :w]


def to_tensor(img) -> torch.Tensor:
    """PIL image or HxWx3 uint8 array -> (3, H, W) float32 in [0, 1]."""
    arr = np.asarray(img.convert("RGB") if isinstance(img, Image.Image) else img)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32) / 255.0).permute(2, 0, 1)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [0, 1] -> HxWx3 uint8, rounded as the degradations are."""
    return (t.detach().float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def save_dae(model: DAE, path: Path, **extra) -> None:
    torch.save({"config": model.config, "model": model.state_dict(), **extra}, path)


def load_dae(path: Path, device: torch.device | str = "cpu") -> DAE:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DAE(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.inference_mode()
def restore(model: DAE, img: Image.Image) -> Image.Image:
    """Restore one PIL image at native resolution (fp32, so the output is reproducible)."""
    device = next(model.parameters()).device
    out = model(to_tensor(img)[None].to(device))[0]
    return Image.fromarray(to_uint8(out))
