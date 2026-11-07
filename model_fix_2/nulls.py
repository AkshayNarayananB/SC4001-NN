"""Content-free "null" images for the contrast branch of VCD.

Stage 2 found that VCD's diffusion-noise contrast image (T=999) is not blind: on clean images its
Yes/No margin still separates present from absent objects (AUROC 0.69). Subtracting it removes
real evidence, and VCD lowers AUROC in every cell. The fix replaces it with a *null* image: one
that carries no object evidence but keeps the input's low-level statistics, so the contrast
subtracts the language prior (and the image-level bias that degradation adds) without
subtracting what the image actually shows.

    phase  Fourier phase scrambling: keep the degraded input's amplitude spectrum (blur level,
           noise level, contrast, colour, texture energy) and randomise its phase (which carries
           object structure; Oppenheim & Lim 1981). A blurred input gives a blurry texture, a
           noisy input a noisy texture, and neither contains objects. THE FIX.
    mean   every pixel set to the input's per-channel mean: content-free and colour-matched but
           with no spectrum. Control that separates "content-free" from "statistics-matched".

All operations act on the normalized pixel tensor the model sees (after resize, crop and
normalization), the same tensor VCD adds its noise to.
"""
import zlib

import torch

# CLIP normalization used by LLaVA-1.5's image processor
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
NULLS = ("phase", "mean")


def null_generator(image_name: str, null: str, seed: int) -> torch.Generator:
    """Seeded by image name only (not degradation level or split), so the same photo gets the
    same random phase in every condition and the comparison across levels is paired."""
    return torch.Generator().manual_seed(zlib.crc32(f"{image_name}|{null}|{seed}".encode()))


def clip_to_valid(pixel_values: torch.Tensor) -> torch.Tensor:
    """Clamp a normalized tensor to the range a real [0, 1] image can produce."""
    lo, hi = (0 - CLIP_MEAN) / CLIP_STD, (1 - CLIP_MEAN) / CLIP_STD
    return torch.maximum(torch.minimum(pixel_values, hi), lo)


def random_phase(h: int, w: int, gen: torch.Generator) -> torch.Tensor:
    """Phase of the FFT of real white noise: uniform on (-pi, pi] and Hermitian-antisymmetric, so
    adding it to a real image's phase keeps the inverse FFT real. DC phase forced to 0 so the
    mean is untouched."""
    phase = torch.angle(torch.fft.fft2(torch.randn(h, w, generator=gen, dtype=torch.float64)))
    phase[0, 0] = 0.0
    return phase


def phase_scramble(pixel_values: torch.Tensor, gen: torch.Generator, strength: float = 1.0,
                   clip: bool = True) -> torch.Tensor:
    """Randomise Fourier phase, keep amplitude. One random phase field is shared by the three
    colour channels, which preserves cross-channel (colour) correlations. strength=1 gives a
    fully random phase; strength=0 returns the input."""
    x = pixel_values.detach().to(torch.float64).cpu()
    spec = torch.fft.fft2(x)
    phase = random_phase(x.shape[-2], x.shape[-1], gen)
    out = torch.fft.ifft2(torch.abs(spec) * torch.exp(1j * (torch.angle(spec) + strength * phase))).real
    out = out.to(pixel_values.dtype)
    return (clip_to_valid(out) if clip else out).to(pixel_values.device)


def mean_image(pixel_values: torch.Tensor) -> torch.Tensor:
    return pixel_values.mean(dim=(-2, -1), keepdim=True).expand_as(pixel_values).clone()


def make_null(null: str, pixel_values: torch.Tensor, image_name: str, seed: int = 0) -> torch.Tensor:
    if null == "phase":
        return phase_scramble(pixel_values, null_generator(image_name, null, seed))
    if null == "mean":
        return mean_image(pixel_values)
    raise ValueError(f"unknown null {null!r}; choose from {NULLS}")
