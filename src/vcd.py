"""VCD building blocks, ported from DAMO-NLP-SG/VCD
(vcd_utils/vcd_add_noise.py and the contrastive step in vcd_utils/vcd_sample.py)."""
import torch

NUM_DIFFUSION_STEPS = 1000


def _noise_schedule():
    betas = torch.sigmoid(torch.linspace(-6, 6, NUM_DIFFUSION_STEPS)) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1 - betas, dim=0)
    return alphas_prod.sqrt(), (1 - alphas_prod).sqrt()


ALPHAS_BAR_SQRT, ONE_MINUS_ALPHAS_BAR_SQRT = _noise_schedule()


def add_diffusion_noise(pixel_values: torch.Tensor, noise_step: int, generator: torch.Generator | None = None):
    """Forward-diffuse a normalized image tensor to step `noise_step` (0-999):
    x_t = sqrt(a_bar_t) * x_0 + sqrt(1 - a_bar_t) * eps."""
    noise = torch.randn(pixel_values.shape, generator=generator, dtype=pixel_values.dtype)
    return ALPHAS_BAR_SQRT[noise_step] * pixel_values + ONE_MINUS_ALPHAS_BAR_SQRT[noise_step] * noise


def plausibility_mask(logits: torch.Tensor, beta: float) -> torch.Tensor:
    """True for tokens to drop: p(y|v,x) < beta * max_w p(w|v,x), computed in logit space
    as in the reference ('version 2' cutoff)."""
    cutoff = torch.log(torch.tensor(beta, dtype=logits.dtype, device=logits.device))
    return logits < cutoff + logits.max(dim=-1, keepdim=True).values


def vcd_logits(logits: torch.Tensor, logits_cd: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    diffs = (1 + alpha) * logits - alpha * logits_cd
    return diffs.masked_fill(plausibility_mask(logits, beta), -float("inf"))
