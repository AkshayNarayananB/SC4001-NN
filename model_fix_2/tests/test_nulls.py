"""CPU tests for the null images and the contrast-margin recomputation.

    python -m pytest model_fix_2/tests -q
"""
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from model_fix_2.analyze import contrast_margin, matched_f1
from model_fix_2.nulls import clip_to_valid, make_null, mean_image, null_generator, phase_scramble
from src.vcd import vcd_logits


def smooth_image(seed=0, size=64, sigma=3.0):
    """A structured test image: blurred noise with a bright square (an 'object')."""
    g = np.random.default_rng(seed)
    x = gaussian_filter(g.normal(size=(3, size, size)), sigma=(0, sigma, sigma))
    x[:, 20:36, 20:36] += 2.0
    return torch.tensor(x, dtype=torch.float32)[None]


def amplitude(x):
    return torch.abs(torch.fft.fft2(x.double()))


def test_phase_scramble_keeps_amplitude_and_mean_and_is_real():
    x = smooth_image()
    y = phase_scramble(x, null_generator("img", "phase", 0), clip=False)
    assert y.dtype == x.dtype and y.shape == x.shape
    torch.testing.assert_close(amplitude(y), amplitude(x), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(y.mean(dim=(-2, -1)), x.mean(dim=(-2, -1)), rtol=1e-5, atol=1e-5)


def test_phase_scramble_destroys_structure():
    x = smooth_image()
    y = phase_scramble(x, null_generator("img", "phase", 0), clip=False)
    corr = np.corrcoef(x.flatten().numpy(), y.flatten().numpy())[0, 1]
    assert abs(corr) < 0.15, corr
    # the "object" square is no longer brighter than the rest
    inside, outside = y[..., 20:36, 20:36].mean(), y.mean()
    assert abs(inside - outside) < 0.5 * (x[..., 20:36, 20:36].mean() - x.mean())


def test_phase_scramble_keeps_blur_level():
    """A blurrier input gives a blurrier null: high-frequency energy is preserved per input."""
    def hf_energy(img):
        a = amplitude(img)[..., 16:48, 16:48]  # high-frequency band of a 64x64 FFT
        return float((a ** 2).sum())
    gen = lambda: null_generator("img", "phase", 0)
    for sigma in (1.0, 4.0):
        x = smooth_image(sigma=sigma)
        y = phase_scramble(x, gen(), clip=False)
        assert abs(hf_energy(y) / hf_energy(x) - 1) < 1e-3


def test_phase_scramble_strength_zero_is_identity_and_seeded():
    x = smooth_image()
    torch.testing.assert_close(phase_scramble(x, null_generator("a", "phase", 0), strength=0.0, clip=False), x,
                               rtol=1e-5, atol=1e-5)
    a = make_null("phase", x, "imgA")
    assert torch.equal(a, make_null("phase", x, "imgA"))       # deterministic per image
    assert not torch.equal(a, make_null("phase", x, "imgB"))   # different images, different phase


def test_shared_phase_across_channels_keeps_colour_correlation():
    x = smooth_image()
    x[:, 1] = x[:, 0] * 0.8 + 0.1  # green tied to red
    y = phase_scramble(x, null_generator("img", "phase", 0), clip=False)
    corr = np.corrcoef(y[0, 0].flatten().numpy(), y[0, 1].flatten().numpy())[0, 1]
    assert corr > 0.99


def test_mean_null_and_clipping():
    x = smooth_image()
    m = mean_image(x)
    torch.testing.assert_close(m[..., 5, 7], x.mean(dim=(-2, -1)))
    assert m.std(dim=(-2, -1)).max() < 1e-6
    big = torch.full((1, 3, 4, 4), 10.0)
    assert clip_to_valid(big).max() < 2.2 and clip_to_valid(-big).min() > -1.9


def test_contrast_margin_matches_full_vcd_rule():
    """The recomputed first-token margin agrees with argmax of the full-vocab VCD logits whenever
    the real branch's top-1 is Yes or No."""
    g = torch.Generator().manual_seed(0)
    yes, no, rows, decisions = 5, 9, [], []
    for _ in range(300):
        real, null = torch.randn(1, 20, generator=g) * 3, torch.randn(1, 20, generator=g) * 3
        real[0, [yes, no]] += 8  # make Yes/No the top tokens
        lp = torch.log_softmax(real, -1)
        tok = int(vcd_logits(real, null, 1.0, 0.1).argmax())
        if tok not in (yes, no):
            continue
        rows.append(dict(logit_yes_real=real[0, yes].item(), logit_no_real=real[0, no].item(),
                         logit_yes_phase=null[0, yes].item(), logit_no_phase=null[0, no].item(),
                         logprob_yes_real=lp[0, yes].item(), logprob_no_real=lp[0, no].item()))
        decisions.append(tok == yes)
    df = pd.DataFrame(rows)
    m = contrast_margin(df, "logit_yes_real", "logit_no_real", "logit_yes_phase", "logit_no_phase",
                        "logprob_yes_real", "logprob_no_real", 1.0)
    assert len(df) > 200 and ((m > 0) == np.array(decisions)).all()


def test_matched_f1_reaches_target_yes_ratio():
    m = np.linspace(-1, 1, 1000)
    gold = m > 0.2
    assert matched_f1(m, 0.4, gold) == 1.0  # threshold at the 60th percentile = 0.2
