"""CPU tests for ViewLayer+DAE (no weights needed).   python -m pytest ViewLayerDAE/tests -q"""
import numpy as np
import pandas as pd
import torch

from scripts.run_pope import seeded_generator
from src.vcd import add_diffusion_noise
from ViewLayer.analyze import score as viewlayer_score
from ViewLayer.run_viewlayer import VIEWS
from ViewLayerDAE.analyze import ALPHAS, LAYERS, VIEW_SETS, best_threshold, dev_sweep, margins, score, select
from ViewLayerDAE.run_viewlayer_dae import EXTRA_VIEWS, contrast_pixel_values


def fake_cell(n=40, seed=0):
    """Records where the dae view separates the labels best and the cd view carries no signal."""
    rng = np.random.default_rng(seed)
    gold = rng.random(n) < 0.5
    sep = {"dae": 3.0, "cd": 0.0}
    cols = {f"m_{v}": [list(sep.get(v, 1.0) * g + rng.normal(size=32)) for g in gold] for v in VIEWS + EXTRA_VIEWS}
    return pd.DataFrame({"label": np.where(gold, "yes", "no"), **cols})


def test_contrast_noise_matches_stage2():
    pv = torch.randn(1, 3, 8, 8)
    expected = add_diffusion_noise(pv, 999, seeded_generator(55, "popular", 123, "noise"))
    torch.testing.assert_close(contrast_pixel_values(pv, "popular", 123), expected, rtol=0, atol=0)
    assert not torch.equal(contrast_pixel_values(pv, "popular", 124), expected)


def test_score_is_vcd_formula_and_falls_back_to_viewlayer():
    df = fake_cell()
    M = margins(df)
    views = ("orig", "blur2", "hflip")
    np.testing.assert_allclose(score(M, views, 20, 0.0), viewlayer_score(df, views, 20))
    pos = np.mean([M[v][:, 19] for v in views], axis=0)
    np.testing.assert_allclose(score(M, views, 20, 1.0), 2 * pos - M["cd"][:, 19])
    # one view, final layer, alpha = 1: the unmasked VCD margin 2 * real - contrast
    np.testing.assert_allclose(score(M, ("orig",), 32, 1.0), 2 * M["orig"][:, 31] - M["cd"][:, 31])


def test_search_space_contains_viewlayer_and_alpha_zero():
    assert 0.0 in ALPHAS and VIEWS in VIEW_SETS and ("orig",) in VIEW_SETS
    assert all("cd" not in vs for vs in VIEW_SETS)      # the contrast image is never a positive view
    assert len(set(VIEW_SETS)) == len(VIEW_SETS)


def test_selection_variants_respect_their_search_spaces():
    dev = {i: (margins(df), (df["label"] == "yes").to_numpy()) for i, df in enumerate([fake_cell(seed=1), fake_cell(seed=2)])}
    sweep = dev_sweep(dev)
    assert len(sweep) == len(VIEW_SETS) * len(LAYERS) * len(ALPHAS)
    v = select(sweep)
    assert v["vl_dae"][3] == sweep["dev_auroc"].max()
    assert v["vl_dae"][3] >= v["no_contrast"][3] >= v["viewlayer"][3] >= v["baseline_devthr"][3]
    assert "dae" in v["no_contrast"][0]                 # the informative view is found
    assert v["no_contrast"][2] == 0 and v["viewlayer"][2] == 0 and "dae" not in v["viewlayer"][0]
    assert v["forced_contrast"][2] > 0 and v["vcd_layer"][2] > 0 and v["vcd_layer"][0] == ("orig",)
    assert v["baseline_devthr"][:3] == (("orig",), 32, 0.0)


def test_best_threshold_separates():
    s, g = np.array([3.0, -1.0, 2.0, 0.0]), np.array([True, False, True, False])
    assert 0 <= best_threshold([s], [g]) < 2
