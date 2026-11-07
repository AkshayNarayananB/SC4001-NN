"""CPU tests for RITUAL (no weights needed).   python -m pytest RITUAL/tests -q"""
import numpy as np
import pandas as pd
import torch
from PIL import Image

from RITUAL.analyze import ritual_margin
from RITUAL.run_ritual import ritual_scores
from RITUAL.transforms import TRANSFORMS, apply_transform, choose_transform


def photo(w=640, h=480, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8))


def test_transforms_deterministic_and_sized():
    img = photo()
    for t in TRANSFORMS:
        a, b = apply_transform(img, t, "imgA"), apply_transform(img, t, "imgA")
        assert np.array_equal(np.asarray(a), np.asarray(b)), t
        assert not np.array_equal(np.asarray(a), np.asarray(img)), t
        assert a.size == ((336, 336) if t == "crop" else img.size), t
    # different images get different random parameters
    assert not np.array_equal(np.asarray(apply_transform(img, "rotate", "imgA")),
                              np.asarray(apply_transform(img, "rotate", "imgB")))


def test_flips_are_exact():
    img = photo()
    assert np.array_equal(np.asarray(apply_transform(img, "hflip", "x")), np.asarray(img)[:, ::-1])
    assert np.array_equal(np.asarray(apply_transform(img, "vflip", "x")), np.asarray(img)[::-1])


def test_choice_is_per_image_and_covers_all():
    picks = {choose_transform(f"COCO_{i}.jpg") for i in range(200)}
    assert picks == set(TRANSFORMS)
    assert choose_transform("COCO_1.jpg") == choose_transform("COCO_1.jpg")


def test_small_image_crop():
    assert apply_transform(photo(300, 250), "crop", "x").size == (250, 250)


def test_ritual_scores_formula_and_mask():
    p_real = torch.tensor([0.6, 0.35, 0.05])
    p_view = torch.tensor([0.1, 0.8, 0.1])
    s = ritual_scores(p_real, p_view, alpha=3.0, beta=0.1)
    torch.testing.assert_close(s[:2], p_real[:2] + 3 * p_view[:2])
    assert s[2] == -float("inf")        # 0.05 < 0.1 * 0.6
    assert int(s.argmax()) == 1          # the view flips the decision
    assert int(ritual_scores(p_real, p_view, alpha=0.0).argmax()) == 0


def test_margin_matches_exact_decision():
    rng = np.random.default_rng(0)
    rows, decisions = [], []
    for i in range(300):
        p_real = torch.softmax(torch.tensor(rng.normal(size=6) * 2), 0).float()
        views = {t: torch.softmax(torch.tensor(rng.normal(size=6) * 2), 0).float() for t in TRANSFORMS}
        p_real[0], p_real[1] = p_real[0] + 0.5, p_real[1] + 0.5      # Yes = 0, No = 1 dominate
        p_real = p_real / p_real.sum()
        t = TRANSFORMS[i % 6]
        tok = int(ritual_scores(p_real, views[t], 3.0).argmax())
        if tok not in (0, 1):
            continue
        row = dict(transform=t, maxprob_real=p_real.max().item(), p_yes_real=p_real[0].item(), p_no_real=p_real[1].item())
        for name, p in views.items():
            row[f"p_yes_{name}"], row[f"p_no_{name}"] = p[0].item(), p[1].item()
        rows.append(row)
        decisions.append(tok == 0)
    m = ritual_margin(pd.DataFrame(rows), "chosen", 3.0)
    assert len(rows) > 150 and ((m > 0) == np.array(decisions)).all()
