"""Uncertainty-gated multi-view readout: ViewLayer's signal at a fraction of its compute.

    python -m scripts.analyze_gated [--conditions blur_s4 noise_s4]

Method (one forward pass to decide, extra passes only when needed):
  1. Run the original image, stopped at layer L (default 20): margin m0 = Yes-No logit at layer L.
  2. If |m0 - c| > tau (confident), answer from m0 alone.
     Otherwise (uncertain), run the four extra views (blur sigma 1/2/3, hflip), also stopped at layer L,
     and answer from the mean of the five margins (ViewLayer's score).
  c and tau are chosen on the 100 DEV images only, as one global pair for all conditions.

Compute, in full-forward-pass units (a pass stopped at layer L costs L/32):
  ViewLayer           5 full passes                     = 5.0 (run at layer 20, not stopped: 5 x 1.0)
  gated method        L/32 + p_uncertain * 4 * L/32     (p_uncertain measured on TEST)

Scoring follows ViewLayer/analyze.py: AUROC on TEST, F1 against a baseline thresholded to the method's
yes-ratio (matched), paired bootstrap. Pilot: only the conditions with view runs (blur s4, noise s4) and their
clean/orig readouts are used, so DEV choices here are from those two conditions only.
"""
import argparse

import numpy as np
import pandas as pd

from scripts.analyze_layers import auroc, f1_from, load_cells, matched_f1, to_arrays, dev_images
from scripts.run_layers import run_dir_name, CONDITIONS, cond_name
from scripts.run_pope import OUTPUT_ROOT
from src.paths import POPE_SPLITS, REPO_ROOT
import json

VIEWS = ["blur1", "blur2", "blur3", "hflip"]
L = 20
FINAL = 32


def load_views(kind, level, split):
    """Per-view layer margins aligned with the orig rows (same question order)."""
    out = {}
    for v in VIEWS:
        path = OUTPUT_ROOT / run_dir_name(kind, level, v) / f"{split}.jsonl"
        with open(path) as f:
            df = pd.DataFrame(json.loads(l) for l in f).sort_values("question_id").reset_index(drop=True)
        out[v] = (np.array(df["layer_yes"].tolist()) - np.array(df["layer_no"].tolist()))[:, L - 1]
    return out


def build(conditions):
    cells = load_cells()
    dev_set = dev_images(cells)
    rows = {}
    for (kind, level, split), df in cells.items():
        if cond_name(kind, level) not in conditions and not (kind == "clean"):
            continue
        views = None
        if kind != "clean" and cond_name(kind, level) in conditions:
            views = load_views(kind, level, split)
        orig = (np.array(df["layer_yes"].tolist()) - np.array(df["layer_no"].tolist()))[:, L - 1]
        part = dict(orig=orig, gold=(df["label"] == "yes").to_numpy(), image=df["image"].to_numpy(),
                    views=views, split_dev=df["image"].isin(dev_set).to_numpy())
        rows[(kind, level, split)] = part
    return rows


def score(part, c, tau, gate=True):
    """Gated score and the fraction of questions that ran the extra views."""
    m0 = part["orig"]
    if not gate or part["views"] is None:
        return m0, 0.0
    uncertain = np.abs(m0 - c) <= tau
    stack = np.stack([m0] + [part["views"][v] for v in VIEWS], axis=1)
    mean5 = stack.mean(1)
    return np.where(uncertain, mean5, m0), uncertain.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["blur_s4", "noise_s4"])
    parser.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()
    parts = build(args.conditions)
    view_keys = [k for k, p in parts.items() if p["views"] is not None]
    if not view_keys:
        raise SystemExit("no view runs found")
    dev = {k: {f: (v[p["split_dev"]] if isinstance(v, np.ndarray) else None) for f, v in p.items()} for k, p in parts.items()}
    test = {k: {f: (v[~p["split_dev"]] if isinstance(v, np.ndarray) else None) for f, v in p.items()} for k, p in parts.items()}
    for k in test:
        test[k]["views"] = None if parts[k]["views"] is None else {v: a[~parts[k]["split_dev"]] for v, a in parts[k]["views"].items()}
        dev[k]["views"] = None if parts[k]["views"] is None else {v: a[parts[k]["split_dev"]] for v, a in parts[k]["views"].items()}

    # ---- DEV: choose (c, tau) to maximise mean DEV AUROC over the view-covered cells
    dev_cells = [k for k in dev if dev[k]["views"] is not None]
    cs = np.quantile(np.concatenate([dev[k]["orig"] for k in dev_cells]), np.linspace(0.3, 0.7, 9))
    taus = np.quantile(np.abs(np.concatenate([dev[k]["orig"] for k in dev_cells])), np.linspace(0.1, 0.95, 18))
    best = None
    for c in cs:
        for tau in taus:
            aucs = [auroc(score(dev[k], c, tau)[0], dev[k]["gold"]) for k in dev_cells]
            if best is None or np.mean(aucs) > best[0]:
                best = (np.mean(aucs), c, tau)
    dev_auc, c, tau = best
    print(f"DEV choice: c={c:.3f} tau={tau:.3f} (mean DEV AUROC {dev_auc:.4f}) over {dev_cells}")

    # ---- TEST
    rng = np.random.default_rng(0)
    rows = []
    for k in test:
        if test[k]["views"] is None and k[0] != "clean":
            continue
        g = test[k]["gold"]
        m_vl = np.mean([test[k]["orig"]] + [test[k]["views"][v] for v in VIEWS], axis=0) if test[k]["views"] else None
        m_gated, p_unc = score(test[k], c, tau)
        m_base = test[k]["orig"]
        idx = rng.integers(0, len(g), size=(args.boot, len(g)))
        for name, m, passes in [("Baseline (layer 32)", None, 1.0),
                                ("Layer 20 only", m_base, 20 / 32),
                                ("ViewLayer (5 views, L20)", m_vl, 5.0),
                                ("Gated ViewLayer (uncertainty-gated)", m_gated, 20 / 32 + p_unc * 4 * 20 / 32)]:
            if m is None or (name.startswith(("ViewLayer", "Gated")) and k[0] == "clean" and test[k]["views"] is None):
                continue
            y_matched = matched_f1(m_base, (m > 0).mean(), g)
            d = np.array([auroc(m[i], g[i]) - auroc(m_base[i], g[i]) for i in idx])
            rows.append(dict(method=name, cond=f"{k[0]} s{k[1]}" if k[0] != "clean" else "clean", split=k[2],
                             auroc=auroc(m, g), d_auroc_lo=np.percentile(d, 2.5), d_auroc_hi=np.percentile(d, 97.5),
                             f1=f1_from(m > 0, g), f1_matched=y_matched, passes=passes,
                             p_uncertain=p_unc if "Gated" in name else np.nan))
    res = pd.DataFrame(rows)
    res.to_csv(REPO_ROOT / "results" / "gated_pilot.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res.groupby(["method", "cond"], sort=False)[["auroc", "d_auroc_lo", "d_auroc_hi", "f1", "f1_matched", "passes", "p_uncertain"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
