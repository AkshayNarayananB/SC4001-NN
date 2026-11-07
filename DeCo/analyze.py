"""Score DeCo against baseline, VCD and the matched-threshold baseline; write the README results.

    python -m DeCo.analyze [--boot 500]

Variants (greedy, first token):
    deco       as published, alpha = 0.6 (official default)
    deco_dev   alpha chosen on DEV images (max mean AUROC)
Layer-wise analysis on TEST: AUROC of each layer's own Yes-No margin, per condition. Tests DeCo's
premise on degraded images: is the evidence still present in the middle layers?
Outputs: results/deco_alpha_dev.csv, results/deco_comparison.csv, results/deco_layer_auroc.csv,
and the Results section of DeCo/README.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from DeCo.deco import ALPHA as PAPER_ALPHA
from DeCo.run_deco import ALPHAS
from fix_eval.common import (auroc, decision_score, dev_images, f1_from, fix_metrics, load_method_cells, reference_row, summarise,
                             write_readme_results)
from src.paths import REPO_ROOT

HERE = Path(__file__).resolve().parent
SHOW_LAYERS = (8, 12, 16, 20, 22, 24, 26, 28, 30, 32)


def deco_margin(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """DeCo Yes-minus-No score; a candidate missing from C is -inf (Yes) or +inf (No)."""
    m = ((df["logit_yes_final"] - df["logit_no_final"])
         + alpha * df["p_max"] * (df["logit_yes_anchor"] - df["logit_no_anchor"])).to_numpy(float)
    yes_in, no_in = df["yes_in_cand"].to_numpy(bool), df["no_in_cand"].to_numpy(bool)
    m[yes_in & ~no_in] = np.inf
    m[~yes_in & no_in] = -np.inf
    m[~yes_in & ~no_in] = 0.0
    return m


def deco_unmasked(df: pd.DataFrame, alpha: float) -> np.ndarray:
    """DeCo's Yes-No score without the candidate-set restriction."""
    return ((df["logit_yes_final"] - df["logit_no_final"])
            + alpha * df["p_max"] * (df["logit_yes_anchor"] - df["logit_no_anchor"])).to_numpy(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()
    cells = load_method_cells("deco")
    if not cells:
        raise SystemExit("no complete DeCo cells yet")
    dev = dev_images(cells)
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)

    rows = []
    for (kind, level, split), df in cells.items():
        d = df[df["image"].isin(dev)].reset_index(drop=True)
        gold = (d["label"] == "yes").to_numpy()
        for a in ALPHAS:
            m_real = (d["logit_yes_final"] - d["logit_no_final"]).to_numpy(float)
            rows.append(dict(kind=kind, level=level, split=split, alpha=a,
                             auroc=auroc(decision_score(deco_margin(d, a), m_real), gold),
                             f1=f1_from((d[f"ans_deco_a{a}"] == "yes").to_numpy(), gold)))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "deco_alpha_dev.csv", index=False, float_format="%.5f")
    curve = sweep.groupby("alpha")[["auroc", "f1"]].mean()
    alpha_dev = float(curve["auroc"].idxmax())
    print(f"{len(cells)} cells, {len(dev)} DEV images\n{curve.round(4)}\nDEV-chosen alpha: {alpha_dev}")

    rng = np.random.default_rng(0)
    comp, layers = [], []
    for (kind, level, split), df in cells.items():
        t = df[~df["image"].isin(dev)].reset_index(drop=True)
        gold = (t["label"] == "yes").to_numpy()
        row = dict(kind=kind, level=level, split=split, **reference_row(t),
                   other_rate=(t[f"ans_deco_a{PAPER_ALPHA}"] == "other").mean(),
                   final_vs_stage2_max_abs_diff=float(np.abs((t["logit_yes_final"] - t["logit_no_final"])
                                                             - (t["logit_yes"] - t["logit_no"])).max()))
        for prefix, a in (("deco_", PAPER_ALPHA), ("decodev_", alpha_dev)):
            y_fix = (t[f"ans_deco_a{a}"] == "yes").to_numpy()
            m_fix = deco_margin(t, a)
            row.update(fix_metrics(t, m_fix, y_fix, args.boot, rng, prefix, m_unmasked=deco_unmasked(t, a)))
            row[f"{prefix}alpha"] = a
            row[f"{prefix}agree"] = ((m_fix > 0) == y_fix).mean()
        comp.append(row)
        ly, ln = np.array(t["layer_yes"].tolist()), np.array(t["layer_no"].tolist())
        layers.append(dict(kind=kind, level=level, split=split,
                           **{f"L{l}": auroc(ly[:, l - 1] - ln[:, l - 1], gold) for l in range(1, ly.shape[1] + 1)},
                           **{f"anchor_share_L{l}": (t["anchor_layer"] == l).mean() for l in range(20, 29)}))
    comp, layers = pd.DataFrame(comp), pd.DataFrame(layers)
    comp.to_csv(out / "deco_comparison.csv", index=False, float_format="%.5f")
    layers.to_csv(out / "deco_layer_auroc.csv", index=False, float_format="%.5f")

    lm = layers.groupby(["kind", "level"], sort=False).mean(numeric_only=True)
    lay_lines = ["| condition | " + " | ".join(f"L{l}" for l in SHOW_LAYERS) + " | most-chosen anchor |",
                 "|---|" + "---|" * (len(SHOW_LAYERS) + 1)]
    for (kind, level), r in lm.iterrows():
        cond = "clean" if kind == "clean" else f"{kind} s{level}"
        best = max(range(20, 29), key=lambda l: r[f"anchor_share_L{l}"])
        lay_lines.append(f"| {cond} | " + " | ".join(f"{r[f'L{l}']:.3f}" for l in SHOW_LAYERS)
                         + f" | L{best} ({100 * r[f'anchor_share_L{best}']:.0f}%) |")

    md = "\n\n".join([
        f"_Auto-generated by `python -m DeCo.analyze` from {len(comp)} condition × split cells "
        f"({int(comp['n'].iloc[0])} TEST questions per cell; 400 TEST images, 100 DEV images held out). "
        "Means over the three POPE splits. Greedy decoding throughout._",
        f"### DeCo as published (α = {PAPER_ALPHA}, layers 20–28, top-p 0.9, top-k 20)",
        summarise(comp, "deco_", "DeCo"),
        f"### DeCo with α chosen on DEV (α = {alpha_dev})",
        summarise(comp, "decodev_", "DeCo-dev"),
        "### Does the evidence survive in the middle layers? TEST AUROC of each layer's own Yes–No margin",
        "\n".join(lay_lines),
        "### Sanity checks",
        f"- Final layer vs Stage 2 baseline margin: max |diff| {comp['final_vs_stage2_max_abs_diff'].max():.3f} logits\n"
        f"- Non-Yes/No first tokens (α={PAPER_ALPHA}): max {comp['other_rate'].max():.4f} per cell\n"
        f"- Recomputed margin vs exact decision agreement: min {comp['deco_agree'].min():.4f}",
    ])
    write_readme_results(HERE / "README.md", md)
    print(md)


if __name__ == "__main__":
    main()
