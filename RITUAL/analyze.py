"""Score RITUAL against baseline, VCD and the matched-threshold baseline; write the README results.

    python -m RITUAL.analyze [--boot 500]

Variants (all greedy, first token, beta = 0.1):
    ritual       as published: original + alpha * (the image's one random view), alpha = 3 (paper)
    ritual_dev   same, alpha chosen on DEV images (max mean AUROC)
    ritual6      original + alpha * mean of all six views, alpha chosen on DEV
Ablation: each single transform as the view, alpha = 3, AUROC on TEST (which views help degraded images?).
Outputs: results/ritual_alpha_dev.csv, results/ritual_comparison.csv, results/ritual_transform_ablation.csv,
and the Results section of RITUAL/README.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fix_eval.common import (BETA, auroc, decision_score, dev_images, f1_from, fix_metrics, load_method_cells, reference_row,
                             summarise, write_readme_results)
from RITUAL.run_ritual import ALPHAS, PAPER_ALPHA
from RITUAL.transforms import TRANSFORMS
from src.paths import REPO_ROOT

HERE = Path(__file__).resolve().parent


def view_probs(df: pd.DataFrame, view: str) -> tuple[np.ndarray, np.ndarray]:
    """Yes/No probabilities of the view: 'chosen' (per-row random transform), 'mean6', or a transform name."""
    if view == "chosen":
        y = np.array([df.at[i, f"p_yes_{t}"] for i, t in enumerate(df["transform"])])
        n = np.array([df.at[i, f"p_no_{t}"] for i, t in enumerate(df["transform"])])
        return y, n
    if view == "mean6":
        return (df[[f"p_yes_{t}" for t in TRANSFORMS]].mean(1).to_numpy(),
                df[[f"p_no_{t}" for t in TRANSFORMS]].mean(1).to_numpy())
    return df[f"p_yes_{view}"].to_numpy(), df[f"p_no_{view}"].to_numpy()


def ritual_unmasked(df: pd.DataFrame, view: str, alpha: float) -> np.ndarray:
    """log P(Yes) - log P(No) with P = p(.|v) + alpha p(.|T(v)), no plausibility mask."""
    vy, vn = view_probs(df, view)
    py, pn = df["p_yes_real"].to_numpy(), df["p_no_real"].to_numpy()
    with np.errstate(divide="ignore"):
        return np.log(py + alpha * vy) - np.log(pn + alpha * vn)


def ritual_margin(df: pd.DataFrame, view: str, alpha: float, beta: float = BETA) -> np.ndarray:
    """Same, with the plausibility mask on p(.|v): forced answers become +-inf."""
    py, pn = df["p_yes_real"].to_numpy(), df["p_no_real"].to_numpy()
    m = ritual_unmasked(df, view, alpha)
    cut = beta * df["maxprob_real"].to_numpy()
    m[pn < cut] = np.inf
    m[py < cut] = -np.inf
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()
    cells = load_method_cells("ritual")
    if not cells:
        raise SystemExit("no complete RITUAL cells yet")
    dev = dev_images(cells)
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)

    # DEV: alpha sweep
    rows = []
    for (kind, level, split), df in cells.items():
        d = df[df["image"].isin(dev)].reset_index(drop=True)
        gold = (d["label"] == "yes").to_numpy()
        for variant, view in (("ritual", "chosen"), ("ritual6", "mean6")):
            for a in ALPHAS:
                m_real = (d["logit_yes"] - d["logit_no"]).to_numpy(float)
                rows.append(dict(kind=kind, level=level, split=split, variant=variant, alpha=a,
                                 auroc=auroc(decision_score(ritual_margin(d, view, a), m_real), gold),
                                 f1=f1_from((d[f"ans_{variant}_a{a}"] == "yes").to_numpy(), gold)))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "ritual_alpha_dev.csv", index=False, float_format="%.5f")
    curve = sweep.groupby(["variant", "alpha"])[["auroc", "f1"]].mean()
    chosen = {v: float(curve.loc[v, "auroc"].idxmax()) for v in ("ritual", "ritual6")}
    print(f"{len(cells)} cells, {len(dev)} DEV images\n{curve.round(4)}\nDEV-chosen alpha: {chosen}")

    # TEST
    rng = np.random.default_rng(0)
    comp, abl = [], []
    for (kind, level, split), df in cells.items():
        t = df[~df["image"].isin(dev)].reset_index(drop=True)
        gold = (t["label"] == "yes").to_numpy()
        row = dict(kind=kind, level=level, split=split, **reference_row(t),
                   other_rate=(t[f"ans_ritual_a{PAPER_ALPHA}"] == "other").mean())
        for prefix, variant, view, a in (("ritual_", "ritual", "chosen", PAPER_ALPHA),
                                         ("ritualdev_", "ritual", "chosen", chosen["ritual"]),
                                         ("ritual6_", "ritual6", "mean6", chosen["ritual6"])):
            y_fix = (t[f"ans_{variant}_a{a}"] == "yes").to_numpy()
            m_fix = ritual_margin(t, view, a)
            row.update(fix_metrics(t, m_fix, y_fix, args.boot, rng, prefix, m_unmasked=ritual_unmasked(t, view, a)))
            row[f"{prefix}alpha"] = a
            row[f"{prefix}agree"] = ((m_fix > 0) == y_fix).mean()  # recomputed margin vs exact decision
        comp.append(row)
        m_real = (t["logit_yes"] - t["logit_no"]).to_numpy(float)
        abl.append(dict(kind=kind, level=level, split=split, auroc_base=row["auroc_base"],
                        **{f"auroc_{tr}": auroc(decision_score(ritual_margin(t, tr, PAPER_ALPHA), m_real), gold)
                           for tr in TRANSFORMS},
                        auroc_view_alone_mean6=auroc(np.log(view_probs(t, "mean6")[0]) - np.log(view_probs(t, "mean6")[1]), gold)))
    comp, abl = pd.DataFrame(comp), pd.DataFrame(abl)
    comp.to_csv(out / "ritual_comparison.csv", index=False, float_format="%.5f")
    abl.to_csv(out / "ritual_transform_ablation.csv", index=False, float_format="%.5f")

    ab = abl.groupby(["kind", "level"], sort=False).mean(numeric_only=True)
    ab_lines = ["| condition | base | " + " | ".join(TRANSFORMS) + " |", "|---|---|" + "---|" * len(TRANSFORMS)]
    for (kind, level), r in ab.iterrows():
        cond = "clean" if kind == "clean" else f"{kind} s{level}"
        ab_lines.append(f"| {cond} | {r['auroc_base']:.3f} | " + " | ".join(f"{r['auroc_' + tr]:.3f}" for tr in TRANSFORMS) + " |")

    md = "\n\n".join([
        f"_Auto-generated by `python -m RITUAL.analyze` from {len(comp)} condition × split cells "
        f"({int(comp['n'].iloc[0])} TEST questions per cell; 400 TEST images, 100 DEV images held out). "
        "Means over the three POPE splits. Greedy decoding throughout._",
        f"### RITUAL as published (α = {PAPER_ALPHA}, one random view per image)",
        summarise(comp, "ritual_", "RITUAL"),
        f"### RITUAL with α chosen on DEV (α = {chosen['ritual']})",
        summarise(comp, "ritualdev_", "RITUAL-dev"),
        f"### All six views averaged (α chosen on DEV = {chosen['ritual6']})",
        summarise(comp, "ritual6_", "RITUAL-6"),
        f"### Which transform helps? TEST AUROC with each single view (α = {PAPER_ALPHA})",
        "\n".join(ab_lines),
        "### Sanity checks",
        f"- Non-Yes/No first tokens (RITUAL α={PAPER_ALPHA}): max {comp['other_rate'].max():.4f} of questions per cell\n"
        f"- Recomputed margin vs exact full-vocab decision agreement: min {comp['ritual_agree'].min():.4f}",
    ])
    write_readme_results(HERE / "README.md", md)
    print(md)


if __name__ == "__main__":
    main()
