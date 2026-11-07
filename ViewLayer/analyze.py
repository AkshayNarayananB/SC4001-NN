"""Select views + layer + threshold on DEV images, then score on TEST; writes ViewLayer/README.md results.

    python -m ViewLayer.analyze [--boot 500]

Method: score(q) = mean over a view set V of the Yes-No margin read out at layer L; answer Yes iff
score > t. (V, L) maximise the mean DEV AUROC over all 27 condition x split cells: one global
choice, so the method never needs to know how degraded an image is. t is one global threshold
maximising mean DEV F1. Everything on TEST uses those fixed choices.

Reported on TEST (fix_eval scoring: AUROC vs baseline and VCD, F1 vs the matched-threshold baseline,
95% paired-bootstrap CIs):
    viewlayer     chosen V and L
    layer_only    V = {orig}, best L on DEV           (DeCo's signal alone)
    views_only    best V on DEV, final layer L = 32    (RITUAL's signal alone)
    + the baseline with its own DEV-tuned threshold, so the threshold tuning is not an unfair edge.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fix_eval.common import auroc, dev_images, f1_from, fix_metrics, load_method_cells, reference_row, summarise, \
    write_readme_results
from src.paths import REPO_ROOT
from ViewLayer.run_viewlayer import VIEWS

HERE = Path(__file__).resolve().parent
VIEW_SETS = [("orig",), ("orig", "blur1"), ("orig", "blur2"), ("orig", "blur3"), ("orig", "hflip"),
             ("orig", "blur1", "hflip"), ("orig", "blur2", "hflip"), ("orig", "blur1", "blur2"),
             VIEWS]
LAYERS = range(12, 33)
FINAL = 32


def score(df: pd.DataFrame, views, layer: int) -> np.ndarray:
    return np.mean([np.array(df[f"m_{v}"].tolist())[:, layer - 1] for v in views], axis=0)


def best_threshold(dev_parts: list[pd.DataFrame], views, layer: int) -> float:
    """One global threshold maximising mean DEV F1 across cells."""
    scores = [score(d, views, layer) for d in dev_parts]
    golds = [(d["label"] == "yes").to_numpy() for d in dev_parts]
    grid = np.quantile(np.concatenate(scores), np.linspace(0.02, 0.98, 193))
    f1s = [np.mean([f1_from(s > t, g) for s, g in zip(scores, golds)]) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def name(views, layer) -> str:
    return f"{'+'.join(views)} @ L{layer}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()
    cells = load_method_cells("viewlayer")
    if not cells:
        raise SystemExit("no complete ViewLayer cells yet")
    dev = dev_images(cells)
    dev_parts = {k: df[df["image"].isin(dev)].reset_index(drop=True) for k, df in cells.items()}
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)

    # ---------------- DEV selection
    rows = []
    for views in VIEW_SETS:
        for layer in LAYERS:
            aucs = [auroc(score(d, views, layer), (d["label"] == "yes").to_numpy()) for d in dev_parts.values()]
            rows.append(dict(views="+".join(views), layer=layer, dev_auroc=np.mean(aucs)))
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "viewlayer_dev_sweep.csv", index=False, float_format="%.5f")

    def pick(sub):
        r = sub.loc[sub["dev_auroc"].idxmax()]
        return tuple(r["views"].split("+")), int(r["layer"]), float(r["dev_auroc"])

    variants = {
        "viewlayer": pick(sweep),
        "layer_only": pick(sweep[sweep["views"] == "orig"]),
        "views_only": pick(sweep[sweep["layer"] == FINAL]),
        "baseline_devthr": (("orig",), FINAL, float(sweep[(sweep["views"] == "orig") & (sweep["layer"] == FINAL)]["dev_auroc"].iloc[0])),
    }
    thresholds = {k: best_threshold(list(dev_parts.values()), v, l) for k, (v, l, _) in variants.items()}
    for k, (v, l, a) in variants.items():
        print(f"{k:16s} {name(v, l):32s} DEV AUROC {a:.4f}  threshold {thresholds[k]:+.3f}")

    # ---------------- TEST
    rng = np.random.default_rng(0)
    comp = []
    for (kind, level, split), df in cells.items():
        t = df[~df["image"].isin(dev)].reset_index(drop=True)
        gold = (t["label"] == "yes").to_numpy()
        row = dict(kind=kind, level=level, split=split, **reference_row(t),
                   # sanity: orig view at layer 32 must reproduce the Stage 2 baseline margin
                   orig_final_vs_stage2_max_abs_diff=float(np.abs(score(t, ("orig",), FINAL)
                                                                  - (t["logit_yes"] - t["logit_no"])).max()))
        for k, (views, layer, _) in variants.items():
            s = score(t, views, layer)
            y = s > thresholds[k]
            if k == "baseline_devthr":
                row.update(f1_base_devthr=f1_from(y, gold), yes_base_devthr=y.mean())
                continue
            row.update(fix_metrics(t, s, y, args.boot, rng, f"{k}_", m_unmasked=s))
            row[f"{k}_f1_vs_base_devthr"] = f1_from(y, gold) - f1_from(score(t, ("orig",), FINAL) > thresholds["baseline_devthr"], gold)
        comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "viewlayer_comparison.csv", index=False, float_format="%.5f")

    # ---------------- README
    by = comp.groupby(["kind", "level"], sort=False).mean(numeric_only=True)
    dt = ["| condition | F1 baseline (thr 0) | F1 baseline (DEV thr) | F1 ViewLayer | F1 layer-only | F1 views-only "
          "| yes% baseline (DEV thr) | yes% ViewLayer |", "|---|" + "---|" * 7]
    for (kind, level), r in by.iterrows():
        cond = "clean" if kind == "clean" else f"{kind} s{level}"
        dt.append(f"| {cond} | {100 * r['f1_base']:.1f} | {100 * r['f1_base_devthr']:.1f} | **{100 * r['viewlayer_f1']:.1f}** "
                  f"| {100 * r['layer_only_f1']:.1f} | {100 * r['views_only_f1']:.1f} "
                  f"| {100 * r['yes_base_devthr']:.1f} | {100 * r['viewlayer_yes']:.1f} |")
    gain = lambda k: (comp[f"{k}_auroc"] - comp["auroc_base"]).mean()
    sel = "\n".join(f"- **{k}**: {name(v, l)} (DEV AUROC {a:.4f}, threshold {thresholds[k]:+.2f})"
                    for k, (v, l, a) in variants.items())
    md = "\n\n".join([
        f"_Auto-generated by `python -m ViewLayer.analyze` from {len(comp)} condition × split cells "
        f"({int(comp['n'].iloc[0])} TEST questions per cell; 400 TEST images, 100 DEV images held out). "
        "Means over the three POPE splits._",
        "### What DEV selected (one global choice for all conditions)", sel,
        "### ViewLayer (selected views + layer, DEV threshold)", summarise(comp, "viewlayer_", "ViewLayer"),
        "### Ablation: which part does the work?",
        f"Mean TEST AUROC gain over the baseline: ViewLayer {gain('viewlayer'):+.4f}, "
        f"layer-only {gain('layer_only'):+.4f}, views-only {gain('views_only'):+.4f}.",
        "#### Layer only (original image, DEV-chosen layer)", summarise(comp, "layer_only_", "layer-only"),
        "#### Views only (final layer, DEV-chosen views)", summarise(comp, "views_only_", "views-only"),
        "### F1 with a fair threshold for the baseline too",
        "ViewLayer's threshold is tuned on DEV, so the baseline also gets its own DEV-tuned global threshold here.",
        "\n".join(dt),
        f"Mean TEST F1 difference ViewLayer − baseline (DEV thr): {100 * comp['viewlayer_f1_vs_base_devthr'].mean():+.2f} points.",
        "### Sanity checks",
        f"- Original view at layer 32 vs Stage 2 baseline margin: max |diff| "
        f"{comp['orig_final_vs_stage2_max_abs_diff'].max():.3f} logits (batching in fp16)",
    ])
    write_readme_results(HERE / "README.md", md)
    print(md)


if __name__ == "__main__":
    main()
