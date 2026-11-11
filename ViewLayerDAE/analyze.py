"""ViewLayer+DAE: ViewLayer with the DAE-restored image as an extra view, plus an optional VCD contrast term.
Select on DEV images, score on TEST; writes ViewLayerDAE/README.md.

    python -m ViewLayerDAE.analyze [--boot 500]

    score(q) = (1 + a) * mean_{v in V} m_L(v)  -  a * m_L(v_cd)          answer Yes iff score > t

m_L(v) is the Yes-No margin of view v read out at layer L; v_cd is VCD's diffusion-noised image
(T=999). V may contain the degraded image, blurred / flipped copies of it (ViewLayer) and its
DAE-restored copy (Fix B). (V, L, a) maximise the mean DEV AUROC over all 27 condition x split
cells: one global choice, so the method never needs to know how degraded an image is. a = 0 is in
the grid, so the selection can fall back to ViewLayer. t is one global threshold maximising mean
DEV F1. Everything on TEST uses those fixed choices.

Reads the ViewLayer records (orig, blur1-3, hflip) and the ViewLayer+DAE records (dae, cd).

Reported on TEST (fix_eval scoring, 95% paired-bootstrap CIs):
    vl_dae          free choice of V, L and a
    no_contrast      a = 0, V may include the DAE view        (does the contrast term earn its place?)
    viewlayer        a = 0, no DAE view                        (reproduces ViewLayer/README.md)
    forced_contrast  a > 0, free V and L                       (what the contrast does when it must be used)
    vcd_layer        V = {orig}, a > 0                         (plain VCD without the mask, read at layer L)
    dae_only         V = {dae}, a = 0, best L                  (the DAE view alone, read at a middle layer)
    dae_final        V = {dae}, a = 0, L = 32                  (Fix B's B0: the baseline on the restored image)
    + the baseline with its own DEV-tuned threshold.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fix_eval.common import OUTPUT_ROOT, auroc, dev_images, f1_from, fix_metrics, load_method_cells, read_jsonl, \
    reference_row, summarise, write_readme_results
from src.paths import REPO_ROOT
from ViewLayer.analyze import VIEW_SETS as VIEWLAYER_SETS
from ViewLayer.run_viewlayer import VIEWS
from ViewLayerDAE.run_viewlayer_dae import EXTRA_VIEWS

HERE = Path(__file__).resolve().parent
VIEW_SETS = list(VIEWLAYER_SETS) + [("dae",)] + [vs + ("dae",) for vs in VIEWLAYER_SETS]
LAYERS = range(12, 33)
ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0)
FINAL = 32
LABELS = {"vl_dae": "ViewLayer+DAE", "no_contrast": "no-contrast", "viewlayer": "ViewLayer",
          "forced_contrast": "forced-contrast", "vcd_layer": "VCD@layer", "dae_only": "DAE-only",
          "dae_final": "DAE@L32"}


def score(M: dict, views, layer: int, alpha: float) -> np.ndarray:
    """M maps a view name to its (questions, 32) array of layer margins."""
    pos = np.mean([M[v][:, layer - 1] for v in views], axis=0)
    return pos if alpha == 0 else (1 + alpha) * pos - alpha * M["cd"][:, layer - 1]


def best_threshold(scores: list[np.ndarray], golds: list[np.ndarray]) -> float:
    """One global threshold maximising mean F1 across cells."""
    grid = np.quantile(np.concatenate(scores), np.linspace(0.02, 0.98, 193))
    f1s = [np.mean([f1_from(s > t, g) for s, g in zip(scores, golds)]) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def dev_sweep(dev: dict) -> pd.DataFrame:
    """Mean DEV AUROC of every (V, L, a). `dev` maps a cell key to (M, gold)."""
    rows = []
    for views in VIEW_SETS:
        for layer in LAYERS:
            for alpha in ALPHAS:
                aucs = [auroc(score(M, views, layer, alpha), gold) for M, gold in dev.values()]
                rows.append(dict(views="+".join(views), layer=layer, alpha=alpha, dev_auroc=np.mean(aucs)))
    return pd.DataFrame(rows)


def select(sweep: pd.DataFrame) -> dict:
    """{variant: (views, layer, alpha, dev_auroc)}: the best DEV row within each variant's search space."""
    has_dae = sweep["views"].str.contains("dae")

    def pick(sub):
        r = sub.loc[sub["dev_auroc"].idxmax()]
        return tuple(r["views"].split("+")), int(r["layer"]), float(r["alpha"]), float(r["dev_auroc"])

    return {
        "vl_dae": pick(sweep),
        "no_contrast": pick(sweep[sweep["alpha"] == 0]),
        "viewlayer": pick(sweep[(sweep["alpha"] == 0) & ~has_dae]),
        "forced_contrast": pick(sweep[sweep["alpha"] > 0]),
        "vcd_layer": pick(sweep[(sweep["alpha"] > 0) & (sweep["views"] == "orig")]),
        "dae_only": pick(sweep[(sweep["alpha"] == 0) & (sweep["views"] == "dae")]),
        "dae_final": pick(sweep[(sweep["alpha"] == 0) & (sweep["views"] == "dae") & (sweep["layer"] == FINAL)]),
        "baseline_devthr": pick(sweep[(sweep["alpha"] == 0) & (sweep["views"] == "orig") & (sweep["layer"] == FINAL)]),
    }


def load_cells() -> dict:
    """{(kind, level, split): ViewLayer records + Stage 2 references + the dae / cd margins}."""
    cells = {}
    for (kind, level, split), df in load_method_cells("viewlayer").items():
        path = OUTPUT_ROOT / f"viewlayer_dae_{kind}_s{level}" / f"{split}.jsonl"
        if not path.exists():
            print(f"skip {kind} s{level} {split}: no ViewLayer+DAE records")
            continue
        extra = read_jsonl(path)[["question_id"] + [f"m_{v}" for v in EXTRA_VIEWS]]
        merged = df.merge(extra, on="question_id")
        if not (len(extra) == len(df) == len(merged)):
            print(f"skip {kind} s{level} {split}: incomplete ({len(extra)} ViewLayer+DAE / {len(df)} ViewLayer)")
            continue
        cells[(kind, level, split)] = merged.sort_values("question_id").reset_index(drop=True)
    return cells


def margins(df: pd.DataFrame) -> dict:
    return {v: np.array(df[f"m_{v}"].tolist(), float) for v in VIEWS + EXTRA_VIEWS}


def name(views, layer, alpha) -> str:
    return f"{'+'.join(views)} @ L{layer}, α = {alpha:g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()
    cells = load_cells()
    if not cells:
        raise SystemExit("no complete ViewLayer+DAE cells yet")
    dev_imgs = dev_images(cells)
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)

    # ---------------- DEV selection
    dev = {}
    for key, df in cells.items():
        d = df[df["image"].isin(dev_imgs)]
        dev[key] = (margins(d), (d["label"] == "yes").to_numpy())
    sweep = dev_sweep(dev)
    sweep.to_csv(out / "viewlayer_dae_dev_sweep.csv", index=False, float_format="%.5f")
    variants = select(sweep)
    thresholds = {k: best_threshold([score(M, v, l, a) for M, _ in dev.values()], [g for _, g in dev.values()])
                  for k, (v, l, a, _) in variants.items()}
    for k, (v, l, a, auc) in variants.items():
        print(f"{k:16s} {name(v, l, a):48s} DEV AUROC {auc:.4f}  threshold {thresholds[k]:+.3f}")

    # ---------------- TEST
    rng = np.random.default_rng(0)
    comp = []
    for (kind, level, split), df in cells.items():
        t = df[~df["image"].isin(dev_imgs)].reset_index(drop=True)
        M, gold = margins(t), (t["label"] == "yes").to_numpy()
        s = {k: score(M, v, l, a) for k, (v, l, a, _) in variants.items()}
        y = {k: s[k] > thresholds[k] for k in variants}
        row = dict(kind=kind, level=level, split=split, **reference_row(t),
                   f1_base_devthr=f1_from(y["baseline_devthr"], gold), yes_base_devthr=y["baseline_devthr"].mean(),
                   # sanity: the cd view at layer 32 must reproduce the Stage 2 contrast-branch margin
                   cd_final_vs_stage2_max_abs_diff=float(np.abs(M["cd"][:, FINAL - 1]
                                                                - (t["logit_yes_cd"] - t["logit_no_cd"])).max()))
        for k in LABELS:
            row.update(fix_metrics(t, s[k], y[k], args.boot, rng, f"{k}_", m_unmasked=s[k]))
            row[f"{k}_f1_vs_base_devthr"] = f1_from(y[k], gold) - row["f1_base_devthr"]
        # paired bootstrap: is ViewLayer+DAE better than its own fallbacks?
        idx = rng.integers(0, len(t), size=(args.boot, len(t)))
        for ref in ("viewlayer", "no_contrast", "dae_only", "dae_final"):
            d = np.array([auroc(s["vl_dae"][i], gold[i]) - auroc(s[ref][i], gold[i]) for i in idx])
            row[f"vl_dae_d_auroc_{ref}_lo"], row[f"vl_dae_d_auroc_{ref}_hi"] = np.percentile(d, [2.5, 97.5])
        comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "viewlayer_dae_comparison.csv", index=False, float_format="%.5f")

    # ---------------- README
    by = comp.groupby(["kind", "level"], sort=False).mean(numeric_only=True)
    cond = lambda kind, level: "clean" if kind == "clean" else f"{kind} s{level}"
    head = ["| condition | AUROC baseline | AUROC VCD | AUROC DAE@L32 | AUROC ViewLayer | AUROC ViewLayer+DAE "
            "| F1 baseline (DEV thr) | F1 VCD | F1 ViewLayer | F1 ViewLayer+DAE | yes% ViewLayer+DAE |", "|---|" + "---|" * 10]
    for (kind, level), r in by.iterrows():
        head.append(f"| {cond(kind, level)} | {r['auroc_base']:.3f} | {r['auroc_vcd']:.3f} | {r['dae_final_auroc']:.3f} | {r['viewlayer_auroc']:.3f} "
                    f"| **{r['vl_dae_auroc']:.3f}** | {100 * r['f1_base_devthr']:.1f} "
                    f"| {100 * r['f1_vcd']:.1f} | {100 * r['viewlayer_f1']:.1f} | **{100 * r['vl_dae_f1']:.1f}** "
                    f"| {100 * r['vl_dae_yes']:.1f} |")
    gain = lambda k: (comp[f"{k}_auroc"] - comp["auroc_base"]).mean()
    wins = lambda ref: (int((comp[f"vl_dae_d_auroc_{ref}_lo"] > 0).sum()), int((comp[f"vl_dae_d_auroc_{ref}_hi"] < 0).sum()))
    sel = "\n".join(f"- **{k}**: {name(v, l, a)} (DEV AUROC {auc:.4f}, threshold {thresholds[k]:+.2f})"
                    for k, (v, l, a, auc) in variants.items())
    alpha = variants["vl_dae"][2]
    verdict = ("DEV selected α = 0: the contrast term is not used, and ViewLayer+DAE equals the no-contrast variant."
               if alpha == 0 else f"DEV selected α = {alpha:g}: the contrast term is used.")
    md = "\n\n".join([
        f"_Auto-generated by `python -m ViewLayerDAE.analyze` from {len(comp)} condition × split cells "
        f"({int(comp['n'].iloc[0])} TEST questions per cell; 400 TEST images, 100 DEV images held out). "
        "Means over the three POPE splits._",
        "### What DEV selected (one global choice for all conditions)", sel, verdict,
        "### Head to head (TEST)", "\n".join(head),
        f"Mean TEST AUROC gain over the baseline: ViewLayer+DAE {gain('vl_dae'):+.4f}, no-contrast {gain('no_contrast'):+.4f}, "
        f"ViewLayer {gain('viewlayer'):+.4f}, forced-contrast {gain('forced_contrast'):+.4f}, VCD@layer {gain('vcd_layer'):+.4f}, "
        f"DAE-only {gain('dae_only'):+.4f}, DAE@L32 {gain('dae_final'):+.4f}.",
        f"Out of {len(comp)} cells (95% paired-bootstrap CIs), ViewLayer+DAE's AUROC is above ViewLayer's in "
        f"**{wins('viewlayer')[0]}** and below in {wins('viewlayer')[1]}; above the no-contrast variant's in "
        f"**{wins('no_contrast')[0]}** and below in {wins('no_contrast')[1]}; above the DAE view alone at layer 32 "
        f"(Fix B's B0) in **{wins('dae_final')[0]}** and below in {wins('dae_final')[1]}; above the DAE view alone at its "
        f"best layer in **{wins('dae_only')[0]}** and below in {wins('dae_only')[1]}.",
        f"Mean TEST F1 difference ViewLayer+DAE − baseline (DEV thr): {100 * comp['vl_dae_f1_vs_base_devthr'].mean():+.2f} points.",
        *[part for k, label in LABELS.items()
          for part in (f"### {label}: {name(*variants[k][:3])}", summarise(comp, f"{k}_", label))],
        "### Sanity checks",
        f"- Contrast view at layer 32 vs Stage 2 contrast-branch margin: max |diff| "
        f"{comp['cd_final_vs_stage2_max_abs_diff'].max():.3f} logits (batching in fp16)\n"
        f"- The `viewlayer` variant should match `ViewLayer/README.md` (same records, same selection).",
    ])
    write_readme_results(HERE / "README.md", md)
    print(md)


if __name__ == "__main__":
    main()
