"""Score the patch-level grounding readout (scripts/run_patches.py) under the Table-4 protocol.

    python -m scripts.analyze_patches [--boot 500]

Scores for a question "Is there a <obj>?" on an image, all label-free and from the same single forward pass
that answers the question (the image tokens precede the question):
  patch         s = statistic over the 576 patches of log p_layer(<obj> token)   (max / top-5 mean / mean)
  patch-z       s standardised across the 79 POPE objects for the same image: removes the image-level shift
                that degradation causes ("everything looks less likely"), keeps which objects stand out
  fused         z(baseline margin) + w * z(patch-z score), z = standardisation with DEV statistics
DEV images (100) choose layer, token mode, statistic, normalisation and w (max mean DEV AUROC) and the
F1 threshold; everything is reported on the 400 TEST images, as in scripts/analyze_layers.py.
Writes results/patches_dev_sweep.csv, results/patches_cells.csv, results/patches_table.{csv,md}.
"""
import argparse
from itertools import product

import numpy as np
import pandas as pd

from scripts.analyze_layers import (FINAL, best_threshold, dev_images, evaluate, layer_margin, load_cells,
                                    markdown, mean_dev_auroc, summary_table, to_arrays)
from scripts.run_layers import CONDITIONS, cond_name
from scripts.run_pope import OUTPUT_ROOT
from src.paths import REPO_ROOT

STATS, TOKMODES = ("max", "top5", "mean"), ("first", "all")


def load_patches():
    out = {}
    for kind, level in CONDITIONS:
        path = OUTPUT_ROOT / f"patches_{cond_name(kind, level)}.npz"
        if not path.exists():
            print(f"skip {kind} s{level}: no patch readout yet")
            continue
        z = np.load(path)
        s = z["stats"].astype(np.float32)                              # (n_img, L, k, 2, 3)
        zs = (s - s.mean(2, keepdims=True)) / (s.std(2, keepdims=True) + 1e-6)
        out[(kind, level)] = dict(img={im: i for i, im in enumerate(z["images"].tolist())},
                                  obj={o: j for j, o in enumerate(z["objects"].tolist())},
                                  layers=list(z["layers"]), raw=s, z=zs)
    return out


def attach(arrays, P, kind, level):
    p = P[(kind, level)]
    arrays["pi"] = np.array([p["img"][i] for i in arrays["image"]])
    arrays["pj"] = np.array([p["obj"][o] for o in arrays["object"]])
    arrays["P"] = p
    return arrays


def patch_score(c, layer, tok, stat, norm):
    p = c["P"]
    arr = p["z"] if norm == "z" else p["raw"]
    return arr[c["pi"], p["layers"].index(layer), c["pj"], TOKMODES.index(tok), STATS.index(stat)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()
    P = load_patches()
    cells = {k: df for k, df in load_cells().items() if k[:2] in P}
    if not cells:
        raise SystemExit("no patch readouts yet: run python -m scripts.run_patches")
    dev_set = dev_images(load_cells())
    dev = {k: attach(to_arrays(df[df["image"].isin(dev_set)]), P, *k[:2]) for k, df in cells.items()}
    test = {k: attach(to_arrays(df[~df["image"].isin(dev_set)]), P, *k[:2]) for k, df in cells.items()}
    out = REPO_ROOT / "results"

    layers = P[next(iter(P))]["layers"]
    sweep = []
    for layer, tok, stat, norm in product(layers, TOKMODES, STATS, ("raw", "z")):
        sweep.append(dict(layer=layer, tok=tok, stat=stat, norm=norm,
                          dev_auroc=mean_dev_auroc(dev, lambda _, c: patch_score(c, layer, tok, stat, norm))))
    sweep = pd.DataFrame(sweep).sort_values("dev_auroc", ascending=False)
    sweep.to_csv(out / "patches_dev_sweep.csv", index=False, float_format="%.5f")
    b = sweep.iloc[0]
    cfg = (int(b.layer), b.tok, b.stat, b.norm)
    patch = lambda _, c: patch_score(c, *cfg)
    print(f"DEV best patch score: layer {cfg[0]}, {cfg[1]} token, {cfg[2]}, {cfg[3]} (DEV AUROC {b.dev_auroc:.4f})")
    print(sweep.head(8).to_string(index=False))

    # fusion with the baseline margin; both standardised with pooled DEV statistics
    base = lambda _, c: layer_margin(c, FINAL)
    pool = lambda fn: np.concatenate([fn(k, c) for k, c in dev.items()])
    mb, sb = pool(base).mean(), pool(base).std()
    mp, sp = pool(patch).mean(), pool(patch).std()
    fused_w = lambda w: (lambda k, c: (base(k, c) - mb) / sb + w * (patch(k, c) - mp) / sp)
    ws = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    dev_w = {w: mean_dev_auroc(dev, fused_w(w)) for w in ws}
    w = max(dev_w, key=dev_w.get)
    print("fusion DEV AUROC by w:", {k: round(v, 4) for k, v in dev_w.items()}, "-> w =", w)

    entries = [("Baseline (threshold 0)", base, 0.0, 1),
               ("Baseline (DEV threshold)*", base, "dev", 1),
               (f"Patch grounding (L{cfg[0]}, {cfg[2]}, {cfg[3]})*", patch, "dev", 1),
               (f"Patch grounding + baseline (w={w:g})*", fused_w(w), "dev", 1)]
    entries = [(lab, fn, best_threshold(dev, fn) if thr == "dev" else thr, n) for lab, fn, thr, n in entries]
    comp = evaluate(test, entries, args.boot, np.random.default_rng(0))
    comp.to_csv(out / "patches_cells.csv", index=False, float_format="%.5f")
    table = summary_table(comp, entries)
    table.to_csv(out / "patches_table.csv", index=False, float_format="%.4f")
    notes = [f"Patch score chosen on DEV: layer {cfg[0]}, {cfg[1]} sub-token, {cfg[2]} over patches, {cfg[3]} "
             f"(DEV AUROC {b.dev_auroc:.4f}); fusion weight w={w:g} chosen on DEV. One forward pass: the patch states "
             "come from the same pass that answers the question."]
    md = markdown(table, comp[["kind", "level", "split"]].drop_duplicates().shape[0], int(comp.n.iloc[0]), args.boot, notes, script="scripts.analyze_patches")
    (out / "patches_table.md").write_text(md + "\n")
    print(md)
    pd.set_option("display.width", 200)
    print(comp.groupby(["method", "kind", "level"], sort=False)[["auroc", "auroc_base", "f1", "yes"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
