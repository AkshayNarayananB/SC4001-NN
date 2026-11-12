"""Probe x views x DAE: the linear probes of scripts/analyze_layers.py read out every ViewLayer view
and the DAE-restored view, and the probe scores are averaged over the views.

    python -m scripts.analyze_probe_views_dae [--boot 500] [--out results]

It combines three earlier results:
  ViewLayer       mean Yes-No margin over test-time views (orig, blur sigma 1/2/3, h-flip), layer 20
  ViewLayer+DAE   the same with the dae_v1-restored image as a sixth view (DEV picked alpha = 0)
  linear probes   logistic regression on the last-token hidden state: the gold probe (POPE labels) and
                  the clean-teacher probe (the model's own clean-image answer; no human labels)
The probe replaces ViewLayer's fixed layer-20 readout:  score(q) = mean_{v in V} probe(h_v(q)).
On the probe-results diagnosis, the probe fixes blur (evidence present but mis-read) and the views,
the restored one above all, fix noise (evidence lost at the input).

Protocol (as scripts/analyze_layers.py and ViewLayerDAE/analyze.py, so rows compare with Table 4):
  - 100 DEV / 400 TEST images (analyze_layers.dev_images); every choice is one global DEV choice.
  - Probes are fit on the ORIGINAL view of the DEV images, all nine conditions. Layers and C are the
    DEV choices already made in analyze_layers (gold L24 C=1e-3, teacher L16+24 C=1e-4), not re-tuned.
  - The view set V (any non-empty subset of the six views) is chosen by mean DEV AUROC over all cells,
    using out-of-fold probe scores (5 image-grouped folds: a DEV image is always scored by a probe
    that never saw it), so the in-sample original view is not favoured.
  - One global DEV F1 threshold per method, from the same out-of-fold scores.
  - TEST: AUROC and F1 with paired-bootstrap significance against the baseline (AUROC) and the
    matched-threshold baseline (F1), and a paired bootstrap against ViewLayer+DAE.
  ViewLayer and ViewLayer+DAE are recomputed from the same logs as reference rows (published configs).

Inputs: $VCD_OUTPUT_ROOT/layers_<cond>[__<view>]/ written by
    python -m scripts.run_layers --views orig blur1 blur2 blur3 hflip dae
Conditions missing a view are skipped with a warning (the table is then partial).
Outputs: <out>/probe_views_dae_{dev_sweep.csv,cells.csv,table.md}
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.analyze_layers as al
from scripts.run_layers import run_dir_name
from src.paths import REPO_ROOT

VIEWS = ("orig", "blur1", "blur2", "blur3", "hflip", "dae")
VIEWLAYER = ("orig", "blur1", "blur2", "blur3", "hflip")
VIEWLAYER_DAE = VIEWLAYER + ("dae",)
READ_LAYER = 20          # ViewLayer / ViewLayer+DAE readout layer (their DEV choice)
PROBES = {               # name: (hidden layers, C, target); the DEV choices of scripts.analyze_layers
    "Gold probe L24": ((24,), 1e-3, "gold"),
    "Clean-teacher probe L16+24 (no labels)": ((16, 24), 1e-4, "teacher"),
}
SHORT = {"Gold probe L24": "Gold probe", "Clean-teacher probe L16+24 (no labels)": "Clean-teacher probe"}
N_FOLDS = 5
CHUNK = 2048


# ================================================================ data
def view_dir(kind, level, view):
    return al.OUTPUT_ROOT / run_dir_name(kind, level, view)


def hidden(kind, level, view):
    path = view_dir(kind, level, view) / "_hidden.f16"
    return np.memmap(path, dtype=np.float16, mode="r").reshape(-1, len(al.HIDDEN_LAYERS), al.HIDDEN_DIM)


def features(H, rows, layers):
    cols = [al.HIDDEN_LAYERS.index(l) for l in layers]
    return np.asarray(H[rows][:, cols], dtype=np.float32).reshape(len(rows), -1)


def load(cells) -> dict:
    """{(kind, level, split): arrays} for every cell whose condition has all six views. Per view v:
    hidx_<v> (row of that view's _hidden.f16; each view's cache has its own row order) and
    m_<v> (n, 32) per-layer Yes-No margins. Also gold, image, Y/N (baseline) and teacher."""
    data, skipped = {}, set()
    for (kind, level, split), df in cells.items():
        if not all((view_dir(kind, level, v) / f"{split}.jsonl").exists() for v in VIEWS[1:]):
            skipped.add(al.cond_name(kind, level))
            continue
        clean = cells[("clean", 0, split)]
        assert (clean["question_id"].to_numpy() == df["question_id"].to_numpy()).all()
        a = al.to_arrays(df)
        c = {"gold": a["gold"], "image": a["image"], "Y": a["Y"], "N": a["N"],
             "teacher": al.layer_margin(al.to_arrays(clean), al.FINAL),
             "hidx_orig": a["hidx"], "m_orig": a["Y"] - a["N"]}
        for v in VIEWS[1:]:
            with open(view_dir(kind, level, v) / f"{split}.jsonl") as f:
                vdf = pd.DataFrame(json.loads(l) for l in f).sort_values("question_id").reset_index(drop=True)
            if not np.array_equal(vdf["question_id"].to_numpy(), df["question_id"].to_numpy()):
                raise SystemExit(f"{run_dir_name(kind, level, v)}/{split}: questions differ from the original view")
            c[f"hidx_{v}"] = vdf["hidx"].to_numpy()
            c[f"m_{v}"] = np.array(vdf["layer_yes"].tolist()) - np.array(vdf["layer_no"].tolist())
        data[(kind, level, split)] = c
    if skipped:
        print(f"warning: skipping {sorted(skipped)}: not every view has been run (python -m scripts.run_layers "
              f"--views {' '.join(VIEWS[1:])})")
    return data


def subset(c, mask):
    return {k: v[mask] for k, v in c.items()}


def fold_of(dev_set) -> dict:
    """Image -> fold, deterministic and grouped by image."""
    return {img: i % N_FOLDS for i, img in enumerate(sorted(dev_set))}


# ================================================================ probes
def fit_probe(dev, layers, C, target, folds):
    """Fit on the unique DEV questions (original view, all conditions): the full model and one model per
    held-out fold. Returns [full, fold 0, ..., fold 4]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, fold = [], [], []
    for kind, level in sorted({k[:2] for k in dev}):
        cs = [c for k, c in dev.items() if k[:2] == (kind, level)]
        merged = {f: np.concatenate([c[f] for c in cs]) for f in ("hidx_orig", "gold", "teacher", "image")}
        _, first = np.unique(merged["hidx_orig"], return_index=True)   # yes-questions repeat across splits
        X.append(features(hidden(kind, level, "orig"), merged["hidx_orig"][first], layers))
        y.append(merged["gold"][first] if target == "gold" else merged["teacher"][first] > 0)
        fold.append(np.array([folds[i] for i in merged["image"][first]]))
    X, y, fold = np.concatenate(X), np.concatenate(y), np.concatenate(fold)
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    models = [make().fit(X, y)] + [make().fit(X[fold != f], y[fold != f]) for f in range(N_FOLDS)]
    print(f"  {target} probe on layers {list(layers)}: {len(y)} DEV questions, C={C:g}", flush=True)
    return models


def probe_scores(models, layers, kinds_levels):
    """{(kind, level, view): (1 + N_FOLDS, n_rows) decision functions on every row of that view's log}."""
    out = {}
    for kind, level in kinds_levels:
        for v in VIEWS:
            H = hidden(kind, level, v)
            S = np.empty((len(models), len(H)))
            for s in range(0, len(H), CHUNK):
                X = features(H, np.arange(s, min(s + CHUNK, len(H))), layers)
                for i, m in enumerate(models):
                    S[i, s:s + len(X)] = m.decision_function(X)
            out[(kind, level, v)] = S
    return out


def per_view(data, S, folds, oof):
    """{key: {view: score per question}}: full-model scores (TEST) or out-of-fold scores (DEV)."""
    out = {}
    for key, c in data.items():
        row = 1 + np.array([folds[i] for i in c["image"]]) if oof else np.zeros(len(c["gold"]), int)
        out[key] = {v: S[(*key[:2], v)][row, c[f"hidx_{v}"]] for v in VIEWS}
    return out


def combine(P, views):
    return {k: np.mean([p[v] for v in views], axis=0) for k, p in P.items()}


def margin_mean(data, views, layer=READ_LAYER):
    return {k: np.mean([c[f"m_{v}"][:, layer - 1] for v in views], axis=0) for k, c in data.items()}


def view_label(views):
    return "+".join(views)


# ================================================================ evaluation
def head_to_head(test, a, b, boot, rng):
    """Per cell: AUROC(a) - AUROC(b) with a paired-bootstrap 95% CI over questions."""
    rows = []
    for key, c in test.items():
        g = c["gold"]
        idx = rng.integers(0, len(g), size=(boot, len(g)))
        d = np.array([al.auroc(a[key][i], g[i]) - al.auroc(b[key][i], g[i]) for i in idx])
        rows.append(dict(kind=key[0], level=key[1], split=key[2], delta=al.auroc(a[key], g) - al.auroc(b[key], g),
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cells = al.load_cells()
    dev_set = al.dev_images(cells)
    data = load(cells)
    if not data:
        raise SystemExit("no condition has all six views yet: run python -m scripts.run_layers --views "
                         + " ".join(VIEWS))
    dev = {k: subset(c, np.isin(c["image"], list(dev_set))) for k, c in data.items()}
    test = {k: subset(c, ~np.isin(c["image"], list(dev_set))) for k, c in data.items()}
    folds = fold_of(dev_set)
    kinds_levels = sorted({k[:2] for k in data})

    # ---- training-free reference rows (published configurations) and the baseline
    base_t = {k: al.layer_margin(c, al.FINAL) for k, c in test.items()}
    base_d = {k: al.layer_margin(c, al.FINAL) for k, c in dev.items()}
    methods = [  # (label, TEST scores, DEV scores for the threshold, passes, fixed threshold or None)
        ("Baseline (threshold 0)", base_t, base_d, 1, 0.0),
        ("Baseline (DEV threshold)*", base_t, base_d, 1, None),
        ("ViewLayer*", margin_mean(test, VIEWLAYER), margin_mean(dev, VIEWLAYER), 5, None),
        ("ViewLayer+DAE*", margin_mean(test, VIEWLAYER_DAE), margin_mean(dev, VIEWLAYER_DAE), 6, None),
    ]

    # ---- probes: fit on DEV, score every view; choose the view set on out-of-fold DEV scores
    print("fitting probes on DEV (original view):")
    all_sets = [s for r in range(1, len(VIEWS) + 1) for s in combinations(VIEWS, r)]
    sweep, picks = [], {}
    for name, (layers, C, target) in PROBES.items():
        S = probe_scores(fit_probe(dev, layers, C, target, folds), layers, kinds_levels)
        P_t, P_d = per_view(test, S, folds, oof=False), per_view(dev, S, folds, oof=True)
        for vs in all_sets:
            sweep.append(dict(probe=name, views=view_label(vs), n_views=len(vs),
                              dev_auroc=np.mean([al.auroc(s, dev[k]["gold"]) for k, s in combine(P_d, vs).items()])))
        sw = pd.DataFrame([r for r in sweep if r["probe"] == name])
        best = tuple(sw.loc[sw.dev_auroc.idxmax(), "views"].split("+"))
        picks[name] = best
        short = SHORT[name]
        fixed = [(("orig",), "1 view"), (("orig", "dae"), "orig + DAE view"),
                 (VIEWLAYER, "× ViewLayer views"), (VIEWLAYER_DAE, "× ViewLayer+DAE views")]
        for vs, desc in fixed:
            methods.append((f"{short}, {desc}*", combine(P_t, vs), combine(P_d, vs), len(vs), None))
        if best not in [vs for vs, _ in fixed]:
            methods.append((f"{short}, DEV views ({view_label(best)})*", combine(P_t, best), combine(P_d, best),
                            len(best), None))
        print(f"  DEV view set for {short}: {view_label(best)} (DEV AUROC {sw.dev_auroc.max():.4f})", flush=True)
    sweep = pd.DataFrame(sweep).sort_values(["probe", "dev_auroc"], ascending=[True, False])
    sweep.to_csv(out / "probe_views_dae_dev_sweep.csv", index=False, float_format="%.5f")

    # ---- TEST
    entries = []
    for label, s_t, s_d, passes, thr in methods:
        if thr is None:
            thr = al.best_threshold(dev, lambda k, c, s_d=s_d: s_d[k])
        entries.append((label, lambda k, c, s_t=s_t: s_t[k], thr, passes))
    comp = al.evaluate(test, entries, args.boot, np.random.default_rng(0))
    comp.to_csv(out / "probe_views_dae_cells.csv", index=False, float_format="%.5f")
    table = al.summary_table(comp, entries)

    vld = dict((m[0], m[1]) for m in methods)["ViewLayer+DAE*"]
    h2h = ["", "### Against ViewLayer+DAE (paired bootstrap on identical TEST questions)", "",
           "| Method | Passes | Mean ΔAUROC vs ViewLayer+DAE | Sig ↑/↓ | Δ Blur s4 | Δ Noise s4 |", "|---|---|---|---|---|---|"]
    for label, s_t, _, passes, _ in methods:
        if not (label.startswith("Gold") or label.startswith("Clean")):
            continue
        h = head_to_head(test, s_t, vld, args.boot, np.random.default_rng(1))
        s4 = lambda kind: h[(h.kind == kind) & (h.level == 4)].delta.mean()
        h2h.append(f"| {label} | {passes} | {h.delta.mean():+.4f} | {int((h.lo > 0).sum())}/{int((h.hi < 0).sum())} "
                   f"| {s4('blur'):+.3f} | {s4('noise'):+.3f} |")

    notes = [f"Probes fit on the original view of the DEV images only ({', '.join(f'{SHORT[n]}: layers {list(l)}, C={C:g}' for n, (l, C, _) in PROBES.items())}); "
             "each view is scored by the same probe and the scores are averaged.",
             "DEV view-set choice (out-of-fold, max mean DEV AUROC over "
             f"{len(all_sets)} subsets): " + "; ".join(f"{SHORT[n]}: {view_label(v)}" for n, v in picks.items()) + ".",
             "ViewLayer / ViewLayer+DAE: mean layer-20 Yes-No margin over 5 / 6 views (their published DEV configs), "
             "recomputed from the same logs."]
    n_cells = comp[["kind", "level", "split"]].drop_duplicates().shape[0]
    md = al.markdown(table, n_cells, int(comp.n.iloc[0]), args.boot, notes, script="scripts.analyze_probe_views_dae")
    if n_cells < 27:
        md = f"**Partial: {n_cells}/27 cells.**\n\n" + md
    (out / "probe_views_dae_table.md").write_text(md + "\n" + "\n".join(h2h) + "\n")
    print(md + "\n" + "\n".join(h2h))
    return table


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(al.N_JOBS):
        main()
