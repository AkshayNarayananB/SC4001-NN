"""Score every single-pass layer-readout method offline from scripts/run_layers.py, under the team's
Phase 2 protocol, and write a Table-4-style summary.

    python -m scripts.analyze_layers [--boot 500] [--no-probe]

Methods (all one forward pass, first-token Yes-No score):
  routing  top-k entropy routing (as run: K=5, T=0.5; and K, T chosen on DEV), soft (K=7), hard
           min-entropy (K=1), gradient early exit, a fixed layer. Pure functions of the per-layer
           Yes/No logits and entropies; nothing is trained.
  probes   logistic regression fit on DEV images only:
             lens-LR     on the 32 per-layer Yes-No margins + 32 per-layer entropies
             probe-L<l>  on the last-token hidden state of layer l
             probe-cat   on several layers' hidden states concatenated
           Regularisation C is chosen by 5-fold image-grouped CV inside DEV. Controls: object-only LR
           (the queried object's name, no image) measures how much a pure object prior gets.

Protocol (ported from origin/main fix_eval/common.py and ViewLayer/analyze.py so numbers are
comparable with Table 4):
  - POPE's 500 images are split by image into 100 DEV / 400 TEST (random.Random(0), sorted names).
  - Every choice (K, T, patience, epsilon, layer, C, threshold) is made on DEV only, as one global
    choice for all conditions (the method is never told how degraded an image is).
  - TEST: AUROC, paired-bootstrap 95% CI of AUROC(method) - AUROC(baseline), and of F1(method) -
    F1(matched-threshold baseline) (the baseline margin thresholded to the method's yes-ratio).
  - "thr 0" rows answer Yes iff score > 0; rows marked * use one global DEV-tuned threshold
    (for probes: tuned on out-of-fold DEV scores).
Baseline = layer 32 of the same forward passes.

Outputs: results/layers_dev_sweep.csv, results/layers_cells.csv, results/layers_table.{csv,md}
"""
import argparse
import json
import random
import re
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.run_layers import CONDITIONS, HIDDEN_DIM, HIDDEN_LAYERS, cond_name
from scripts.run_pope import OUTPUT_ROOT
from src.paths import POPE_SPLITS, REPO_ROOT, pope_question_file

N_DEV_IMAGES, SPLIT_SEED = 100, 0
SOFT_LAYERS = [18, 20, 22, 24, 26, 28, 30]   # src/decode.py SOFT_LAYERS
GRAD_LAYERS = list(range(16, 31))            # src/decode.py GRAD_LAYERS
FINAL = 32
N_JOBS = 8                                   # keep the machine responsive


# ================================================================ metrics
def auroc(score, gold) -> float:
    r = rankdata(score)
    n_pos, n_neg = gold.sum(), (~gold).sum()
    return (r[gold].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def f1_from(pred, gold) -> float:
    tp, fp, fn = (pred & gold).sum(), (pred & ~gold).sum(), (~pred & gold).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def matched_f1(m_real, yes_ratio, gold) -> float:
    return f1_from(m_real > np.quantile(m_real, 1 - yes_ratio), gold)


# ================================================================ routing rules (score = Yes - No)
# c: dict with Y, N, E arrays (n_questions, 32); column l-1 is layer l.
def layer_margin(c, layer):
    return c["Y"][:, layer - 1] - c["N"][:, layer - 1]


def topk_margin(c, k, temp, layers=SOFT_LAYERS):
    """src/decode.route_top_k: blend the k lowest-entropy candidate layers with softmax(-H / temp)."""
    idx = np.array(layers) - 1
    m, e = (c["Y"] - c["N"])[:, idx], c["E"][:, idx]
    order = np.argsort(e, axis=1, kind="stable")[:, :k]
    m_k, e_k = np.take_along_axis(m, order, 1), np.take_along_axis(e, order, 1)
    w = np.exp(-(e_k - e_k.min(1, keepdims=True)) / temp)
    w /= w.sum(1, keepdims=True)
    return (w * m_k).sum(1), (w * np.array(layers)[order]).sum(1)


def gradient_margin(c, patience, epsilon, ema=0.7):
    """src/decode.route_gradient over layers 16..30."""
    out, chosen = np.empty(len(c["E"])), np.empty(len(c["E"]))
    for i, ent in enumerate(c["E"]):
        smooth, best, best_l, strikes = None, np.inf, GRAD_LAYERS[0], 0
        for l in GRAD_LAYERS:
            smooth = ent[l - 1] if smooth is None else ema * ent[l - 1] + (1 - ema) * smooth
            if smooth < best:
                best, best_l = smooth, l
            strikes = strikes + 1 if smooth > best + epsilon else 0
            if strikes >= patience:
                break
        out[i], chosen[i] = c["Y"][i, best_l - 1] - c["N"][i, best_l - 1], best_l
    return out, chosen


# ================================================================ data
def question_texts() -> dict:
    texts = {}
    for split in POPE_SPLITS:
        with open(pope_question_file(split)) as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    texts[(split, q["question_id"])] = q["text"]
    return texts


def load_cells() -> dict:
    texts = question_texts()
    cells = {}
    for kind, level in CONDITIONS:
        for split in POPE_SPLITS:
            path = OUTPUT_ROOT / f"layers_{cond_name(kind, level)}" / f"{split}.jsonl"
            if not path.exists():
                print(f"skip {kind} s{level} {split}: not run yet")
                continue
            with open(path) as f:
                df = pd.DataFrame(json.loads(l) for l in f).sort_values("question_id").reset_index(drop=True)
            df["object"] = [re.sub(r"^Is there an? (.*) in the im.*$", r"\1", texts[(split, q)]).strip().lower()
                            for q in df["question_id"]]
            cells[(kind, level, split)] = df
    return cells


def to_arrays(df) -> dict:
    return {"Y": np.array(df["layer_yes"].tolist()), "N": np.array(df["layer_no"].tolist()),
            "E": np.array(df["layer_ent"].tolist()), "gold": (df["label"] == "yes").to_numpy(),
            "hidx": df["hidx"].to_numpy(), "image": df["image"].to_numpy(), "object": df["object"].to_numpy()}


def dev_images(cells) -> set:
    images = sorted({img for df in cells.values() for img in df["image"]})
    return set(random.Random(SPLIT_SEED).sample(images, N_DEV_IMAGES))


def hidden_memmap(kind, level):
    path = OUTPUT_ROOT / f"layers_{cond_name(kind, level)}" / "_hidden.f16"
    return np.memmap(path, dtype=np.float16, mode="r").reshape(-1, len(HIDDEN_LAYERS), HIDDEN_DIM)


# ================================================================ DEV selection helpers
def mean_dev_auroc(dev, fn) -> float:
    return float(np.mean([auroc(fn(k, c), c["gold"]) for k, c in dev.items()]))


def best_threshold(dev, fn) -> float:
    """One global threshold maximising mean DEV F1 across cells (ViewLayer/analyze.best_threshold)."""
    scores = [fn(k, c) for k, c in dev.items()]
    golds = [c["gold"] for c in dev.values()]
    grid = np.quantile(np.concatenate(scores), np.linspace(0.02, 0.98, 193))
    f1s = [np.mean([f1_from(s > t, g) for s, g in zip(scores, golds)]) for t in grid]
    return float(grid[int(np.argmax(f1s))])


# ================================================================ probes
class Probe:
    """Logistic regression on per-question features, fit on the unique DEV questions of the given
    conditions. score(key, c) = decision function on TEST rows; oof(key, c) = 5-fold image-grouped
    out-of-fold decision function on DEV rows (for thresholds).

    target="gold":    fit to the human POPE labels.
    target="teacher": fit to the model's OWN answer on the clean version of the same image (sign of the
                      clean-image layer-32 margin): self-distillation, no human labels in training.
    In both cases C is selected by the out-of-fold AUROC against the DEV gold labels (DEV labels used only
    for model selection, as ViewLayer uses them to choose its views, layer and threshold)."""

    def __init__(self, name, featurize, dev, Cs=(1e-3, 1e-2, 1e-1, 1.0), target="gold", train_conds=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.name, self.featurize = name, featurize
        X, y, groups, keys = [], [], [], []
        gold = []
        for (kind, level), rows in unique_rows(dev).items():
            if train_conds is not None and (kind, level) not in train_conds:
                continue
            X.append(featurize(kind, level, rows))
            y.append(rows["gold"] if target == "gold" else rows["teacher"] > 0)
            gold.append(rows["gold"])
            groups.append(rows["image"])
            keys += [(kind, level, h) for h in rows["hidx"]]
        X, y, groups, gold = np.concatenate(X), np.concatenate(y), np.concatenate(groups), np.concatenate(gold)
        make = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        folds = list(GroupKFold(5).split(X, y, groups))
        cv = {}
        for C in Cs:
            oof = np.empty(len(y))
            for tr, va in folds:
                oof[va] = make(C).fit(X[tr], y[tr]).decision_function(X[va])
            cv[C] = (auroc(oof, gold), oof)
        self.C = max(cv, key=lambda C: cv[C][0])
        self.cv_auroc = cv[self.C][0]
        self.oof_by_key = dict(zip(keys, cv[self.C][1]))
        self.model = make(self.C).fit(X, y)
        agree = f", teacher agrees with gold on {(y == gold).mean():.3f}" if target == "teacher" else ""
        print(f"  {name}: {X.shape[1]} features, {len(y)} DEV questions, C={self.C:g}{agree}, "
              f"CV AUROC {self.cv_auroc:.4f}  ({', '.join(f'{C:g}:{a:.4f}' for C, (a, _) in cv.items())})")

    def score(self, key, c):
        kind, level, _ = key
        return self.model.decision_function(self.featurize(kind, level, c))

    def oof(self, key, c):
        kind, level, _ = key
        return np.array([self.oof_by_key[(kind, level, h)] for h in c["hidx"]])


def unique_rows(part: dict) -> dict:
    """{(kind, level): arrays of the unique questions (by hidx) across the three splits}."""
    out = {}
    for kind, level in CONDITIONS:
        cs = [c for (k, l, _), c in part.items() if (k, l) == (kind, level)]
        if not cs:
            continue
        merged = {f: np.concatenate([c[f] for c in cs]) for f in cs[0]}
        _, first = np.unique(merged["hidx"], return_index=True)
        out[(kind, level)] = {f: v[first] for f, v in merged.items()}
    return out


def lens_features(kind, level, c):
    return np.concatenate([c["Y"] - c["N"], c["E"]], axis=1)


def hidden_features(layers):
    cols = [HIDDEN_LAYERS.index(l) for l in layers]

    def featurize(kind, level, c):
        H = hidden_memmap(kind, level)
        return np.asarray(H[c["hidx"]][:, cols], dtype=np.float32).reshape(len(c["hidx"]), -1)
    return featurize


def object_features(vocab):
    index = {o: i for i, o in enumerate(vocab)}

    def featurize(kind, level, c):
        X = np.zeros((len(c["object"]), len(vocab) + 1), np.float32)
        for i, o in enumerate(c["object"]):
            X[i, index.get(o, len(vocab))] = 1
        return X
    return featurize


# ================================================================ TEST evaluation + table
def evaluate(test, entries, boot, rng):
    """entries: (label, score_fn(key, c), threshold, passes). Returns per-cell rows."""
    rows = []
    for key, c in test.items():
        gold, m_base = c["gold"], layer_margin(c, FINAL)
        idx = rng.integers(0, len(gold), size=(boot, len(gold)))
        auc_base_boot = np.array([auroc(m_base[i], gold[i]) for i in idx])
        for label, fn, thr, passes in entries:
            m = fn(key, c)
            y = m > thr
            row = dict(method=label, passes=passes, kind=key[0], level=key[1], split=key[2], n=len(gold),
                       auroc=auroc(m, gold), auroc_base=auroc(m_base, gold), f1=f1_from(y, gold),
                       yes=y.mean(), acc=(y == gold).mean(), f1_matched=matched_f1(m_base, y.mean(), gold))
            if not label.startswith("Baseline"):
                d_auc = np.array([auroc(m[i], gold[i]) for i in idx]) - auc_base_boot
                d_f1 = np.array([f1_from(y[i], gold[i]) - matched_f1(m_base[i], y[i].mean(), gold[i]) for i in idx])
                row.update(d_auroc_lo=np.percentile(d_auc, 2.5), d_auroc_hi=np.percentile(d_auc, 97.5),
                           d_f1m_lo=np.percentile(d_f1, 2.5), d_f1m_hi=np.percentile(d_f1, 97.5))
            rows.append(row)
    return pd.DataFrame(rows)


def summary_table(comp, entries):
    def at(sub, kind, level, col):
        return sub[(sub.kind == kind) & (sub.level == level)][col].mean()

    table = []
    for label, _, thr, passes in entries:
        sub = comp[comp.method == label]
        base = label.startswith("Baseline")
        table.append({
            "Method": label, "Passes": passes,
            "Mean dAUROC": np.nan if base else (sub.auroc - sub.auroc_base).mean(),
            "Sig AUROC": "–" if base else f"{int((sub.d_auroc_lo > 0).sum())}/{int((sub.d_auroc_hi < 0).sum())}",
            "Sig F1": "–" if base else f"{int((sub.d_f1m_lo > 0).sum())}/{int((sub.d_f1m_hi < 0).sum())}",
            "AUROC Clean": at(sub, "clean", 0, "auroc"), "AUROC Blur s4": at(sub, "blur", 4, "auroc"),
            "AUROC Noise s4": at(sub, "noise", 4, "auroc"),
            "F1 Blur s4": 100 * at(sub, "blur", 4, "f1"), "F1 Noise s4": 100 * at(sub, "noise", 4, "f1"),
            "Yes% Blur s4": 100 * at(sub, "blur", 4, "yes"), "threshold": thr,
        })
    return pd.DataFrame(table)


def markdown(table, n_cells, n_test, boot, notes, script="scripts.analyze_layers"):
    f3 = lambda v: f"{v:.3f}".replace("0.", ".", 1)
    lines = ["| Method | Passes | Mean ΔAUROC | Sig ↑/↓ AUROC | Sig ↑/↓ F1 | AUROC Clean | AUROC Blur s4 "
             "| AUROC Noise s4 | F1 Blur s4 | F1 Noise s4 |", "|---|" + "---|" * 9]
    for _, r in table.iterrows():
        d = "–" if np.isnan(r["Mean dAUROC"]) else f"{r['Mean dAUROC']:+.4f}".replace("0.", ".", 1)
        lines.append(f"| {r.Method} | {r.Passes} | {d} | {r['Sig AUROC']} | {r['Sig F1']} | {f3(r['AUROC Clean'])} "
                     f"| {f3(r['AUROC Blur s4'])} | {f3(r['AUROC Noise s4'])} | {r['F1 Blur s4']:.1f} | {r['F1 Noise s4']:.1f} |")
    head = (f"TEST images only: {n_cells} condition × split cells, {n_test} questions each (400 TEST images; the "
            f"100 DEV images are used only for choosing hyperparameters, thresholds and fitting probes). "
            f"Sig ↑/↓: cells whose 95% paired-bootstrap interval ({boot} resamples) lies above/below zero, "
            "against the baseline (AUROC) or the matched-threshold baseline (F1). * = DEV-tuned global threshold.")
    return "\n".join([head, "", *lines, "", *notes, "", f"Generated by `python -m {script}`."])


# ================================================================ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()
    cells = load_cells()
    if not cells:
        raise SystemExit("no layer readouts yet: run python -m scripts.run_layers")
    if len(cells) < 27:
        print(f"warning: only {len(cells)}/27 cells available; results are partial")
    dev_set = dev_images(cells)
    for (kind, level, split), df in cells.items():
        # the model's own clean-image Yes-No margin for the same question (teacher for self-distillation)
        clean = cells[("clean", 0, split)]
        assert (clean["question_id"].to_numpy() == df["question_id"].to_numpy()).all()
        df["teacher"] = layer_margin(to_arrays(clean), FINAL)
    dev = {k: {**to_arrays(df[df["image"].isin(dev_set)]), "teacher": df.loc[df["image"].isin(dev_set), "teacher"].to_numpy()}
           for k, df in cells.items()}
    test = {k: to_arrays(df[~df["image"].isin(dev_set)]) for k, df in cells.items()}
    out = REPO_ROOT / "results"

    # ---- routing: DEV sweeps (max mean DEV AUROC over all cells)
    sweep = []
    for k, t in product(range(1, 8), [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]):
        sweep.append(dict(family="topk", k=k, temp=t, dev_auroc=mean_dev_auroc(dev, lambda _, c: topk_margin(c, k, t)[0])))
    for p, e in product([1, 2, 3], [0.01, 0.05, 0.1, 0.25]):
        sweep.append(dict(family="gradient", patience=p, epsilon=e, dev_auroc=mean_dev_auroc(dev, lambda _, c: gradient_margin(c, p, e)[0])))
    for l in range(12, 33):
        sweep.append(dict(family="layer", layer=l, dev_auroc=mean_dev_auroc(dev, lambda _, c: layer_margin(c, l))))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "layers_dev_sweep.csv", index=False, float_format="%.5f")
    pick = lambda fam: sweep[sweep.family == fam].loc[lambda s: s.dev_auroc.idxmax()]
    tk, gr, ly = pick("topk"), pick("gradient"), pick("layer")
    tk_k, tk_t, gp, ge, bl = int(tk.k), float(tk.temp), int(gr.patience), float(gr.epsilon), int(ly.layer)
    base_dev = sweep[(sweep.family == "layer") & (sweep.layer == FINAL)].dev_auroc.iloc[0]
    notes = [f"DEV choices: top-k K={tk_k}, T={tk_t:g} (DEV AUROC {tk.dev_auroc:.4f} vs baseline {base_dev:.4f}); "
             f"gradient patience={gp}, ε={ge:g} ({gr.dev_auroc:.4f}); fixed layer {bl} ({ly.dev_auroc:.4f})."]
    print(notes[0])

    base = lambda _, c: layer_margin(c, FINAL)
    entries = [  # (label, score fn, threshold or "dev", passes)
        ("Baseline (threshold 0)", base, 0.0, 1),
        ("Baseline (DEV threshold)*", base, "dev", 1),
        ("Top-k as run (K=5, T=0.5)", lambda _, c: topk_margin(c, 5, 0.5)[0], 0.0, 1),
        (f"Top-k (DEV K={tk_k}, T={tk_t:g})", lambda _, c: topk_margin(c, tk_k, tk_t)[0], 0.0, 1),
        (f"Top-k (DEV K={tk_k}, T={tk_t:g})*", lambda _, c: topk_margin(c, tk_k, tk_t)[0], "dev", 1),
        ("Soft routing (K=7, T=0.5)", lambda _, c: topk_margin(c, 7, 0.5)[0], 0.0, 1),
        ("Hard min-entropy (K=1)", lambda _, c: topk_margin(c, 1, 1.0)[0], 0.0, 1),
        (f"Gradient exit (DEV p={gp}, ε={ge:g})", lambda _, c: gradient_margin(c, gp, ge)[0], 0.0, 1),
        (f"Layer {bl} only (DEV)*", lambda _, c: layer_margin(c, bl), "dev", 1),
    ]
    dev_fns = {}  # label -> fn used on DEV rows for threshold tuning (out-of-fold for probes)

    # ---- probes
    if not args.no_probe:
        print("fitting probes on DEV:")
        vocab = sorted({o for c in dev.values() for o in c["object"]})
        probes = [Probe("Object prior only (no image)", object_features(vocab), dev),
                  Probe("Lens-LR (32 margins + 32 entropies)", lens_features, dev)]
        for l in HIDDEN_LAYERS:
            probes.append(Probe(f"Probe L{l}", hidden_features([l]), dev))
        best2 = sorted(probes[2:], key=lambda p: -p.cv_auroc)[:3]
        cat_layers = sorted(int(p.name.split("L")[-1]) for p in best2)
        probes.append(Probe(f"Probe L{'+'.join(map(str, cat_layers))}", hidden_features(cat_layers), dev))
        # clean-teacher probes (no human labels in training): layer set and C chosen by DEV-CV vs gold
        teachers = [Probe(f"Clean-teacher probe L{'+'.join(map(str, ls))} (no labels)", hidden_features(list(ls)), dev,
                          Cs=(1e-4, 3e-4, 1e-3), target="teacher") for ls in [(20,), (24,), (16, 24)]]
        best_teacher = max(teachers, key=lambda p: p.cv_auroc)
        probes.append(best_teacher)
        for p in probes:
            entries.append((p.name + "*", p.score, "dev", 1))
            dev_fns[p.name + "*"] = p.oof
        holdout_specs = [("clean+blur", "noise"), ("clean+noise", "blur")]
        layers_t = [int(x) for x in re.findall(r"\d+", best_teacher.name.split("(")[0])]
        notes.append(f"Clean-teacher probe: logistic regression on layers {layers_t} fit to the model's own clean-image "
                     "answer for each degraded DEV question (no human labels); layer set and C chosen by DEV-CV.")
        notes.append("Probes: logistic regression (standardised features) fit on the unique DEV questions of all nine "
                     "conditions; C by 5-fold image-grouped CV on DEV: " +
                     "; ".join(f"{p.name} C={p.C:g} (CV AUROC {p.cv_auroc:.4f})" for p in probes) + ".")

    entries = [(lab, fn, best_threshold(dev, dev_fns.get(lab, fn)) if thr == "dev" else thr, passes)
               for lab, fn, thr, passes in entries]
    comp = evaluate(test, entries, args.boot, np.random.default_rng(0))
    comp.to_csv(out / "layers_cells.csv", index=False, float_format="%.5f")
    table = summary_table(comp, entries)
    table.to_csv(out / "layers_table.csv", index=False, float_format="%.4f")
    md = markdown(table, comp[["kind", "level", "split"]].drop_duplicates().shape[0], int(comp.n.iloc[0]), args.boot, notes)
    (out / "layers_table.md").write_text(md + "\n")
    print(md)
    pd.set_option("display.width", 220)
    print(comp.groupby(["method", "kind", "level"], sort=False)[["auroc", "auroc_base", "f1", "f1_matched", "yes"]]
          .mean().round(4).to_string())

    # ---- held-out degradation type: does the clean-teacher readout transfer to an unseen corruption?
    if not args.no_probe:
        ho_lines = ["| Trained on | Tested on (unseen) | Mean ΔAUROC | Sig ↑/↓ AUROC | AUROC s4 | baseline AUROC s4 |", "|---|---|---|---|---|---|"]
        for train_kinds, test_kind in holdout_specs:
            tc = [c for c in CONDITIONS if c[0] in train_kinds.split("+")]
            p = Probe(f"teacher {train_kinds}", hidden_features(layers_t), dev, Cs=(best_teacher.C,), target="teacher", train_conds=tc)
            sub_dev = {k: c for k, c in dev.items() if k[0] in train_kinds.split("+")}
            thr = best_threshold(sub_dev, p.oof)
            sub = {k: c for k, c in test.items() if k[0] == test_kind}
            r = evaluate(sub, [("Baseline", base, 0.0, 1), ("teacher", p.score, thr, 1)], args.boot, np.random.default_rng(0))
            t = r[r.method == "teacher"]
            s4 = t[t.level == 4]
            ho_lines.append(f"| {train_kinds} | {test_kind} s1–s4 | {(t.auroc - t.auroc_base).mean():+.4f} | "
                            f"{int((t.d_auroc_lo > 0).sum())}/{int((t.d_auroc_hi < 0).sum())} | {s4.auroc.mean():.3f} | {s4.auroc_base.mean():.3f} |")
        ho = "\n".join(["", "### Held-out degradation type (clean-teacher probe)", "", *ho_lines])
        (out / "layers_table.md").write_text(md + "\n" + ho + "\n")
        print(ho)

    # per-question TEST scores for downstream analyses (selective answering)
    np.savez_compressed(out / "layers_scores.npz", **{
        f"{lab}|{k[0]}|{k[1]}|{k[2]}": fn(k, c) for lab, fn, _, _ in entries for k, c in test.items()},
        **{f"__gold|{k[0]}|{k[1]}|{k[2]}": c["gold"] for k, c in test.items()},
        **{f"__thr|{lab}": np.array(thr) for lab, _, thr, _ in entries})


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(N_JOBS):
        main()
