"""Evaluation protocol for the journal extension (shared by every analysis script).

* Data: a run directory of scripts/journal/run_readout.py, joined with the POPE question files of
  a POPE set, gives one array per split (cell = condition x split).
* DEV = POPE_SET "pope" (the 500 original images); TEST = "pope_fresh" (2000 disjoint images).
  Every hyperparameter, probe weight and threshold is fitted on DEV only.
* AUROC with decision-preserving ties for masked methods (VCD), matched-threshold baseline F1,
  and an image-clustered paired bootstrap (resample images, keep all their questions).
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from scripts.run_pope import OUTPUT_ROOT
from src.paths import DATA_ROOT, POPE_SPLITS

JOURNAL_ROOT = OUTPUT_ROOT / "journal"


def run_dir(pope_set: str, model: str, cond: str, view: str = "orig") -> Path:
    return JOURNAL_ROOT / pope_set / model / (cond + ("" if view == "orig" else f"__{view}"))


@lru_cache(maxsize=None)
def questions(pope_set: str, split: str) -> tuple:
    p = DATA_ROOT / pope_set / "questions" / f"coco_pope_{split}.json"
    return tuple(json.loads(l) for l in open(p) if l.strip())


class Run:
    """Rows + hidden states of one (POPE set, model, condition, view)."""

    def __init__(self, pope_set, model, cond, view="orig"):
        self.dir = d = run_dir(pope_set, model, cond, view)
        if not (d / "DONE").exists():
            raise FileNotFoundError(f"{d} not finished")
        self.meta = json.loads((d / "meta.json").read_text())
        self.rows = [json.loads(l) for l in open(d / "rows.jsonl")]
        self.index = {(r["image"], r["text"]): i for i, r in enumerate(self.rows)}
        self.pope_set = pope_set

    def col(self, key) -> np.ndarray:
        return np.array([r[key] for r in self.rows], dtype=float)

    def layer_margins(self) -> np.ndarray:            # [n, L]
        return np.array([r["lm"] for r in self.rows], dtype=np.float32)

    def hidden(self) -> np.ndarray:                   # [n, n_hidden_layers, dim] float16 memmap
        k = len(self.meta["hidden_layers"])
        raw = np.memmap(self.dir / "_hidden.f16", dtype=np.float16, mode="r")
        return raw.reshape(len(self.rows), k, -1)

    def split_index(self, split):
        """(row indices, gold-yes bool array, image ids) for one POPE split, in question order."""
        qs = questions(self.pope_set, split)
        idx = np.array([self.index[(q["image"], q["text"])] for q in qs])
        gold = np.array([q["label"] == "yes" for q in qs])
        imgs = np.array([q["image"] for q in qs])
        return idx, gold, imgs


def vcd_margin(run: Run, alpha=1.0, beta=0.1) -> np.ndarray:
    """First-token VCD margin with the plausibility mask (+-inf = forced answer). Exact: the cutoff
    uses the logged vocabulary max of the real branch."""
    yes, no, top = run.col("yes"), run.col("no"), run.col("top")
    z = (1 + alpha) * (yes - no) - alpha * (run.col("cd_yes") - run.col("cd_no"))
    cut = np.log(beta) + top
    z[no < cut] = np.inf
    z[yes < cut] = -np.inf
    return z


def decision_score(m, m_real):
    s = np.asarray(m, float).copy()
    pos, neg = np.isposinf(s), np.isneginf(s)
    s[pos] = 1e6 + m_real[pos]
    s[neg] = -1e6 + m_real[neg]
    return s


def auroc(score, gold) -> float:
    r = rankdata(score)
    n_pos, n_neg = gold.sum(), (~gold).sum()
    return float((r[gold].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def f1(pred, gold) -> float:
    tp = (pred & gold).sum()
    return float(2 * tp / max(pred.sum() + gold.sum(), 1))


def matched_f1(base_score, yes_ratio, gold) -> float:
    """F1 of the baseline thresholded so its yes-ratio equals `yes_ratio` (a pure bias shift)."""
    k = int(round(yes_ratio * len(base_score)))
    pred = np.zeros(len(base_score), bool)
    if k > 0:
        pred[np.argsort(-base_score, kind="stable")[:k]] = True
    return f1(pred, gold)


def cluster_boot(images, n_boot, rng):
    """Index arrays for an image-clustered bootstrap (all questions of a resampled image are kept)."""
    uniq, inv = np.unique(images, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick])


def delta_auroc_ci(a, b, gold, images, n_boot=1000, seed=0):
    """AUROC(a) - AUROC(b) with a 95% image-clustered paired bootstrap interval."""
    rng = np.random.default_rng(seed)
    d = auroc(a, gold) - auroc(b, gold)
    bs = [auroc(a[i], gold[i]) - auroc(b[i], gold[i]) for i in cluster_boot(images, n_boot, rng)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d, lo, hi


def best_threshold(scores_by_cell, golds_by_cell, grid=None) -> float:
    """Single global threshold maximising the mean F1 over cells (fit on DEV)."""
    allv = np.concatenate(scores_by_cell)
    grid = np.quantile(allv, np.linspace(0.02, 0.98, 193)) if grid is None else grid
    means = [np.mean([f1(s > t, g) for s, g in zip(scores_by_cell, golds_by_cell)]) for t in grid]
    return float(grid[int(np.argmax(means))])


SPLITS = POPE_SPLITS
