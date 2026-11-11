"""Shared evaluation for the Phase 2 fixes (model_fix_2, RITUAL, DeCo), so every method is scored
the same way and against the same references.

References come from the Stage 2 greedy runs (already on ASPIRE): baseline answers, VCD answers
(T=999, alpha=1, beta=0.1), and the real-branch / VCD-contrast first-token Yes/No logits.

Bars a fix has to clear (set by Stage 2, docs/findings.md):
  1. AUROC above the baseline's (VCD's is above it in only 1 of 27 cells)
  2. F1 above the matched-threshold baseline (baseline thresholded to the fix's own yes-ratio), i.e.
     the gain is not just a Yes/No bias shift

AUROC convention (important): VCD-style plausibility masks and DeCo's candidate set decide 60-70%
of POPE answers outright (the masked answer is forced). Scoring those as +-inf (as
scripts/analyze.py does) ties them all and drags AUROC down even for a method that changes
nothing: the baseline with only VCD's mask applied drops from 0.750 to 0.710 AUROC at blur s4.
We therefore rank with `decision_score`: each method's decisions are kept exactly, and forced
answers are ordered among themselves by the real-branch margin (on those questions the method
and the baseline agree). We also report the AUROC of the method's unmasked score.

Held-out protocol (handout: keep test data separate from development): POPE's 500 images are split
by image, seeded, into 100 DEV and 400 TEST images. Hyperparameters are only ever chosen on DEV.
"""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scripts.analyze import auroc, decision_score
from scripts.analyze import f1_from as _f1_from

from src.paths import POPE_SPLITS, REPO_ROOT

OUTPUT_ROOT = Path(os.environ.get("VCD_OUTPUT_ROOT", REPO_ROOT / "results" / "raw"))
CONDITIONS = [("clean", 0)] + [(k, l) for k in ("blur", "noise") for l in (1, 2, 3, 4)]
N_DEV_IMAGES = 100
SPLIT_SEED = 0
BETA = 0.1


def f1_from(pred_yes, gold_yes):
    """scripts.analyze.f1_from, returned as a plain float for 1-D inputs (it returns a 0-d array)."""
    out = _f1_from(np.asarray(pred_yes), np.asarray(gold_yes))
    return float(out) if np.ndim(out) == 0 else out

__all__ = ["OUTPUT_ROOT", "CONDITIONS", "POPE_SPLITS", "auroc", "f1_from", "read_jsonl", "load_stage2",
           "load_method_cells", "dev_images", "says_yes", "matched_f1", "vcd_margin", "vcd_margin_unmasked",
           "decision_score", "reference_row",
           "fix_metrics", "summarise", "write_readme_results"]


def read_jsonl(path) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.loads(line) for line in f)


def load_stage2(kind: str, level: int, split: str) -> pd.DataFrame | None:
    """Baseline answer + VCD answer and logits, one row per question (None if not on disk)."""
    base_p = OUTPUT_ROOT / f"stage2_baseline_greedy_{kind}_s{level}" / f"{split}.jsonl"
    vcd_p = OUTPUT_ROOT / f"stage2_vcd_greedy_{kind}_s{level}" / f"{split}.jsonl"
    if not (base_p.exists() and vcd_p.exists()):
        return None
    base = read_jsonl(base_p)[["question_id", "answer"]].rename(columns={"answer": "answer_base"})
    vcd = read_jsonl(vcd_p).rename(columns={"answer": "answer_vcd"})
    cols = ["question_id", "answer_vcd", "logit_yes", "logit_no", "logprob_yes", "logprob_no",
            "logit_yes_cd", "logit_no_cd", "js"]
    return base.merge(vcd[cols], on="question_id")


def load_method_cells(run_prefix: str, n_expected: int = 3000) -> dict:
    """{(kind, level, split): method records merged with Stage 2 references}. Incomplete cells are
    reported and skipped, never silently mixed in."""
    cells = {}
    for kind, level in CONDITIONS:
        for split in POPE_SPLITS:
            path = OUTPUT_ROOT / f"{run_prefix}_{kind}_s{level}" / f"{split}.jsonl"
            ref = load_stage2(kind, level, split)
            if not path.exists() or ref is None:
                print(f"skip {kind} s{level} {split}: {'no method records' if not path.exists() else 'no Stage 2'}")
                continue
            rec = read_jsonl(path)
            df = rec.merge(ref, on="question_id")
            if not (len(rec) == len(ref) == len(df) == n_expected):
                print(f"skip {kind} s{level} {split}: incomplete ({len(rec)} method / {len(ref)} stage2)")
                continue
            cells[(kind, level, split)] = df.sort_values("question_id").reset_index(drop=True)
    return cells


def dev_images(cells: dict) -> set:
    images = sorted({img for df in cells.values() for img in df["image"]})
    return set(random.Random(SPLIT_SEED).sample(images, N_DEV_IMAGES))


def says_yes(answers: pd.Series) -> np.ndarray:
    """Reference POPE scoring (src/eval_pope.py): an answer counts as 'yes' iff it contains 'yes'."""
    return answers.astype(str).str.lower().str.contains("yes").to_numpy()


def matched_f1(m_real: np.ndarray, yes_ratio: float, gold: np.ndarray) -> float:
    """F1 of the baseline margin thresholded to a given yes-ratio: what a pure bias shift gets."""
    return f1_from(m_real > np.quantile(m_real, 1 - yes_ratio), gold)


def vcd_margin(df: pd.DataFrame, alpha: float = 1.0, beta: float = BETA) -> np.ndarray:
    """First-token VCD Yes-minus-No margin from the Stage 2 logits (same as scripts/analyze.vcd_margin)."""
    z_yes = (1 + alpha) * df["logit_yes"] - alpha * df["logit_yes_cd"]
    z_no = (1 + alpha) * df["logit_no"] - alpha * df["logit_no_cd"]
    cut = np.log(beta) + np.maximum(df["logprob_yes"], df["logprob_no"])
    m = (z_yes - z_no).to_numpy(float)
    m[(df["logprob_no"] < cut).to_numpy()] = np.inf
    m[(df["logprob_yes"] < cut).to_numpy()] = -np.inf
    return m


def vcd_margin_unmasked(df: pd.DataFrame, alpha: float = 1.0) -> np.ndarray:
    return (((1 + alpha) * df["logit_yes"] - alpha * df["logit_yes_cd"])
            - ((1 + alpha) * df["logit_no"] - alpha * df["logit_no_cd"])).to_numpy(float)


def reference_row(t: pd.DataFrame) -> dict:
    gold = (t["label"] == "yes").to_numpy()
    m_real = (t["logit_yes"] - t["logit_no"]).to_numpy(float)
    y_base, y_vcd = says_yes(t["answer_base"]), says_yes(t["answer_vcd"])
    return dict(n=len(t), auroc_base=auroc(m_real, gold), auroc_vcd=auroc(decision_score(vcd_margin(t), m_real), gold),
                auroc_vcd_nomask=auroc(vcd_margin_unmasked(t), gold),
                f1_base=f1_from(y_base, gold), f1_vcd=f1_from(y_vcd, gold),
                acc_base=(y_base == gold).mean(), acc_vcd=(y_vcd == gold).mean(),
                yes_base=y_base.mean(), yes_vcd=y_vcd.mean(),
                f1_matched_vcd=matched_f1(m_real, y_vcd.mean(), gold))


def fix_metrics(t: pd.DataFrame, m_fix: np.ndarray, y_fix: np.ndarray, n_boot: int, rng, prefix: str,
                m_unmasked: np.ndarray | None = None) -> dict:
    """AUROC / F1 / yes-ratio of a fix on TEST rows `t`, with paired-bootstrap 95% CIs for
    AUROC(fix) - AUROC(baseline), AUROC(fix) - AUROC(VCD) and F1(fix) - F1(matched baseline)."""
    gold = (t["label"] == "yes").to_numpy()
    m_real = (t["logit_yes"] - t["logit_no"]).to_numpy(float)
    m_fix = decision_score(m_fix, m_real)
    m_vcd = decision_score(vcd_margin(t), m_real)
    idx = rng.integers(0, len(t), size=(n_boot, len(t)))
    d_base = np.array([auroc(m_fix[i], gold[i]) - auroc(m_real[i], gold[i]) for i in idx])
    d_vcd = np.array([auroc(m_fix[i], gold[i]) - auroc(m_vcd[i], gold[i]) for i in idx])
    d_f1m = np.array([f1_from(y_fix[i], gold[i]) - matched_f1(m_real[i], y_fix[i].mean(), gold[i]) for i in idx])
    p = prefix
    return {
        f"{p}auroc": auroc(m_fix, gold),
        f"{p}auroc_nomask": auroc(m_unmasked, gold) if m_unmasked is not None else np.nan,
        f"{p}d_auroc_base_lo": np.percentile(d_base, 2.5), f"{p}d_auroc_base_hi": np.percentile(d_base, 97.5),
        f"{p}d_auroc_vcd_lo": np.percentile(d_vcd, 2.5), f"{p}d_auroc_vcd_hi": np.percentile(d_vcd, 97.5),
        f"{p}f1": f1_from(y_fix, gold), f"{p}acc": (y_fix == gold).mean(), f"{p}yes": y_fix.mean(),
        f"{p}f1_matched": matched_f1(m_real, y_fix.mean(), gold),
        f"{p}d_f1_matched_lo": np.percentile(d_f1m, 2.5), f"{p}d_f1_matched_hi": np.percentile(d_f1m, 97.5),
    }


def summarise(comp: pd.DataFrame, prefix: str, label: str) -> str:
    """Markdown summary of one method's TEST results: per-condition means over the 3 splits, plus
    how many of the 27 cells clear each bar with a 95% CI."""
    p = prefix
    by = comp.groupby(["kind", "level"], sort=False)
    t = by[["auroc_base", "auroc_vcd", f"{p}auroc", "auroc_vcd_nomask", f"{p}auroc_nomask", "f1_base", "f1_vcd",
            f"{p}f1", f"{p}f1_matched", "yes_base", "yes_vcd", f"{p}yes"]].mean()
    lines = [f"| condition | AUROC base | AUROC VCD | AUROC {label} | unmasked AUROC VCD | unmasked AUROC {label} "
             f"| F1 base | F1 VCD | F1 {label} | F1 matched-baseline | yes% base | yes% VCD | yes% {label} |",
             "|---|" + "---|" * 12]
    for (kind, level), r in t.iterrows():
        cond = "clean" if kind == "clean" else f"{kind} s{level}"
        nm = r[p + "auroc_nomask"]
        lines.append(f"| {cond} | {r['auroc_base']:.3f} | {r['auroc_vcd']:.3f} | **{r[p + 'auroc']:.3f}** "
                     f"| {r['auroc_vcd_nomask']:.3f} | {'—' if np.isnan(nm) else f'{nm:.3f}'} "
                     f"| {100 * r['f1_base']:.1f} | {100 * r['f1_vcd']:.1f} | **{100 * r[p + 'f1']:.1f}** "
                     f"| {100 * r[p + 'f1_matched']:.1f} | {100 * r['yes_base']:.1f} | {100 * r['yes_vcd']:.1f} "
                     f"| {100 * r[p + 'yes']:.1f} |")
    n = len(comp)
    count = lambda col, op: int((op(comp[col])).sum())
    lines += ["", "AUROC = decision-preserving ranking (forced answers ordered by the real-branch margin; see "
              "`fix_eval/common.py`). Unmasked AUROC = the method's raw score before its mask/candidate set.", "",
              f"Out of {n} condition × split cells (95% paired-bootstrap CIs, TEST images only):",
              f"- AUROC above baseline in **{count(p + 'd_auroc_base_lo', lambda s: s > 0)}**, "
              f"below in {count(p + 'd_auroc_base_hi', lambda s: s < 0)}",
              f"- AUROC above VCD in **{count(p + 'd_auroc_vcd_lo', lambda s: s > 0)}**, "
              f"below in {count(p + 'd_auroc_vcd_hi', lambda s: s < 0)}",
              f"- F1 above the matched-threshold baseline in **{count(p + 'd_f1_matched_lo', lambda s: s > 0)}**, "
              f"below in {count(p + 'd_f1_matched_hi', lambda s: s < 0)}"]
    return "\n".join(lines)


def write_readme_results(readme: Path, markdown: str) -> None:
    """Replace the block between the RESULTS markers in a README (or append one)."""
    start, end = "<!-- RESULTS START -->", "<!-- RESULTS END -->"
    text = readme.read_text() if readme.exists() else ""
    block = f"{start}\n{markdown.strip()}\n{end}"
    if start in text and end in text:
        text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    readme.write_text(text)
