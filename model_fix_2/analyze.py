"""Score the null-contrast fix against baseline, VCD and the matched-threshold baseline.

    python -m model_fix_2.analyze [--boot 500]

Inputs (on ASPIRE, under $VCD_OUTPUT_ROOT):
    fix2_nulls_<kind>_s<level>/<split>.jsonl       model_fix_2/run_nulls.py
    stage2_{baseline,vcd}_greedy_<kind>_s<level>/   Stage 2 greedy runs (baseline answers, VCD answers
                                                    and the real/VCD-contrast first-token logits)

Held-out protocol (handout: keep test data separate from development):
    POPE's 500 images are split by image, seeded: 100 DEV images, 400 TEST images. alpha is chosen
    per null on DEV only (max mean AUROC over all 27 condition x split cells). Every TEST number
    uses that single alpha. The DEV sweep is saved so the choice is auditable.

Outputs (results/):
    fix2_alpha_dev.csv     DEV: AUROC and F1 per (null, alpha, cell)
    fix2_comparison.csv    TEST: per cell, baseline / VCD / fix, AUROC, F1, yes-ratio, matched-threshold
                           F1, null-branch AUROC (is the null really content-free?), JS, bootstrap CIs
"""
import argparse
import json
import random

import numpy as np
import pandas as pd

from model_fix_2.nulls import NULLS
from model_fix_2.run_nulls import ALPHAS, BETA, CONDITIONS, OUTPUT_ROOT, run_name
from scripts.analyze import auroc, f1_from
from fix_eval.common import decision_score, fix_metrics, reference_row, summarise, write_readme_results
from src.paths import POPE_SPLITS, REPO_ROOT

N_DEV_IMAGES = 100
SPLIT_SEED = 0


def read_jsonl(path) -> pd.DataFrame:
    with open(path) as f:
        return pd.DataFrame(json.loads(line) for line in f)


def load_cells() -> dict:
    """{(kind, level, split): merged per-question frame}; incomplete cells are skipped loudly."""
    cells = {}
    for kind, level in CONDITIONS:
        for split in POPE_SPLITS:
            paths = {
                "null": OUTPUT_ROOT / run_name(kind, level) / f"{split}.jsonl",
                "base": OUTPUT_ROOT / f"stage2_baseline_greedy_{kind}_s{level}" / f"{split}.jsonl",
                "vcd": OUTPUT_ROOT / f"stage2_vcd_greedy_{kind}_s{level}" / f"{split}.jsonl",
            }
            missing = [k for k, p in paths.items() if not p.exists()]
            if missing:
                print(f"skip {kind} s{level} {split}: missing {missing}")
                continue
            null = read_jsonl(paths["null"])
            base = read_jsonl(paths["base"])[["question_id", "answer"]].rename(columns={"answer": "answer_base"})
            vcd = read_jsonl(paths["vcd"]).rename(columns={"answer": "answer_vcd"})
            df = null.merge(base, on="question_id").merge(
                vcd[["question_id", "answer_vcd", "logit_yes", "logit_no", "logprob_yes", "logprob_no",
                     "logit_yes_cd", "logit_no_cd", "js"]], on="question_id")
            if not (len(df) == len(null) == len(base) == len(vcd) == 3000):
                print(f"skip {kind} s{level} {split}: incomplete ({len(null)}/{len(base)}/{len(vcd)})")
                continue
            cells[(kind, level, split)] = df.sort_values("question_id").reset_index(drop=True)
    return cells


def dev_images(cells: dict) -> set:
    images = sorted({img for df in cells.values() for img in df["image"]})
    return set(random.Random(SPLIT_SEED).sample(images, N_DEV_IMAGES))


def contrast_margin(df, yes, no, yes_c, no_c, lp_yes, lp_no, alpha, beta=BETA) -> np.ndarray:
    """First-token Yes-minus-No margin under the VCD rule (1+a)*real - a*contrast, with the
    plausibility mask. Same recomputation as scripts/analyze.vcd_margin (the vocab max is taken as
    max(logprob Yes, No), exact whenever the real branch's top-1 is Yes or No)."""
    z_yes = (1 + alpha) * df[yes] - alpha * df[yes_c]
    z_no = (1 + alpha) * df[no] - alpha * df[no_c]
    cut = np.log(beta) + np.maximum(df[lp_yes], df[lp_no])
    m = (z_yes - z_no).to_numpy(float)
    m[(df[lp_no] < cut).to_numpy()] = np.inf
    m[(df[lp_yes] < cut).to_numpy()] = -np.inf
    return m


def null_margin(df, null, alpha):
    return contrast_margin(df, "logit_yes_real", "logit_no_real", f"logit_yes_{null}", f"logit_no_{null}",
                           "logprob_yes_real", "logprob_no_real", alpha)


def vcd_margin(df):
    return contrast_margin(df, "logit_yes", "logit_no", "logit_yes_cd", "logit_no_cd",
                           "logprob_yes", "logprob_no", 1.0)


def says_yes(answers: pd.Series) -> np.ndarray:
    """Reference POPE scoring treats an answer as 'yes' iff it contains 'yes'."""
    return answers.str.lower().str.contains("yes").to_numpy()


def matched_f1(m_real: np.ndarray, yes_ratio: float, gold: np.ndarray) -> float:
    """F1 of the baseline margin thresholded to a given yes-ratio: a pure bias shift."""
    return f1_from(m_real > np.quantile(m_real, 1 - yes_ratio), gold)


def alpha_sweep(cells: dict, dev: set) -> pd.DataFrame:
    rows = []
    for (kind, level, split), df in cells.items():
        d = df[df["image"].isin(dev)]
        gold = (d["label"] == "yes").to_numpy()
        for null in NULLS:
            for a in ALPHAS:
                m_real = (d["logit_yes"] - d["logit_no"]).to_numpy(float)
                rows.append(dict(kind=kind, level=level, split=split, null=null, alpha=a,
                                 auroc=auroc(decision_score(null_margin(d, null, a), m_real), gold),
                                 f1=f1_from((d[f"ans_{null}_a{a}"] == "yes").to_numpy(), gold)))
    return pd.DataFrame(rows)


def compare(cells: dict, dev: set, chosen: dict, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for (kind, level, split), df in cells.items():
        t = df[~df["image"].isin(dev)].reset_index(drop=True)
        gold = (t["label"] == "yes").to_numpy()
        row = dict(kind=kind, level=level, split=split, **reference_row(t), js_vcd=t["js"].mean(),
                   # sanity: real branch recomputed in our batch vs Stage 2's logged real branch
                   real_branch_max_abs_diff=float(np.abs(
                       (t["logit_yes_real"] - t["logit_no_real"]) - (t["logit_yes"] - t["logit_no"])).max()))
        for null in NULLS:
            a = chosen[null]
            m_fix = null_margin(t, null, a)
            y_fix = (t[f"ans_{null}_a{a}"] == "yes").to_numpy()
            m_null = (t[f"logit_yes_{null}"] - t[f"logit_no_{null}"]).to_numpy(float)
            m_real_b = (t["logit_yes_real"] - t["logit_no_real"]).to_numpy(float)
            p = f"{null}_"
            row.update(fix_metrics(t, m_fix, y_fix, n_boot, rng, p, m_unmasked=(1 + a) * m_real_b - a * m_null))
            row.update({
                p + "alpha": a,
                p + "auroc_null_branch": auroc(m_null, gold),  # ~0.5 if the null is content-free
                p + "js": t[f"js_{null}"].mean(),
                p + "other_rate": (t[f"ans_{null}_a{a}"] == "other").mean(),
                p + "agree": ((m_fix > 0) == y_fix).mean(),  # recomputed margin vs exact decision
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()

    cells = load_cells()
    if not cells:
        raise SystemExit("no complete cells")
    dev = dev_images(cells)
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)

    sweep = alpha_sweep(cells, dev)
    sweep.to_csv(out / "fix2_alpha_dev.csv", index=False, float_format="%.5f")
    curve = sweep.groupby(["null", "alpha"])[["auroc", "f1"]].mean()
    chosen = {n: float(curve.loc[n, "auroc"].idxmax()) for n in NULLS}
    print(f"{len(cells)} cells; {len(dev)} dev images. DEV mean over cells:\n{curve.round(4)}\nchosen alpha: {chosen}")
    for n, a in chosen.items():
        if a in (min(ALPHAS), max(ALPHAS)):
            print(f"WARNING: alpha for {n} is at the edge of the grid {ALPHAS}; consider widening it")

    comp = compare(cells, dev, chosen, args.boot, np.random.default_rng(0))
    comp.to_csv(out / "fix2_comparison.csv", index=False, float_format="%.5f")

    pd.set_option("display.width", 250, "display.max_columns", None)
    by = comp.groupby(["kind", "level"], sort=False)
    cols = ["auroc_base", "auroc_vcd"] + [f"{n}_auroc" for n in NULLS] + [f"{n}_auroc_null_branch" for n in NULLS]
    print(f"\n== TEST AUROC, mean over splits\n{by[cols].mean().round(4)}")
    cols = (["f1_base", "f1_vcd", "f1_matched_vcd"] + [c for n in NULLS for c in (f"{n}_f1", f"{n}_f1_matched")]
            + ["yes_base", "yes_vcd"] + [f"{n}_yes" for n in NULLS])
    print(f"\n== TEST F1 / yes-ratio (%), mean over splits\n{(by[cols].mean() * 100).round(2)}")
    for n in NULLS:
        wins = (comp[f"{n}_d_auroc_base_lo"] > 0).sum()
        losses = (comp[f"{n}_d_auroc_base_hi"] < 0).sum()
        print(f"{n}: AUROC CI above baseline in {wins}/{len(comp)} cells, below in {losses}; "
              f"F1 CI above matched in {(comp[f'{n}_d_f1_matched_lo'] > 0).sum()}, "
              f"below in {(comp[f'{n}_d_f1_matched_hi'] < 0).sum()}; "
              f"margin/decision agreement {comp[f'{n}_agree'].min():.4f} (min)")
    print(f"real-branch max |diff| vs Stage 2: {comp['real_branch_max_abs_diff'].max():.3f} logits")
    md = "\n\n".join([
        f"_Auto-generated by `python -m model_fix_2.analyze` from {len(comp)} condition × split cells "
        f"({int(comp['n'].iloc[0])} TEST questions per cell; 400 TEST images, 100 DEV images held out). "
        "Means over the three POPE splits. Greedy decoding throughout._",
        f"### Phase-scrambled null contrast (α chosen on DEV = {chosen['phase']})",
        summarise(comp, "phase_", "phase"),
        f"### Mean-image null contrast, control (α chosen on DEV = {chosen['mean']})",
        summarise(comp, "mean_", "mean"),
        "### Is the null content-free? TEST AUROC of the null branch on its own (0.5 = no object evidence)",
        "\n".join(["| condition | " + " | ".join(NULLS) + " |", "|---|" + "---|" * len(NULLS)] + [
            f"| {'clean' if k == 'clean' else f'{k} s{l}'} | " + " | ".join(f"{r[f'{n}_auroc_null_branch']:.3f}" for n in NULLS) + " |"
            for (k, l), r in comp.groupby(["kind", "level"], sort=False).mean(numeric_only=True).iterrows()]),
    ])
    write_readme_results(REPO_ROOT / "model_fix_2" / "README.md", md)


if __name__ == "__main__":
    main()
