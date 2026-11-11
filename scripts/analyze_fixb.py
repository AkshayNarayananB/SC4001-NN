"""Phase 2 fix B analysis: DAE-restored input vs the Stage 2 baseline and VCD, greedy, noise cells.

    python -m scripts.analyze_fixb [--boot 1000]

Variants (per question, aligned by question id within each (kind, level, split) cell):
  baseline, vcd        Stage 2 greedy runs on the degraded image
  b0  baseline + DAE   control: what restoration alone does
  b1  VCD + DAE        contrast = restored + diffusion noise (T=999)
  b2  VCD + DAE vs degraded   contrast = the original degraded image

Per variant: F1, yes-ratio, AUROC of the first-token Yes-No margin (VCD margins recomputed from the
logged logits, as in analyze.margins), and F1 of the *baseline* margin thresholded to the variant's
yes-ratio (a pure bias shift); for b1/b2 also f1_matched_b0_<v>, the restored baseline (b0) thresholded the
same way, which is the fair control for VCD on a restored input. For VCD variants also JS and mean |dm| (contrast size).

Key paired deltas with bootstrap 95% CIs over questions:
  d_auroc_restore  = AUROC(b0) - AUROC(baseline)            does restoration help on its own?
  d_auroc_b1_vcd   = AUROC(b1) - AUROC(vcd)                 does the fix beat plain VCD?
  interaction_auroc = [AUROC(b1) - AUROC(b0)] - [AUROC(vcd) - AUROC(baseline)]
      > 0 means VCD contributes more on a restored input: the fix helps VCD itself, not just the input.
Same for F1. Writes results/fixb_comparison.csv; run on ASPIRE where the raw records live.
"""
import argparse

import numpy as np
import pandas as pd

from scripts.analyze import auroc, f1_from, load_records, vcd_margin
from src.paths import REPO_ROOT

KEYS = ["kind", "level", "split"]


def variant(r) -> str:
    if r["restore"]:
        if r["method"] == "baseline":
            return "b0"
        return "b2" if r["contrast"] == "degraded" else "b1"
    return r["method"]


def margin(cell: pd.DataFrame) -> np.ndarray:
    if cell["method"].iloc[0] == "baseline":
        return (cell["logit_yes"] - cell["logit_no"]).to_numpy(float)
    return vcd_margin(cell)


def matched_f1(m_base, yes_ratio, gold):
    return f1_from(m_base > np.quantile(m_base, 1 - yes_ratio), gold)


def analyze(df: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    rows = []
    for key, cell in df.groupby(KEYS):
        by = {v: c.sort_values("question_id") for v, c in cell.groupby("variant")}
        if not {"baseline", "vcd"} <= set(by) or len(by) < 3:
            continue
        qids = by["baseline"]["question_id"].to_numpy()
        assert all((c["question_id"].to_numpy() == qids).all() for c in by.values()), key
        gold = (by["baseline"]["label"] == "yes").to_numpy()
        m = {v: margin(c) for v, c in by.items()}
        row = dict(zip(KEYS, key), n=gold.size)
        for v, mv in m.items():
            row[f"f1_{v}"] = float(f1_from(mv > 0, gold))
            row[f"yes_{v}"] = (mv > 0).mean()
            row[f"auroc_{v}"] = auroc(mv, gold)
            row[f"f1_matched_{v}"] = float(matched_f1(m["baseline"], (mv > 0).mean(), gold))
            if "b0" in m and v in ("b1", "b2"):  # restored baseline at this variant's yes-ratio: the fair bias-shift control
                row[f"f1_matched_b0_{v}"] = float(matched_f1(m["b0"], (mv > 0).mean(), gold))
            if "js" in by[v] and by[v]["js"].notna().all():
                c = by[v]
                row[f"js_{v}"] = c["js"].mean()
                row[f"abs_dm_{v}"] = ((c["logit_yes"] - c["logit_no"]) - (c["logit_yes_cd"] - c["logit_no_cd"])).abs().mean()

        idx = rng.integers(0, gold.size, size=(n_boot, gold.size))
        def stat(fn):
            return np.array([fn(i) for i in idx])
        A = lambda v, i: auroc(m[v][i], gold[i])
        F = lambda v, i: f1_from(m[v][i] > 0, gold[i])
        deltas = {}
        if "b0" in m:
            deltas["d_auroc_restore"] = lambda i: A("b0", i) - A("baseline", i)
            deltas["d_f1_restore"] = lambda i: F("b0", i) - F("baseline", i)
        for v in ["b1", "b2"]:
            if v in m:
                deltas[f"d_auroc_{v}_vcd"] = lambda i, v=v: A(v, i) - A("vcd", i)
                deltas[f"d_auroc_{v}_baseline"] = lambda i, v=v: A(v, i) - A("baseline", i)
                deltas[f"d_f1_{v}_vcd"] = lambda i, v=v: F(v, i) - F("vcd", i)
        if "b0" in m and "b1" in m:
            deltas["interaction_auroc"] = lambda i: (A("b1", i) - A("b0", i)) - (A("vcd", i) - A("baseline", i))
            deltas["interaction_f1"] = lambda i: (F("b1", i) - F("b0", i)) - (F("vcd", i) - F("baseline", i))
        all_i = np.arange(gold.size)
        for name, fn in deltas.items():
            boot = stat(fn)
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = float(fn(all_i)), np.percentile(boot, 2.5), np.percentile(boot, 97.5)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    df = pd.concat([load_records("stage2"), load_records("fixb")], ignore_index=True)
    df = df[(df["decoding"] == "greedy") & (df["kind"] == "noise") & df["limit"].isna()]
    df["variant"] = df.apply(variant, axis=1)
    print(f"{len(df)} greedy noise records; variants {sorted(df['variant'].unique())}")
    out = analyze(df, args.boot, np.random.default_rng(0))
    out.to_csv(REPO_ROOT / "results" / "fixb_comparison.csv", index=False, float_format="%.5f")

    pd.set_option("display.width", 250)
    variants = [v for v in ["baseline", "vcd", "b0", "b1", "b2"] if f"f1_{v}" in out]
    mean = out.groupby(["kind", "level"]).mean(numeric_only=True)
    for metric in ["f1", "f1_matched", "f1_matched_b0", "yes", "auroc", "js", "abs_dm"]:
        cols = [f"{metric}_{v}" for v in variants if f"{metric}_{v}" in mean]
        scale = 1 if metric in ("auroc", "js", "abs_dm") else 100
        print(f"\n== {metric}, mean over splits\n{(mean[cols] * scale).round(3)}")
    deltas = [c for c in out if c.startswith(("d_", "interaction")) and not c.endswith(("_lo", "_hi"))]
    print("\n== paired deltas per cell (value [95% CI])")
    for _, r in out.iterrows():
        print(f"noise s{r['level']} {r['split']:<11} " + "  ".join(
            f"{d}={r[d]:+.3f}[{r[d + '_lo']:+.3f},{r[d + '_hi']:+.3f}]" for d in deltas if d in r and pd.notna(r[d])))


if __name__ == "__main__":
    main()
