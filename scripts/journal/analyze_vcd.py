"""Does VCD ground or only recalibrate? Per-cell table for one model on one POPE set.

    python -m scripts.journal.analyze_vcd --model llava-1.5-7b --set pope_fresh [--boot 1000]

Per cell (condition x split): baseline and VCD F1 / yes-ratio / FPR, AUROC of the baseline margin and of VCD
(decision-preserving ties), dAUROC with a 95% image-clustered bootstrap CI, and F1(VCD) - F1(matched-threshold
baseline) with its CI (the baseline thresholded to VCD's yes-ratio within every resample). Conditions whose run
is not finished are skipped. Writes results/journal/vcd_<set>_<model>.csv and a per-condition summary .md
(mean over splits, counts of cells whose CI excludes zero).
"""
import argparse

import numpy as np
import pandas as pd

from scripts.journal import protocol as P
from scripts.journal.run_readout import all_conditions
from src.paths import REPO_ROOT

OUT = REPO_ROOT / "results" / "journal"


def cell_stats(base, vcd, gold, imgs, n_boot, rng):
    dec = P.decision_score(vcd, base)
    pv, pb = vcd > 0, base > 0
    yr = pv.mean()
    row = dict(n=len(gold), f1_base=P.f1(pb, gold), f1_vcd=P.f1(pv, gold), f1_matched=P.matched_f1(base, yr, gold),
               yes_base=pb.mean(), yes_vcd=yr, fpr_base=pb[~gold].mean(), fpr_vcd=pv[~gold].mean(),
               auroc_base=P.auroc(base, gold), auroc_vcd=P.auroc(dec, gold), forced=np.isinf(vcd).mean())
    da, df = [], []
    for i in P.cluster_boot(imgs, n_boot, rng):
        g = gold[i]
        da.append(P.auroc(dec[i], g) - P.auroc(base[i], g))
        df.append(P.f1(pv[i], g) - P.matched_f1(base[i], pv[i].mean(), g))
    row["d_auroc"] = row["auroc_vcd"] - row["auroc_base"]
    row["d_auroc_lo"], row["d_auroc_hi"] = np.percentile(da, [2.5, 97.5])
    row["d_f1_matched"] = row["f1_vcd"] - row["f1_matched"]
    row["d_f1_lo"], row["d_f1_hi"] = np.percentile(df, [2.5, 97.5])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--set", default="pope_fresh")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for cond in all_conditions():
        try:
            run = P.Run(a.set, a.model, cond)
        except FileNotFoundError:
            continue
        base, vcd = run.col("m"), P.vcd_margin(run)
        for split in P.SPLITS:
            idx, gold, imgs = run.split_index(split)
            rows.append(dict(model=a.model, set=a.set, condition=cond, split=split,
                             **cell_stats(base[idx], vcd[idx], gold, imgs, a.boot, rng)))
            print(cond, split, f"dAUROC {rows[-1]['d_auroc']:+.4f}  dF1(vs matched) {rows[-1]['d_f1_matched']:+.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"vcd_{a.set}_{a.model}"
    df.to_csv(OUT / f"{stem}.csv", index=False)

    g = df.groupby("condition", sort=False)
    summ = g[["f1_base", "f1_vcd", "f1_matched", "yes_base", "yes_vcd", "fpr_base", "fpr_vcd",
              "auroc_base", "auroc_vcd", "d_auroc"]].mean()
    summ["auroc_sig_up/down"] = g.apply(lambda c: f"{(c.d_auroc_lo > 0).sum()}/{(c.d_auroc_hi < 0).sum()}")
    summ["f1_vs_matched_up/down"] = g.apply(lambda c: f"{(c.d_f1_lo > 0).sum()}/{(c.d_f1_hi < 0).sum()}")
    pct = ["f1_base", "f1_vcd", "f1_matched", "yes_base", "yes_vcd", "fpr_base", "fpr_vcd"]
    summ[pct] = (100 * summ[pct]).round(1)
    summ[["auroc_base", "auroc_vcd"]] = summ[["auroc_base", "auroc_vcd"]].round(3)
    summ["d_auroc"] = summ["d_auroc"].round(4)
    n_up = int((df.d_auroc_lo > 0).sum()); n_dn = int((df.d_auroc_hi < 0).sum())
    f_up = int((df.d_f1_lo > 0).sum()); f_dn = int((df.d_f1_hi < 0).sum())
    md = [f"# VCD vs baseline: {a.model} on {a.set}", "",
          f"{len(df)} cells ({df.condition.nunique()} conditions x splits), {a.boot} image-clustered bootstrap resamples. "
          f"VCD AUROC uses decision-preserving ties. Mean dAUROC {df.d_auroc.mean():+.4f}; "
          f"cells with CI above/below zero: AUROC {n_up}/{n_dn}, F1 vs matched-threshold baseline {f_up}/{f_dn}.", "",
          "Per condition, mean of the three splits (F1, yes-ratio, FPR in %):", "", summ.to_markdown(), ""]
    (OUT / f"{stem}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
