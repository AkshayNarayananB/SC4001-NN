"""Aggregate results/metrics/*.csv: mean ± std over seeds per (method, decoding, kind, level, split).

    python -m scripts.summarize                 # everything
    python -m scripts.summarize --prefix stage0 # one stage, with the VCD paper's numbers alongside
"""
import argparse

import pandas as pd

from src.paths import REPO_ROOT

METRICS_DIR = REPO_ROOT / "results" / "metrics"

# VCD paper (Leng et al., CVPR 2024) Table 1, LLaVA-1.5, POPE MSCOCO, sampling, mean of 5 runs (in %)
PAPER = {
    ("baseline", "random"): dict(accuracy=83.29, precision=92.13, recall=72.80, f1=81.33),
    ("vcd", "random"): dict(accuracy=87.73, precision=91.42, recall=83.28, f1=87.16),
    ("baseline", "popular"): dict(accuracy=81.88, precision=88.93, recall=72.80, f1=80.06),
    ("vcd", "popular"): dict(accuracy=85.38, precision=86.92, recall=83.28, f1=85.06),
    ("baseline", "adversarial"): dict(accuracy=78.96, precision=83.06, recall=72.75, f1=77.57),
    ("vcd", "adversarial"): dict(accuracy=80.88, precision=79.45, recall=83.29, f1=81.33),
}
METRICS = ["accuracy", "precision", "recall", "f1", "yes_ratio"]
SPLIT_ORDER = {"random": 0, "popular": 1, "adversarial": 2}
KIND_ORDER = {"clean": 0, "blur": 1, "noise": 2}


def load(prefix: str) -> pd.DataFrame:
    files = sorted(METRICS_DIR.glob(f"{prefix}*.csv"))
    assert files, f"no metrics files matching {prefix}* in {METRICS_DIR}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "param" not in df:
        df["param"] = pd.NA
    df["param"] = df["param"].fillna("")  # '' = the dataset's cached level, a number = on-the-fly sweep
    for col in ["restore", "contrast"]:  # fix B columns; absent from older metrics files
        df[col] = df[col].fillna("") if col in df else ""
    return df[df["n"] == df["n"].max()]  # drop smoke/partial runs


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "alpha", "beta", "noise_step", "restore", "contrast", "decoding", "kind", "level", "param", "split"]
    grouped = df.groupby(keys, dropna=False)  # baseline rows have empty alpha/beta/noise_step
    agg = grouped[METRICS].agg(["mean", "std"]) * 100
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["seeds"] = grouped.size()
    agg = agg.reset_index()
    agg["_split"] = agg["split"].map(SPLIT_ORDER)
    return agg.sort_values(["decoding", "kind", "level", "_split", "method", "noise_step"]).drop(columns="_split")


def label(r) -> str:
    restore = f"+{r['restore']}" if r["restore"] else ""
    if r["method"] == "baseline":
        return "baseline" + restore
    contrast = "T=degraded" if r["contrast"] == "degraded" else f"T={r['noise_step']:g}"
    return f"{r['method']}{restore}(a={r['alpha']:g},b={r['beta']:g},{contrast})"


def fmt(mean, std):
    return f"{mean:6.2f}" if pd.isna(std) else f"{mean:6.2f}±{std:4.2f}"


def print_pivot(table: pd.DataFrame, metric: str) -> None:
    """One metric as a grid: rows = (decoding, method, kind, level), columns = split."""
    table = table.assign(method_label=table.apply(label, axis=1))
    grid = table.pivot_table(index=["decoding", "method_label", "kind", "level", "param"], columns="split",
                             values=f"{metric}_mean", sort=False)
    grid = grid[[s for s in SPLIT_ORDER if s in grid.columns]]
    grid["mean"] = grid.mean(axis=1)
    print(f"{metric} (%)")
    print(grid.round(2).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="")
    parser.add_argument("--pivot", choices=METRICS, help="print one metric as a level x split grid")
    args = parser.parse_args()

    table = summarize(load(args.prefix))
    if args.pivot:
        table = (table.assign(_kind=table["kind"].map(KIND_ORDER), _param=pd.to_numeric(table["param"], errors="coerce"))
                 .sort_values(["decoding", "method", "noise_step", "_kind", "level", "_param"]).drop(columns=["_kind", "_param"]))
        print_pivot(table, args.pivot)
        return
    show_paper = args.prefix.startswith("stage0")
    for _, r in table.iterrows():
        line = (f"{r['decoding']:<6} {r['kind']}-s{r['level']} {r['split']:<11} {label(r):<24} n_seeds={r['seeds']}  "
                + "  ".join(f"{m[:4]} {fmt(r[m + '_mean'], r[m + '_std'])}" for m in METRICS))
        paper = PAPER.get((r["method"], r["split"]))
        if show_paper and r["decoding"] == "sample" and paper:
            line += f"   | paper acc {paper['accuracy']:.2f} f1 {paper['f1']:.2f} (Δf1 {r['f1_mean'] - paper['f1']:+.2f})"
        print(line)


if __name__ == "__main__":
    main()
