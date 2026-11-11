"""Paired baseline-vs-DoLa analysis from per-question records.

    python -m scripts.analyze_dola --stage stage2 [--boot 1000]

Reads $VCD_OUTPUT_ROOT/<name>/<split>.jsonl for every config under configs/<stage>/ and writes:
  results/<stage>_dola_comparison.csv   per (decoding, kind, level, split): F1/acc/yes-ratio for both
                                        methods, DoLa gain with a paired-bootstrap 95% CI
  results/<stage>_dola_diagnostics.csv  DoLa runs only: average JSD and routing layer distribution
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from scripts.run_pope import OUTPUT_ROOT, select_questions
import yaml

def safe_load_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("splits", ["random", "popular", "adversarial"])
    cfg.setdefault("limit", 3000)
    return cfg

from src.eval_pope import is_correct_reference
from src.paths import REPO_ROOT, pope_question_file

KEYS = ["decoding", "kind", "level", "split"]


def expected_ids(split: str, limit) -> set:
    with open(pope_question_file(split)) as f:
        return {q["question_id"] for q in select_questions([json.loads(l) for l in f if l.strip()], limit)}


def load_records(stage: str) -> pd.DataFrame:
    """Records of every finished (config, split); missing or still-running ones are skipped."""
    frames = []
    for path in sorted((REPO_ROOT / "configs" / stage).rglob("*.yaml")):
        # cfg = load_config(str(path))
        cfg = safe_load_config(str(path))
        for split in cfg["splits"]:
            raw = OUTPUT_ROOT / cfg["name"] / f"{split}.jsonl"
            if not raw.exists():
                print(f"missing {cfg['name']}/{split}")
                continue
            with open(raw) as f:
                df = pd.DataFrame(json.loads(line) for line in f)
            if set(df["question_id"]) != expected_ids(split, cfg["limit"]):
                print(f"partial {cfg['name']}/{split} ({len(df)} records)")
                continue
            df = df.assign(method=cfg["method"], decoding=cfg["decoding"], seed=cfg["seed"],
                           kind=cfg["kind"], level=cfg["level"], split=split)
            frames.append(df)
    
    if not frames:
        return pd.DataFrame()
        
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = [is_correct_reference(l, a) for l, a in zip(df["label"], df["answer"])]
    df["pred_yes"] = np.where(df["label"] == "yes", df["correct"], ~df["correct"])
    return df


def f1_from(pred_yes: np.ndarray, gold_yes: np.ndarray) -> np.ndarray:
    """F1 along the last axis; works on (..., n) arrays."""
    tp = (pred_yes & gold_yes).sum(-1)
    fp = (pred_yes & ~gold_yes).sum(-1)
    fn = (~pred_yes & gold_yes).sum(-1)
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def matrix(cell: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(seeds, questions) predicted-yes and correct matrices, plus gold labels, aligned by question id."""
    sub = cell[cell["method"] == method]
    pred = sub.pivot(index="seed", columns="question_id", values="pred_yes").sort_index(axis=1)
    corr = sub.pivot(index="seed", columns="question_id", values="correct").sort_index(axis=1)
    gold = sub.drop_duplicates("question_id").set_index("question_id")["label"].sort_index() == "yes"
    return pred.to_numpy(bool), corr.to_numpy(bool), gold.to_numpy(bool)


def compare(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for key, cell in df.groupby(KEYS):
        seeds = set(cell.loc[cell["method"] == "baseline", "seed"]) & set(cell.loc[cell["method"] == "dola_anchor", "seed"])
        cell = cell[cell["seed"].isin(seeds)]
        if not seeds:
            continue
            
        pb, cb, gold = matrix(cell, "baseline")
        pdola, cdola, _ = matrix(cell, "dola_anchor")
        f1_b, f1_d = f1_from(pb, gold).mean(), f1_from(pdola, gold).mean()
        
        idx = rng.integers(0, gold.size, size=(n_boot, gold.size))
        boot = (f1_from(pdola[:, idx], gold[idx]).mean(0) - f1_from(pb[:, idx], gold[idx]).mean(0))
        
        row = dict(zip(KEYS, key))
        row.update(
            n_questions=gold.size, n_seeds=pb.shape[0],
            f1_base=f1_b, f1_dola=f1_d, d_f1=f1_d - f1_b,
            d_f1_lo=np.percentile(boot, 2.5), d_f1_hi=np.percentile(boot, 97.5),
            acc_base=cb.mean(), acc_dola=cdola.mean(),
            yes_base=pb.mean(), yes_dola=pdola.mean(),
            dola_fixes=int((~cb & cdola).sum()), dola_breaks=int((cb & ~cdola).sum()),
        )
        n = row["dola_fixes"] + row["dola_breaks"]
        row["mcnemar_p"] = binomtest(row["dola_fixes"], n, 0.5).pvalue if n else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    dola = df[df["method"] == "dola_anchor"].copy()
    if dola.empty or "dola_layer" not in dola.columns:
        return pd.DataFrame()
        
    rows = []
    for key, cell in dola.groupby(KEYS):
        row = dict(zip(KEYS, key))
        row.update(mean_jsd=cell["dola_entropy"].mean())
        
        layer_counts = cell["dola_layer"].value_counts(normalize=True) * 100
        for l in [16, 18, 20, 22, 24, 26, 28]:
            row[f"layer_{l}_pct"] = layer_counts.get(l, 0.0)
            
        rows.append(row)
    return pd.DataFrame(rows)

from scipy.stats import rankdata

def auroc(score: np.ndarray, gold: np.ndarray) -> float:
    """Mann-Whitney AUROC of `score` for gold-yes vs gold-no (ties count half)."""
    r = rankdata(score)
    n_pos, n_neg = gold.sum(), (~gold).sum()
    return (r[gold].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def margins_auroc(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates AUROC based on the raw logit margins of the selected DoLa layer."""
    dola = df[df["method"] == "dola_anchor"].copy()
    if dola.empty or "logit_yes" not in dola.columns:
        return pd.DataFrame()
        
    rows = []
    for key, cell in dola.groupby(KEYS):
        cell = cell.sort_values("question_id")
        gold = (cell["label"] == "yes").to_numpy()
        # Margin = Logit(Yes) - Logit(No)
        m = (cell["logit_yes"] - cell["logit_no"]).to_numpy(float)
        
        row = dict(zip(KEYS, key))
        row["auroc"] = auroc(m, gold)
        rows.append(row)
    return pd.DataFrame(rows)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--stage", required=True, help="Config directory (e.g. stage0 or dola_eval)")
#     parser.add_argument("--boot", type=int, default=1000)
#     args = parser.parse_args()

#     df = load_records(args.stage)
#     if df.empty:
#         print(f"No records found for stage '{args.stage}'")
#         return
        
#     print(f"{len(df)} records loaded")
    
#     comp = compare(df, args.boot, np.random.default_rng(0))
#     diag = diagnostics(df)
    
#     out = REPO_ROOT / "results"
#     out.mkdir(exist_ok=True)
    
#     if not comp.empty:
#         comp.to_csv(out / f"{args.stage}_dola_comparison.csv", index=False, float_format="%.5f")
#         pd.set_option("display.width", 200)
#         for decoding in sorted(comp["decoding"].unique()):
#             sub = comp[comp["decoding"] == decoding]
#             view = sub.groupby(["kind", "level"], sort=False)[["f1_base", "f1_dola", "d_f1", "yes_base", "yes_dola"]].mean() * 100
#             print(f"\n== {decoding}: mean over splits (%)\n{view.round(2)}")
            
#     if not diag.empty:
#         diag.to_csv(out / f"{args.stage}_dola_diagnostics.csv", index=False, float_format="%.5f")
#         view = diag.groupby(["decoding", "kind", "level"], sort=False)[["mean_jsd", "layer_16_pct", "layer_22_pct", "layer_28_pct"]].mean()
#         print(f"\n== DoLa Routing Diagnostics, mean over splits\n{view.round(4)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="Config directory (e.g. stage0 or dola_eval)")
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    df = load_records(args.stage)
    if df.empty:
        print(f"No records found for stage '{args.stage}'")
        return
        
    print(f"{len(df)} records loaded")
    
    comp = compare(df, args.boot, np.random.default_rng(0))
    diag = diagnostics(df)
    auc_df = margins_auroc(df)  # Calculate AUROC
    
    out = REPO_ROOT / "results"
    out.mkdir(exist_ok=True)
    
    if not comp.empty:
        comp.to_csv(out / f"{args.stage}_dola_comparison.csv", index=False, float_format="%.5f")
        pd.set_option("display.width", 200)
        for decoding in sorted(comp["decoding"].unique()):
            sub = comp[comp["decoding"] == decoding]
            view = sub.groupby(["kind", "level"], sort=False)[["f1_base", "f1_dola", "d_f1", "yes_base", "yes_dola"]].mean() * 100
            print(f"\n== {decoding}: mean over splits (%)\n{view.round(2)}")
            
    if not diag.empty:
        diag.to_csv(out / f"{args.stage}_dola_diagnostics.csv", index=False, float_format="%.5f")
        view = diag.groupby(["decoding", "kind", "level"], sort=False)[["mean_jsd", "layer_16_pct", "layer_22_pct", "layer_28_pct"]].mean()
        print(f"\n== DoLa Routing Diagnostics, mean over splits\n{view.round(4)}")

    if not auc_df.empty:
        view = auc_df.groupby(["decoding", "kind", "level"], sort=False)[["auroc"]].mean()
        print(f"\n== Minimum Entropy AUROC, mean over splits\n{view.round(4)}")

        auc_df.to_csv(out / f"{args.stage}_auroc.csv", index=False)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     main()