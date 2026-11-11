"""Paired baseline-vs-VCD analysis from per-question records.

    python -m scripts.analyze --stage stage2 [--boot 1000]

Reads $VCD_OUTPUT_ROOT/<name>/<split>.jsonl for every config under configs/<stage>/ and writes:
  results/<stage>_comparison.csv   per (decoding, kind, level, split): F1/acc/yes-ratio for both
                                   methods, VCD gain with a paired-bootstrap 95% CI, McNemar p (greedy)
  results/<stage>_margins.csv      greedy only: AUROC of real / contrast / VCD Yes-No margins, and F1 of a
                                   baseline thresholded to VCD's yes-ratio (is the gain a bias shift?)
  results/<stage>_diagnostics.csv  per (decoding, kind, level, split), VCD runs only: first-token
                                   contrast term dm = margin_real - margin_cd (margin = logit Yes - No),
                                   split by gold label, plus JS(real || contrast)
Run on ASPIRE (where the raw records live); the outputs are small and get copied back with results/.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata

from scripts.run_pope import OUTPUT_ROOT, load_config, select_questions
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
        cfg = load_config(str(path))
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
            df = df.assign(name=cfg["name"], method=cfg["method"], decoding=cfg["decoding"], seed=cfg["seed"],
                           kind=cfg["kind"], level=cfg["level"], split=split, limit=cfg["limit"],
                           restore=cfg.get("restore") or "", contrast=cfg.get("contrast", "diffusion"))
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["correct"] = [is_correct_reference(l, a) for l, a in zip(df["label"], df["answer"])]
    # predicted-yes consistent with the reference scoring rule
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
        # pair only seeds both methods have finished
        seeds = set(cell.loc[cell["method"] == "baseline", "seed"]) & set(cell.loc[cell["method"] == "vcd", "seed"])
        cell = cell[cell["seed"].isin(seeds)]
        if not seeds:
            continue
        pb, cb, gold = matrix(cell, "baseline")
        pv, cv, _ = matrix(cell, "vcd")
        f1_b, f1_v = f1_from(pb, gold).mean(), f1_from(pv, gold).mean()
        # paired bootstrap over questions (same resample for both methods and all seeds)
        idx = rng.integers(0, gold.size, size=(n_boot, gold.size))
        boot = (f1_from(pv[:, idx], gold[idx]).mean(0) - f1_from(pb[:, idx], gold[idx]).mean(0))
        row = dict(zip(KEYS, key))
        row.update(
            n_questions=gold.size, n_seeds=pb.shape[0],
            f1_base=f1_b, f1_vcd=f1_v, d_f1=f1_v - f1_b,
            d_f1_lo=np.percentile(boot, 2.5), d_f1_hi=np.percentile(boot, 97.5),
            acc_base=cb.mean(), acc_vcd=cv.mean(),
            yes_base=pb.mean(), yes_vcd=pv.mean(),
            # discordant pairs pooled over seeds: VCD fixes vs VCD breaks
            vcd_fixes=int((~cb & cv).sum()), vcd_breaks=int((cb & ~cv).sum()),
        )
        if row["decoding"] == "greedy":
            n = row["vcd_fixes"] + row["vcd_breaks"]
            row["mcnemar_p"] = binomtest(row["vcd_fixes"], n, 0.5).pvalue if n else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    vcd = df[df["method"] == "vcd"].copy()
    vcd["margin"] = vcd["logit_yes"] - vcd["logit_no"]
    vcd["margin_cd"] = vcd["logit_yes_cd"] - vcd["logit_no_cd"]
    vcd["dm"] = vcd["margin"] - vcd["margin_cd"]
    vcd["abs_dm"] = vcd["dm"].abs()
    rows = []
    for key, cell in vcd.groupby(KEYS):
        row = dict(zip(KEYS, key))
        row.update(mean_abs_dm=cell["abs_dm"].mean(), mean_js=cell["js"].mean(),
                   mean_margin=cell["margin"].mean(), mean_margin_cd=cell["margin_cd"].mean())
        for label in ["yes", "no"]:
            sub = cell[cell["label"] == label]
            row[f"dm_gold_{label}"] = sub["dm"].mean()
            row[f"margin_gold_{label}"] = sub["margin"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def auroc(score: np.ndarray, gold: np.ndarray) -> float:
    """Mann-Whitney AUROC of `score` for gold-yes vs gold-no (ties count half)."""
    r = rankdata(score)
    n_pos, n_neg = gold.sum(), (~gold).sum()
    return (r[gold].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def vcd_margin(cell: pd.DataFrame, alpha: float = 1.0, beta: float = 0.1) -> np.ndarray:
    """First-token VCD Yes-minus-No logit margin, recomputed from the logged logits. The plausibility
    cutoff uses max(logprob Yes, No) as the vocab max (exact whenever the top-1 token is Yes or No)."""
    z_yes = (1 + alpha) * cell["logit_yes"] - alpha * cell["logit_yes_cd"]
    z_no = (1 + alpha) * cell["logit_no"] - alpha * cell["logit_no_cd"]
    top = np.maximum(cell["logprob_yes"], cell["logprob_no"])
    cut = np.log(beta) + top
    m = (z_yes - z_no).to_numpy(float)
    m[(cell["logprob_no"] < cut).to_numpy()] = np.inf    # No masked out -> Yes
    m[(cell["logprob_yes"] < cut).to_numpy()] = -np.inf  # Yes masked out -> No
    return m


def decision_score(m: np.ndarray, m_real: np.ndarray) -> np.ndarray:
    """Turn a masked margin (+-inf = answer forced by the plausibility mask) into a ranking score that
    keeps every decision but orders forced answers by the real-branch margin instead of tying them.
    The mask forces 60-70% of POPE answers; left as +-inf they all tie, which lowers AUROC even for a
    method that changes no answer (the baseline with only VCD's mask: 0.750 -> 0.710 at blur s4)."""
    s = np.asarray(m, float).copy()
    pos, neg = np.isposinf(s), np.isneginf(s)
    s[pos] = 1e6 + m_real[pos]
    s[neg] = -1e6 + m_real[neg]
    return s


def margins(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    """Does VCD improve discrimination, or only shift the Yes/No threshold? Greedy VCD records only
    (first-token logits). For each cell: AUROC of the real-branch margin (= the baseline's), of the
    contrast-branch margin, and of VCD; and F1 of the baseline margin thresholded so its yes-ratio
    matches VCD's ("matched"), i.e. what a pure bias shift would get.

    VCD's AUROC is reported three ways: auroc_vcd ranks VCD's decisions with `decision_score` (the
    convention of fix_eval/common.py, and the one to quote); auroc_vcd_nomask is the raw contrastive
    margin before the plausibility mask; auroc_vcd_tied leaves masked answers as +-inf ties (the
    pre-2026-11-11 number, kept for reference only)."""
    rows = []
    vcd = df[(df["method"] == "vcd") & (df["decoding"] == "greedy")]
    for key, cell in vcd.groupby(KEYS):
        cell = cell.sort_values("question_id")
        gold = (cell["label"] == "yes").to_numpy()
        m = (cell["logit_yes"] - cell["logit_no"]).to_numpy(float)
        m_cd = (cell["logit_yes_cd"] - cell["logit_no_cd"]).to_numpy(float)
        mv = vcd_margin(cell)
        sv = decision_score(mv, m)
        mu = 2 * m - m_cd          # unmasked VCD margin at alpha = 1
        yes_vcd = (mv > 0).mean()

        def matched_f1(m, mv, gold):
            t = np.quantile(m, 1 - (mv > 0).mean())
            return f1_from(m > t, gold)

        idx = rng.integers(0, gold.size, size=(n_boot, gold.size))
        d_auc = np.array([auroc(sv[i], gold[i]) - auroc(m[i], gold[i]) for i in idx])
        d_auc_nm = np.array([auroc(mu[i], gold[i]) - auroc(m[i], gold[i]) for i in idx])
        d_f1m = np.array([f1_from(mv[i] > 0, gold[i]) - matched_f1(m[i], mv[i], gold[i]) for i in idx])
        row = dict(zip(KEYS, key))
        row.update(
            auroc_real=auroc(m, gold), auroc_cd=auroc(m_cd, gold), auroc_vcd=auroc(sv, gold),
            d_auroc_lo=np.percentile(d_auc, 2.5), d_auroc_hi=np.percentile(d_auc, 97.5),
            auroc_vcd_nomask=auroc(mu, gold),
            d_auroc_nomask_lo=np.percentile(d_auc_nm, 2.5), d_auroc_nomask_hi=np.percentile(d_auc_nm, 97.5),
            auroc_vcd_tied=auroc(mv, gold),
            f1_real=f1_from(m > 0, gold), f1_vcd=f1_from(mv > 0, gold), f1_matched=matched_f1(m, mv, gold),
            d_f1_vs_matched_lo=np.percentile(d_f1m, 2.5), d_f1_vs_matched_hi=np.percentile(d_f1m, 97.5),
            yes_real=(m > 0).mean(), yes_vcd=yes_vcd,
            # agreement of the recomputed VCD decision with the decoded answer (sanity check)
            agree_answer=((mv > 0) == cell["answer"].str.lower().str.startswith("yes").to_numpy()).mean(),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="stage2")
    parser.add_argument("--boot", type=int, default=1000)
    args = parser.parse_args()

    df = load_records(args.stage)
    print(f"{len(df)} records loaded")
    comp = compare(df, args.boot, np.random.default_rng(0))
    diag = diagnostics(df)
    marg = margins(df, min(args.boot, 500), np.random.default_rng(0))
    out = REPO_ROOT / "results"
    marg.to_csv(out / f"{args.stage}_margins.csv", index=False, float_format="%.5f")
    comp.to_csv(out / f"{args.stage}_comparison.csv", index=False, float_format="%.5f")
    diag.to_csv(out / f"{args.stage}_diagnostics.csv", index=False, float_format="%.5f")

    pd.set_option("display.width", 200)
    for decoding in sorted(comp["decoding"].unique()):
        sub = comp[comp["decoding"] == decoding]
        view = sub.groupby(["kind", "level"], sort=False)[["f1_base", "f1_vcd", "d_f1", "yes_base", "yes_vcd"]].mean() * 100
        print(f"\n== {decoding}: mean over splits (%)\n{view.round(2)}")
    view = diag.groupby(["decoding", "kind", "level"], sort=False)[["mean_abs_dm", "dm_gold_yes", "dm_gold_no", "mean_js"]].mean()
    print(f"\n== VCD contrast diagnostics, mean over splits\n{view.round(4)}")
    view = marg.groupby(["kind", "level"], sort=False)[["auroc_real", "auroc_cd", "auroc_vcd", "auroc_vcd_nomask", "auroc_vcd_tied", "f1_real", "f1_matched", "f1_vcd", "agree_answer"]].mean()
    print(f"\n== greedy first-token margins, mean over splits\n{view.round(4)}")


if __name__ == "__main__":
    main()
