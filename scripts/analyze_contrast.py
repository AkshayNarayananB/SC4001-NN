"""Zero / shuffle-hook VCD rows for the Table-4 summary (first token, greedy).

    python -m scripts.analyze_contrast [--boot 500]

Real branch = layer-32 Yes/No logits from scripts/run_layers.py; contrast branch = scripts/run_contrast.py
(projector output zeroed, or vision patches shuffled; src/hooks.py). VCD rule, alpha=1, beta=0.1:
    m = (1+a)(y - n) - a(y_cd - n_cd), with the plausibility mask: if log p(No) < log(beta) + log p(Yes) the
    answer is forced Yes (and vice versa). Only the Yes/No difference enters the mask, so logits suffice.
AUROC uses fix_eval's decision-preserving score: forced answers are ordered by the real-branch margin.
Writes results/contrast_table.{csv,md} and results/contrast_cells.csv.
"""
import argparse
import json

import numpy as np

from scripts.analyze_layers import FINAL, dev_images, evaluate, layer_margin, load_cells, markdown, summary_table, to_arrays
from scripts.run_contrast import contrast_dir
from scripts.run_layers import CONDITIONS
from src.paths import REPO_ROOT

ALPHA, BETA = 1.0, 0.1
MODES = ("zero", "shuffle")


def load_cd(mode, kind, level):
    path = contrast_dir(mode, kind, level) / "_cache.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path)]
    n = max(r["hidx"] for r in rows) + 1
    cd = np.full(n, np.nan)
    for r in rows:
        cd[r["hidx"]] = r["cd_yes"] - r["cd_no"]
    return cd


def vcd_score(m_real, m_cd):
    m = (1 + ALPHA) * m_real - ALPHA * m_cd
    s = m.copy()
    s[-m_real < np.log(BETA)] = 1e6 + m_real[-m_real < np.log(BETA)]    # No implausible -> forced Yes
    s[m_real < np.log(BETA)] = -1e6 + m_real[m_real < np.log(BETA)]     # Yes implausible -> forced No
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args()
    cells = load_cells()
    dev_set = dev_images(cells)
    cds = {(m, k, l): load_cd(m, k, l) for m in MODES for k, l in CONDITIONS}
    complete = [m for m in MODES if all(cds[(m, k, l)] is not None and not np.isnan(cds[(m, k, l)]).any() for k, l in CONDITIONS)]
    if not complete:
        raise SystemExit("no complete contrast runs yet")
    print("complete modes:", complete)
    test = {}
    for key, df in cells.items():
        a = to_arrays(df[~df["image"].isin(dev_set)])
        for m in complete:
            a[f"cd_{m}"] = cds[(m, key[0], key[1])][a["hidx"]]
        test[key] = a
    base = lambda _, c: layer_margin(c, FINAL)
    entries = [("Baseline (threshold 0)", base, 0.0, 1)]
    for m in complete:
        entries.append((f"VCD, {m}-hook contrast (a=1, b=0.1)", lambda _, c, m=m: vcd_score(layer_margin(c, FINAL), c[f"cd_{m}"]), 0.0, 2))
    comp = evaluate(test, entries, args.boot, np.random.default_rng(0))
    comp.to_csv(REPO_ROOT / "results" / "contrast_cells.csv", index=False, float_format="%.5f")
    table = summary_table(comp, entries)
    table.to_csv(REPO_ROOT / "results" / "contrast_table.csv", index=False, float_format="%.4f")
    notes = ["Contrast branch = the same degraded image with the projector output zeroed (zero) or the vision patches "
             "shuffled (shuffle); VCD combination and plausibility mask as in the paper; decisions at threshold 0 (VCD's own rule)."]
    md = markdown(table, comp[["kind", "level", "split"]].drop_duplicates().shape[0], int(comp.n.iloc[0]), args.boot, notes,
                  script="scripts.analyze_contrast")
    (REPO_ROOT / "results" / "contrast_table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
