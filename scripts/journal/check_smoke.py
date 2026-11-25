"""Consistency checks for run_readout (LLaVA-1.5, DEV set):
1. margin m and layer-20 margin equal the course pipeline's logs (results/raw/layers_<cond>/_cache.jsonl);
2. batched real+VCD pass gives the same real-branch margin as the unbatched pass.
    python -m scripts.journal.check_smoke --dir <smoke run root>/llava-1.5-7b
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_pope import OUTPUT_ROOT


def load(p):
    return {(r["image"], r["text"]): r for r in map(json.loads, open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--novcd-dir", default=None, help="same smoke run without --vcd (batched vs unbatched)")
    ap.add_argument("--max-diff", type=float, default=None, help="exit 1 if any max |dm| exceeds this (gate)")
    a = ap.parse_args()
    worst = 0.0
    for d in sorted(Path(a.dir).iterdir()):
        new = load(d / "rows.jsonl")
        old_p = OUTPUT_ROOT / f"layers_{d.name}" / "_cache.jsonl"
        if old_p.exists():
            old = load(old_p)
            keys = [k for k in new if k in old]
            dm = np.array([new[k]["m"] - (old[k]["layer_yes"][-1] - old[k]["layer_no"][-1]) for k in keys])
            d20 = np.array([new[k]["lm"][19] - (old[k]["layer_yes"][19] - old[k]["layer_no"][19]) for k in keys])
            print(f"{d.name}: n={len(keys)} |m - course| max {np.abs(dm).max():.4f} mean {np.abs(dm).mean():.4f}; "
                  f"layer20 max {np.abs(d20).max():.4f}; sign agreement {np.mean(np.sign([new[k]['m'] for k in keys]) == np.sign([old[k]['layer_yes'][-1]-old[k]['layer_no'][-1] for k in keys])):.4f}")
            worst = max(worst, np.abs(dm).max(), np.abs(d20).max())
        if a.novcd_dir and (Path(a.novcd_dir) / d.name / "rows.jsonl").exists():
            nv = load(Path(a.novcd_dir) / d.name / "rows.jsonl")
            dm = np.array([new[k]["m"] - nv[k]["m"] for k in new if k in nv])
            print(f"{d.name}: batched-vs-unbatched |dm| max {np.abs(dm).max():.4f} mean {np.abs(dm).mean():.4f}")
            worst = max(worst, np.abs(dm).max())
    print(f"worst |diff| = {worst:.4f}")
    if a.max_diff is not None and worst > a.max_diff:
        raise SystemExit(f"GATE FAILED: {worst:.4f} > {a.max_diff}")


if __name__ == "__main__":
    main()
