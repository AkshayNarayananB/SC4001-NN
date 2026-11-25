"""Consistency gate for run_readout (LLaVA-1.5, DEV set).
1. Correctness: the UNBATCHED run (--novcd-dir) reproduces the course pipeline's logs
   (results/raw/layers_<cond>/_cache.jsonl) at every logged layer: max |diff| <= --max-course (rounding level).
2. Batching: the batched real+VCD pass (--dir) differs from the unbatched pass only by fp16 kernel noise:
   mean |diff| over all layers <= --max-batch-mean and final-layer decision agreement >= --min-agree.
   (fp16 GEMMs at batch 2 use different kernels; the size of this noise is the same as A100 vs RTX 4500.)
    python -m scripts.journal.check_smoke --dir <smoke10>/llava-1.5-7b --novcd-dir <smoke10_novcd>/llava-1.5-7b
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
    ap.add_argument("--novcd-dir", required=True)
    ap.add_argument("--max-course", type=float, default=0.01)
    ap.add_argument("--max-batch-mean", type=float, default=0.05)
    ap.add_argument("--min-agree", type=float, default=0.99)
    a = ap.parse_args()
    fails = []
    for d in sorted(Path(a.novcd_dir).iterdir()):
        nv, bv = load(d / "rows.jsonl"), load(Path(a.dir) / d.name / "rows.jsonl")
        old_p = OUTPUT_ROOT / f"layers_{d.name}" / "_cache.jsonl"
        if old_p.exists():
            old = load(old_p)
            keys = [k for k in nv if k in old]
            diff = np.array([[nv[k]["lm"][l] - (old[k]["layer_yes"][l] - old[k]["layer_no"][l]) for l in range(32)]
                             for k in keys])
            print(f"{d.name}: unbatched vs course n={len(keys)} max |diff| over all layers {np.abs(diff).max():.4f}")
            if np.abs(diff).max() > a.max_course:
                fails.append(f"{d.name} course {np.abs(diff).max():.4f}")
        keys = [k for k in nv if k in bv]
        diff = np.array([np.array(bv[k]["lm"]) - np.array(nv[k]["lm"]) for k in keys])
        agree = np.mean([(bv[k]["m"] > 0) == (nv[k]["m"] > 0) for k in keys])
        print(f"{d.name}: batched vs unbatched mean |diff| {np.abs(diff).mean():.4f} max {np.abs(diff).max():.4f}; "
              f"decision agreement {agree:.4f}")
        if np.abs(diff).mean() > a.max_batch_mean or agree < a.min_agree:
            fails.append(f"{d.name} batch mean {np.abs(diff).mean():.4f} agree {agree:.4f}")
    if fails:
        raise SystemExit("GATE FAILED: " + "; ".join(fails))
    print("GATE OK")


if __name__ == "__main__":
    main()
