"""Calibrate severity levels of the sensor-realistic degradations on DEV images only.

    python -m scripts.journal.calibrate_levels [--n 600]

Greedy LLaVA-1.5 baseline (first-token Yes-No margin) on a fixed subset of the original POPE
random split (DEV), with each candidate (kind, parameter) applied on the fly at native resolution.
Writes results/journal/calibration.csv: kind, param, acc, f1, yes_ratio, auroc. Resumable per row.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from scripts.run_pope import select_questions
from src.degradations import apply, seed_for
from src.model import answer_token_ids, encode, load_model
from src.paths import POPE_SOURCE_IMAGES, REPO_ROOT, pope_question_file

CANDIDATES = [("clean", 0)] + \
    [("lowlight", p) for p in (200, 60, 20, 8, 3, 1.5)] + \
    [("motion", p) for p in (10, 15, 25, 40, 60, 90)] + \
    [("jpeg", p) for p in (10, 5, 3, 1)] + \
    [("blur", 10), ("noise", 0.7)]        # reference points from the existing ladder
OUT = REPO_ROOT / "results" / "journal" / "calibration.csv"


@torch.inference_mode()
def margins(model, processor, ids, qs, kind, param):
    out, cur, img = [], None, None
    for q in qs:
        if q["image"] != cur:
            cur = q["image"]
            img = Image.open(POPE_SOURCE_IMAGES / cur).convert("RGB")
            if kind != "clean":
                img = apply(img, kind, param, seed_for(cur, kind, 1))
        input_ids, pv = encode(processor, img, q["text"])
        logits = model(input_ids=input_ids.to(model.device), pixel_values=pv.to(model.device, model.dtype)).logits[0, -1]
        out.append(float(logits[ids["yes"]] - logits[ids["no"]]))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    qs = select_questions([json.loads(l) for l in open(pope_question_file("random")) if l.strip()], args.n)
    qs.sort(key=lambda q: q["image"])
    y = np.array([q["label"] == "yes" for q in qs])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(r["kind"], float(r["param"])) for r in csv.DictReader(open(OUT))}
    model, processor = load_model()
    ids = answer_token_ids(processor)
    new = not OUT.exists()
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "param", "n", "acc", "f1", "yes_ratio", "auroc"])
        if new:
            w.writeheader()
        for kind, param in CANDIDATES:
            if (kind, float(param)) in done:
                continue
            m = margins(model, processor, ids, qs, kind, param)
            pred = m > 0
            tp = (pred & y).sum()
            f1 = 2 * tp / (pred.sum() + y.sum())
            row = dict(kind=kind, param=param, n=len(qs), acc=round(100 * (pred == y).mean(), 2),
                       f1=round(100 * f1, 2), yes_ratio=round(100 * pred.mean(), 2),
                       auroc=round(roc_auc_score(y, m), 4))
            w.writerow(row)
            f.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
