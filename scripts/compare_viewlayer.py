"""Head-to-head on identical TEST questions: ViewLayer (recomputed here) vs the clean-teacher probe.

    python -m scripts.compare_viewlayer

ViewLayer exactly as its DEV selection chose (origin/main ViewLayer/README.md): mean Yes-No margin over the
views {orig, blur1, blur2, blur3, hflip} at layer 20, with a DEV-tuned threshold. Views come from
scripts/run_layers.py --views (available for the conditions in CONDS). The clean-teacher probe scores
are read from results/layers_scores.npz (scripts/analyze_layers.py). Paired bootstrap over questions for
AUROC(teacher) - AUROC(ViewLayer); risk-coverage (AURC) for both. Writes results/compare_viewlayer.md.
"""
import json

import numpy as np
import pandas as pd

from scripts.analyze_layers import auroc, best_threshold, dev_images, f1_from, layer_margin, load_cells, to_arrays
from scripts.analyze_selective import risk_coverage
from scripts.run_layers import run_dir_name
from scripts.run_pope import OUTPUT_ROOT
from src.paths import POPE_SPLITS, REPO_ROOT

CONDS = [("blur", 4), ("noise", 4)]
VIEWS = ("blur1", "blur2", "blur3", "hflip")
LAYER = 20
import os
TEACHER = os.environ.get("COMPARE_METHOD", "Clean-teacher probe L16+24 (no labels)*")


def view_margin(kind, level, split, view, keep):
    with open(OUTPUT_ROOT / run_dir_name(kind, level, view) / f"{split}.jsonl") as f:
        df = pd.DataFrame(json.loads(l) for l in f).sort_values("question_id").reset_index(drop=True)
    m = np.array(df["layer_yes"].tolist())[:, LAYER - 1] - np.array(df["layer_no"].tolist())[:, LAYER - 1]
    return m[keep]


def main():
    cells = load_cells()
    dev_set = dev_images(cells)
    z = np.load(REPO_ROOT / "results" / "layers_scores.npz")
    thr_teacher = float(z[f"__thr|{TEACHER}"])
    parts = {"dev": {}, "test": {}}
    for kind, level in CONDS:
        for split in POPE_SPLITS:
            df = cells[(kind, level, split)]
            for part, keep in (("dev", df["image"].isin(dev_set).to_numpy()), ("test", ~df["image"].isin(dev_set).to_numpy())):
                a = to_arrays(df[keep])
                views = [layer_margin(a, LAYER)] + [view_margin(kind, level, split, v, keep) for v in VIEWS]
                a["viewlayer"] = np.mean(views, axis=0)
                parts[part][(kind, level, split)] = a
    thr_vl = best_threshold(parts["dev"], lambda _, c: c["viewlayer"])

    rng = np.random.default_rng(0)
    rows = []
    for key, c in parts["test"].items():
        g = c["gold"]
        t = z["|".join([TEACHER, key[0], str(key[1]), key[2]])]
        assert len(t) == len(g)
        base = layer_margin(c, 32)
        idx = rng.integers(0, len(g), size=(1000, len(g)))
        d = np.array([auroc(t[i], g[i]) - auroc(c["viewlayer"][i], g[i]) for i in idx])
        rows.append(dict(cond=f"{key[0]} s{key[1]}", split=key[2],
                         auroc_base=auroc(base, g), auroc_viewlayer=auroc(c["viewlayer"], g), auroc_teacher=auroc(t, g),
                         d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5),
                         f1_viewlayer=f1_from(c["viewlayer"] > thr_vl, g), f1_teacher=f1_from(t > thr_teacher, g),
                         aurc_viewlayer=risk_coverage(c["viewlayer"], thr_vl, g)[1], aurc_teacher=risk_coverage(t, thr_teacher, g)[1]))
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / "results" / "compare_viewlayer.csv", index=False, float_format="%.4f")
    lines = [f"| Cell | AUROC baseline | AUROC ViewLayer (5 passes) | AUROC {TEACHER} (1 pass) | Δ method − ViewLayer [95% CI] "
             "| F1 ViewLayer | F1 method | AURC ViewLayer | AURC method |", "|---|" + "---|" * 8]
    for _, r in df.iterrows():
        d = r.auroc_teacher - r.auroc_viewlayer
        lines.append(f"| {r.cond} {r.split} | {r.auroc_base:.3f} | {r.auroc_viewlayer:.3f} | {r.auroc_teacher:.3f} | "
                     f"{d:+.3f} [{r.d_lo:+.3f}, {r.d_hi:+.3f}] | {100 * r.f1_viewlayer:.1f} | {100 * r.f1_teacher:.1f} | "
                     f"{r.aurc_viewlayer:.3f} | {r.aurc_teacher:.3f} |")
    m = df.groupby("cond")[["auroc_base", "auroc_viewlayer", "auroc_teacher", "f1_viewlayer", "f1_teacher", "aurc_viewlayer", "aurc_teacher"]].mean()
    for cond, r in m.iterrows():
        lines.append(f"| **{cond} mean** | {r.auroc_base:.3f} | {r.auroc_viewlayer:.3f} | {r.auroc_teacher:.3f} | "
                     f"{r.auroc_teacher - r.auroc_viewlayer:+.3f} | {100 * r.f1_viewlayer:.1f} | {100 * r.f1_teacher:.1f} | "
                     f"{r.aurc_viewlayer:.3f} | {r.aurc_teacher:.3f} |")
    md = "\n".join(["ViewLayer recomputed on the same TEST questions (views orig+blur1+blur2+blur3+hflip @ layer 20, "
                    f"DEV threshold {thr_vl:+.2f}) vs the clean-teacher probe. AURC: lower is better.", "", *lines])
    (REPO_ROOT / "results" / ("compare_viewlayer.md" if "teacher" in TEACHER else "compare_viewlayer_goldprobe.md")).write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
