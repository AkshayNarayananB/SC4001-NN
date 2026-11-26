"""Response surface of a capture sweep (run_capture.py): POPE metrics per capture setting.

    python -m scripts.journal.analyze_capture --set pope --model llava-1.5-7b --dir capture_calib_lim100

Per setting (light, speed, t, n, fusion), averaged over the 3 POPE splits: AUROC of the margin, F1 at threshold 0,
yes-ratio, accuracy; plus the capture time (n * t). Questions are restricted to the images present in the run.
Writes results/journal/capture_<set>_<model>_<dir>.csv and prints, per (light, speed), the settings ranked by AUROC.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from scripts.journal import protocol as P
from src.paths import REPO_ROOT

PAT = re.compile(r"L(?P<light>[\d.]+)_v(?P<speed>[\d.]+)_t(?P<t>[\d.]+)_n(?P<n>\d+)_(?P<fusion>\w+)")


def load_rows(d):
    files = sorted(d.glob("shard*/rows.jsonl")) or [d / "rows.jsonl"]
    return pd.DataFrame([json.loads(l) for f in files for l in open(f) if l.strip()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="pope")
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--dir", default="capture_calib_lim100")
    a = ap.parse_args()
    df = load_rows(P.JOURNAL_ROOT / a.set / a.model / a.dir)
    imgs = set(df.image)
    m = df.set_index(["cond", "image", "text"])["m"]
    recs = []
    for cond in df.cond.unique():
        mc = m.loc[cond]
        res = []
        for split in P.SPLITS:
            qs = [q for q in P.questions(a.set, split) if q["image"] in imgs]
            s = mc.reindex([(q["image"], q["text"]) for q in qs]).to_numpy()
            g = np.array([q["label"] == "yes" for q in qs])
            ok = ~np.isnan(s)
            s, g = s[ok], g[ok]
            res.append((P.auroc(s, g), P.f1(s > 0, g), float((s > 0).mean()), float(((s > 0) == g).mean())))
        r = PAT.match(cond).groupdict()
        rec = dict(cond=cond, light=float(r["light"]), speed=float(r["speed"]), t_ms=float(r["t"]), n=int(r["n"]),
                   fusion=r["fusion"])
        rec["capture_ms"] = rec["t_ms"] * rec["n"]
        rec.update(zip(("auroc", "f1", "yes", "acc"), np.mean(res, 0)))
        recs.append(rec)
    out = pd.DataFrame(recs).sort_values(["light", "speed", "capture_ms", "t_ms"])
    p = REPO_ROOT / "results" / "journal" / f"capture_{a.set}_{a.model}_{a.dir}.csv"
    out.to_csv(p, index=False)
    pd.set_option("display.width", 200)
    for (L, v), g in out.groupby(["light", "speed"]):
        print(f"\n== light {L} e-/ms (peak {L * 33.3:.1f} e- at 33 ms), speed {v} px/ms ==")
        print(g[["t_ms", "n", "fusion", "capture_ms", "auroc", "f1", "yes", "acc"]].round(3).to_string(index=False))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
