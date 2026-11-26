"""Why does the DEV-fit probe lose to the baseline on ExDark? And do 'safe' variants fix it?

    python -m scripts.journal.diagnose_transfer --model llava-1.5-7b

Teacher probe (C fixed from analyze_readout) fit on all DEV conditions, then on ExDark and a few fresh TEST cells:
  pooled AUROC vs within-object AUROC (mean over objects): if the probe is worse pooled but not within objects,
  the loss comes from object-prior miscalibration, not from the visual signal.
Variants:
  teacher        logistic on standardised hidden states (as in the paper draft)
  residual       logistic on [margin * 100, Z]: the margin column is effectively unpenalised, so the probe is
                 the model's own margin plus an L2-shrunk correction (falls back to the margin when unsure)
  objcentred     hidden states centred per queried object (object means from DEV) before the probe
  residual+obj   both
"""
import argparse
import re

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.journal import protocol as P
from scripts.journal.run_readout import all_conditions

OBJ_RE = re.compile(r"Is there an? (.+) in the image\?")


def obj(text):
    return OBJ_RE.match(text).group(1)


def within_object_auroc(score, gold, objs):
    vals, w = [], []
    for o in np.unique(objs):
        m = objs == o
        if gold[m].any() and (~gold[m]).any():
            vals.append(P.auroc(score[m], gold[m])); w.append(m.sum())
    return float(np.average(vals, weights=w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--C", type=float, default=1e-4)
    a = ap.parse_args()
    dev = {c: P.Run("pope", a.model, c) for c in all_conditions()}
    hl, L = dev["clean_s0"].meta["hidden_layers"], dev["clean_s0"].meta["n_layers"]
    li = [hl.index(min(hl, key=lambda l: abs(l - f * L))) for f in (0.5, 0.75)]
    clean = dev["clean_s0"]
    teach = {k: clean.rows[i]["m"] > 0 for k, i in clean.index.items()}

    def feats(r):
        return np.asarray(r.hidden()[:, li, :], dtype=np.float32).reshape(len(r.rows), -1)

    X, M, Y, O = [], [], [], []
    for r in dev.values():
        X.append(feats(r)); M.append(r.col("m"))
        Y += [teach[(x["image"], x["text"])] for x in r.rows]
        O += [obj(x["text"]) for x in r.rows]
    X, M, Y, O = np.concatenate(X), np.concatenate(M), np.array(Y), np.array(O)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    omean = {o: Z[O == o].mean(0) for o in np.unique(O)}
    Zc = Z - np.stack([omean[o] for o in O])
    print(f"DEV rows {len(Z)}", flush=True)

    fit = lambda F: LogisticRegression(C=a.C, max_iter=3000).fit(F, Y)
    models = {"teacher": fit(Z), "residual": fit(np.c_[M[:, None] * 100, Z]),
              "objcentred": fit(Zc), "residual+obj": fit(np.c_[M[:, None] * 100, Zc])}
    del X
    print("fitted", flush=True)

    tests = [("exdark", "clean_s0")] + [("pope_fresh", c) for c in ("clean_s0", "blur_s4", "noise_s4", "lowlight_s4", "motion_s4")]
    for tset, cond in tests:
        r = P.Run(tset, a.model, cond)
        Zt = (feats(r) - mu) / sd
        Ot = np.array([obj(x["text"]) for x in r.rows])
        Zct = Zt - np.stack([omean.get(o, np.zeros(Zt.shape[1], np.float32)) for o in Ot])
        m = r.col("m")
        S = {"base": m, "teacher": models["teacher"].decision_function(Zt),
             "residual": models["residual"].decision_function(np.c_[m[:, None] * 100, Zt]),
             "objcentred": models["objcentred"].decision_function(Zct),
             "residual+obj": models["residual+obj"].decision_function(np.c_[m[:, None] * 100, Zct])}
        out = []
        for name, s in S.items():
            pooled, within = [], []
            for split in P.SPLITS:
                idx, gold, _ = r.split_index(split)
                pooled.append(P.auroc(s[idx], gold)); within.append(within_object_auroc(s[idx], gold, Ot[idx]))
            out.append(f"{name} {np.mean(pooled):.3f}/{np.mean(within):.3f}")
        print(f"{tset}/{cond} (pooled/within-object AUROC): " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
