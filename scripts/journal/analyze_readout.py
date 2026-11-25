"""Make-or-break test of the readout method: probes fit on DEV (POPE's 500 images, all logged conditions),
evaluated without any tuning on held-out TEST images (pope_fresh) and on real low-light photos (ExDark).

    python -m scripts.journal.analyze_readout --model llava-1.5-7b [--boot 500]

Scores per question (higher = "yes"):
  base       the model's first-token Yes-No margin, answered at threshold 0
  thr_const  the same margin with ONE bias b* tuned on DEV (the strongest pure-recalibration baseline,
             cf. arXiv 2607.25196): same AUROC as base, best possible F1 for a global shift
  layer_mid  logit-lens Yes-No margin at the layer nearest 62.5% depth (layer 20 of 32 for LLaVA), DEV threshold
  vcd        VCD (alpha=1, beta=0.1, T=999), decision-preserving AUROC, threshold 0 (as published)
  gold       logistic probe on standardised last-token hidden states (layers at 50% + 75% depth), POPE labels
  teacher    the same probe trained on the model's OWN clean-image answers (1[m_clean > 0]): no human labels
Probe regularisation: 5-fold CV on DEV grouped by image. Thresholds for layer_mid / probes / thr_const: one global
value maximising mean DEV F1 over DEV cells. Reported per test cell: AUROC (+ image-clustered 95% CI of the gain
over base), F1 at the DEV threshold, F1 gain over thr_const with CI, yes-ratio, AURC (selective answering).
ExDark is additionally split into luminance quartiles. Writes results/journal/readout_<model>.{csv,md}.
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from scripts.journal import protocol as P
from scripts.journal.run_readout import all_conditions
from src.paths import DATA_ROOT, REPO_ROOT

OUT = REPO_ROOT / "results" / "journal"
CS = (1e-4, 3e-4, 1e-3, 3e-3)


def labels(pope_set):
    lab = {}
    for s in P.SPLITS:
        for q in P.questions(pope_set, s):
            lab[(q["image"], q["text"])] = q["label"] == "yes"
    return lab


def load_runs(pope_set, model, conds):
    runs = {}
    for c in conds:
        try:
            runs[c] = P.Run(pope_set, model, c)
        except FileNotFoundError:
            pass
    return runs


def feats(run, idx_layers):
    H = run.hidden()
    return np.asarray(H[:, idx_layers, :], dtype=np.float32).reshape(len(run.rows), -1)


def aurc(score, gold, thr):
    """Area under the risk-coverage curve: abstain on the least confident |score - thr| first."""
    pred = score > thr
    err = (pred != gold).astype(float)
    order = np.argsort(-np.abs(score - thr), kind="stable")
    cum = np.cumsum(err[order]) / np.arange(1, len(err) + 1)
    return float(cum.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava-1.5-7b")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--test-sets", nargs="+", default=["pope_fresh", "exdark"])
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    # ---------- DEV ----------
    dev = load_runs("pope", a.model, all_conditions())
    assert "clean_s0" in dev, "DEV clean run needed for teacher labels"
    meta = dev["clean_s0"].meta
    L, hl = meta["n_layers"], meta["hidden_layers"]
    idx_layers = [hl.index(min(hl, key=lambda l: abs(l - f * L))) for f in (0.5, 0.75)]
    mid = round(0.625 * L) - 1                    # index into lm (layer 20 of 32 -> lm[19])
    lab = labels("pope")
    clean = dev["clean_s0"]
    teach = {k: clean.rows[i]["m"] > 0 for k, i in clean.index.items()}
    print(f"DEV conditions: {list(dev)}; probe layers {[hl[i] for i in idx_layers]}; mid layer {mid + 1}", flush=True)

    X, yg, yt, grp, cell_ix = [], [], [], [], []
    for c, r in dev.items():
        keys = [(x["image"], x["text"]) for x in r.rows]
        X.append(feats(r, idx_layers))
        yg += [lab[k] for k in keys]
        yt += [teach[k] for k in keys]
        grp += [k[0] for k in keys]
    X = np.concatenate(X)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    yg, yt, grp = np.array(yg), np.array(yt), np.array(grp)
    print(f"DEV rows {len(X)}; teacher agrees with gold on {np.mean(yg == yt):.3f}", flush=True)

    probes = {}
    gkf = GroupKFold(n_splits=5)
    for name, y in (("gold", yg), ("teacher", yt)):
        best = None
        for C in CS:
            oof = np.zeros(len(y))
            for tr, va in gkf.split(X, y, grp):
                oof[va] = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr]).decision_function(X[va])
            cv = P.auroc(oof, yg)                  # selection by gold AUROC on DEV (disclosed in the paper)
            print(f"  {name} C={C:g} CV AUROC(gold) {cv:.4f}", flush=True)
            if best is None or cv > best[0]:
                best = (cv, C)
        probes[name] = LogisticRegression(C=best[1], max_iter=2000).fit(X, y)
        print(f"{name}: C={best[1]:g}", flush=True)

    def scores(run):
        lm = run.layer_margins()
        Z = (feats(run, idx_layers) - mu) / sd
        s = {"base": run.col("m"), "layer_mid": lm[:, mid].astype(float),
             "gold": probes["gold"].decision_function(Z), "teacher": probes["teacher"].decision_function(Z)}
        if "cd_yes" in run.rows[0]:
            s["vcd"] = P.vcd_margin(run)
        return s

    # DEV thresholds: one global value per score over all DEV cells (cells = condition x split)
    dev_s = {c: scores(r) for c, r in dev.items()}
    thr = {"base": 0.0, "vcd": 0.0}
    for name in ("base_tuned", "layer_mid", "gold", "teacher"):
        src = "base" if name == "base_tuned" else name
        cells_s, cells_g = [], []
        for c, r in dev.items():
            for s in P.SPLITS:
                idx, gold, _ = r.split_index(s)
                cells_s.append(dev_s[c][src][idx]); cells_g.append(gold)
        thr[name] = P.best_threshold(cells_s, cells_g)
    print("DEV thresholds:", {k: round(v, 3) for k, v in thr.items()}, flush=True)

    # ---------- TEST ----------
    rows = []
    lum = pd.read_csv(DATA_ROOT / "exdark" / "luminance.csv").set_index("image")["mean_srgb"] \
        if (DATA_ROOT / "exdark" / "luminance.csv").exists() else None
    for tset in a.test_sets:
        conds = ["clean_s0"] if tset == "exdark" else all_conditions()
        for c, r in load_runs(tset, a.model, conds).items():
            sc = scores(r)
            for split in P.SPLITS:
                idx, gold, imgs = r.split_index(split)
                groups = [("all", np.ones(len(idx), bool))]
                if tset == "exdark" and lum is not None:
                    q = pd.qcut(lum.reindex(imgs).to_numpy(), 4, labels=False)
                    groups += [(f"lum_q{i + 1}", q == i) for i in range(4)]
                for gname, msk in groups:
                    ii, g, im = idx[msk], gold[msk], imgs[msk]
                    base = sc["base"][ii]
                    rec = dict(model=a.model, set=tset, condition=c, split=split, group=gname, n=len(ii))
                    f1_tc = P.f1(base > thr["base_tuned"], g)
                    rec["f1_base0"], rec["f1_thr_const"] = P.f1(base > 0, g), f1_tc
                    rec["auroc_base"] = P.auroc(base, g)
                    rec["aurc_thr_const"] = aurc(base, g, thr["base_tuned"])
                    for name in ("layer_mid", "gold", "teacher", "vcd"):
                        if name not in sc:
                            continue
                        s = sc[name][ii]
                        sr = P.decision_score(s, base) if name == "vcd" else s
                        t = thr[name]
                        rec[f"auroc_{name}"] = P.auroc(sr, g)
                        rec[f"f1_{name}"] = P.f1(s > t, g)
                        rec[f"yes_{name}"] = float((s > t).mean())
                        if name != "vcd":
                            rec[f"aurc_{name}"] = aurc(s, g, t)
                        if gname == "all" and name in ("teacher", "gold"):
                            da, df = [], []
                            for bi in P.cluster_boot(im, a.boot, rng):
                                gg = g[bi]
                                da.append(P.auroc(sr[bi], gg) - P.auroc(base[bi], gg))
                                df.append(P.f1(s[bi] > t, gg) - P.f1(base[bi] > thr["base_tuned"], gg))
                            rec[f"d_auroc_{name}_lo"], rec[f"d_auroc_{name}_hi"] = np.percentile(da, [2.5, 97.5])
                            rec[f"d_f1_{name}_lo"], rec[f"d_f1_{name}_hi"] = np.percentile(df, [2.5, 97.5])
                    rows.append(rec)
            print(f"{tset}/{c}: base {rows[-1]['auroc_base']:.3f} teacher {rows[-1].get('auroc_teacher', float('nan')):.3f}",
                  flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"readout_{a.model}.csv", index=False)

    md = [f"# Readout make-or-break: {a.model}", "",
          f"DEV conditions used for fitting: {', '.join(dev)}. Probe layers {[hl[i] for i in idx_layers]}, "
          f"mid layer {mid + 1}. Thresholds fit on DEV: " + ", ".join(f"{k} {v:.2f}" for k, v in thr.items()), ""]
    for tset in a.test_sets:
        d = df[(df.set == tset) & (df.group == "all")]
        if d.empty:
            continue
        cols = [c for c in ["auroc_base", "auroc_vcd", "auroc_layer_mid", "auroc_teacher", "auroc_gold",
                            "f1_base0", "f1_thr_const", "f1_vcd", "f1_layer_mid", "f1_teacher", "f1_gold",
                            "aurc_thr_const", "aurc_teacher"] if c in d]
        t = d.groupby("condition", sort=False)[cols].mean()
        t[[c for c in cols if c.startswith("f1")]] *= 100
        sig = d.groupby("condition", sort=False).apply(lambda x: pd.Series({
            "teacher AUROC up/down": f"{(x.d_auroc_teacher_lo > 0).sum()}/{(x.d_auroc_teacher_hi < 0).sum()}",
            "teacher F1 vs thr_const up/down": f"{(x.d_f1_teacher_lo > 0).sum()}/{(x.d_f1_teacher_hi < 0).sum()}"}))
        md += [f"## {tset} (mean of 3 splits; F1 in %)", "", t.round(3).join(sig).to_markdown(), ""]
        if tset == "exdark":
            q = df[(df.set == "exdark") & (df.group != "all")].groupby("group")[
                [c for c in ["auroc_base", "auroc_teacher", "auroc_gold", "auroc_layer_mid"] if c in df]].mean()
            md += ["ExDark by luminance quartile (q1 = darkest):", "", q.round(3).to_markdown(), ""]
    (OUT / f"readout_{a.model}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
