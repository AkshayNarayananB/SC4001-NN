"""Probe x views: apply the (unchanged) gold and clean-teacher probes to ViewLayer's 5 views and average the scores.
Probes are fit exactly as in analyze_layers (DEV, original view, all 9 conditions). TEST on blur s4 / noise s4."""
import json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.analyze_layers import CONDITIONS, HIDDEN_LAYERS, HIDDEN_DIM, auroc, dev_images, hidden_memmap, layer_margin, load_cells, to_arrays, FINAL
from scripts.run_layers import run_dir_name
from scripts.run_pope import OUTPUT_ROOT
from src.paths import POPE_SPLITS

VIEWS = ("blur1", "blur2", "blur3", "hflip")
cells = load_cells(); dev_set = dev_images(cells)

def hid(kind, level, view=None):
    p = OUTPUT_ROOT / run_dir_name(kind, level, view) / "_hidden.f16"
    return np.memmap(p, dtype=np.float16, mode="r").reshape(-1, len(HIDDEN_LAYERS), HIDDEN_DIM)

def feats(H, idx, layers):
    return np.concatenate([np.asarray(H[idx][:, HIDDEN_LAYERS.index(l)], np.float32) for l in layers], 1)

def view_rows(kind, level, split, view, keep):
    with open(OUTPUT_ROOT / run_dir_name(kind, level, view) / f"{split}.jsonl") as f:
        df = pd.DataFrame(json.loads(l) for l in f).sort_values("question_id").reset_index(drop=True)
    m = np.array(df["layer_yes"].tolist())[:, 19] - np.array(df["layer_no"].tolist())[:, 19]
    return df["hidx"].to_numpy()[keep], m[keep]

PROBES = {"gold L24": ((24,), 1e-3, "gold"), "teacher L16+24": ((16, 24), 1e-4, "teacher")}
with threadpool_limits(8):
    fitted = {}
    for name, (layers, C, target) in PROBES.items():
        X, y = [], []
        for kind, level in CONDITIONS:
            seen = set(); H = hidden_memmap(kind, level)
            for split in POPE_SPLITS:
                df = cells[(kind, level, split)]; a = to_arrays(df)
                teach = layer_margin(to_arrays(cells[("clean", 0, split)]), FINAL)
                sel = [i for i in np.where(df["image"].isin(dev_set))[0] if not (a["hidx"][i] in seen or seen.add(a["hidx"][i]))]
                X.append(feats(H, a["hidx"][sel], layers)); y.append(a["gold"][sel] if target == "gold" else teach[sel] > 0)
        X = np.concatenate(X); y = np.concatenate(y)
        sc = StandardScaler().fit(X); lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X), y)
        fitted[name] = (layers, sc, lr); print(f"fit {name}: {len(y)} rows", flush=True)

    rows = []
    for kind, level in [("blur", 4), ("noise", 4)]:
        for split in POPE_SPLITS:
            df = cells[(kind, level, split)]; keep = ~df["image"].isin(dev_set).to_numpy(); a = to_arrays(df[keep]); g = a["gold"]
            views = [view_rows(kind, level, split, v, keep) for v in VIEWS]
            vl = np.mean([layer_margin(a, 20)] + [m for _, m in views], 0)
            r = dict(cond=f"{kind} s{level}", split=split, base=auroc(layer_margin(a, FINAL), g), viewlayer=auroc(vl, g))
            for name, (layers, sc, lr) in fitted.items():
                s = [lr.decision_function(sc.transform(feats(hidden_memmap(kind, level), a["hidx"], layers)))]
                s += [lr.decision_function(sc.transform(feats(hid(kind, level, v), h, layers))) for v, (h, _) in zip(VIEWS, views)]
                r[f"{name} (1 view)"] = auroc(s[0], g)
                r[f"{name} + blur1 (2 views)"] = auroc(np.mean(s[:2], 0), g)
                r[f"{name} x 5 views"] = auroc(np.mean(s, 0), g)
                # bootstrap vs ViewLayer
                rng = np.random.default_rng(0); idx = rng.integers(0, len(g), (500, len(g)))
                d = np.array([auroc(np.mean(s, 0)[i], g[i]) - auroc(vl[i], g[i]) for i in idx])
                r[f"{name} x5 - VL CI"] = f"[{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]"
            rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(out.round(3).T.to_string())
    print(out.groupby("cond").mean(numeric_only=True).round(3).T.to_string())
