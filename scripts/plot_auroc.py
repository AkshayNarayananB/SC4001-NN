import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import rankdata

def auroc(score: np.ndarray, gold: np.ndarray) -> float:
    """Mann-Whitney AUROC of `score` for gold-yes vs gold-no."""
    r = rankdata(score)
    n_pos, n_neg = gold.sum(), (~gold).sum()
    return (r[gold].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def load_all_metrics(config_dirs=["configs/dola_eval", "configs/stage2", "configs/stage0"]):
    """Scans for all configs to load baseline, VCD, and DoLa runs together."""
    rows = []
    for config_dir in config_dirs:
        p = Path(config_dir)
        if not p.exists(): continue
        
        for path in p.glob("*.yaml"):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            
            method = cfg.get("method", "baseline")
            # Unify VCD ablated names if present
            if method in ["vcd_zero", "vcd_shuffle"]:
                method = "vcd"
                
            kind = cfg.get("kind", "clean")
            level = cfg.get("level", 0)
            name = cfg.get("name")
            splits = cfg.get("splits", ["random", "popular", "adversarial"])
            
            for split in splits:
                raw = Path("results/raw") / name / f"{split}.jsonl"
                if not raw.exists():
                    continue
                
                with open(raw) as f:
                    for line in f:
                        data = json.loads(line)
                        if "logit_yes" in data and "logit_no" in data and "label" in data:
                            rows.append({
                                "method": method,
                                "kind": kind,
                                "level": level,
                                "question_id": data["question_id"],
                                "label": data["label"],
                                "margin": data["logit_yes"] - data["logit_no"]
                            })
    return pd.DataFrame(rows)

def main():
    print("Loading records from configs...")
    df = load_all_metrics()
    if df.empty:
        print("No valid JSONL records found.")
        return

    print("Calculating AUROC across all conditions...")
    results = []
    # Group by condition and calculate pooled AUROC across splits
    for (method, kind, level), cell in df.groupby(["method", "kind", "level"]):
        gold = (cell["label"] == "yes").to_numpy()
        m = cell["margin"].to_numpy(float)
        auc = auroc(m, gold)
        results.append({"method": method, "kind": kind, "level": level, "auroc": auc})
        
    res_df = pd.DataFrame(results)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Styling mappings
    colors = {"baseline": "#6c757d", "vcd": "#d90429", "dola_anchor": "#0077b6"}
    labels = {"baseline": "Baseline", "vcd": "Standard VCD", "dola_anchor": "Min Entropy Router (Ours)"}
    markers = {"baseline": "o", "vcd": "s", "dola_anchor": "^"}

    for i, kind in enumerate(["blur", "noise"]):
        ax = axes[i]
        # Pull the specific degradation kind, and inject the clean data as level 0
        kind_df = res_df[(res_df["kind"] == kind) | (res_df["kind"] == "clean")].copy()
        kind_df.loc[kind_df["kind"] == "clean", "level"] = 0
        
        for method in kind_df["method"].unique():
            m_df = kind_df[kind_df["method"] == method].sort_values("level")
            if not m_df.empty:
                ax.plot(m_df["level"], m_df["auroc"], 
                        marker=markers.get(method, "o"), 
                        color=colors.get(method, "black"), 
                        linewidth=2.5, 
                        markersize=9,
                        label=labels.get(method, method))
        
        ax.set_title(f"Visual Anchor Preservation on {kind.capitalize()}", fontsize=14, pad=15)
        ax.set_xlabel("Degradation Severity (0 = Clean, 5 = Severe)", fontsize=12)
        if i == 0:
            ax.set_ylabel("AUROC (Discriminative Power)", fontsize=12)
        
        ax.set_xticks(range(6))
        ax.grid(True, linestyle="--", alpha=0.6)
        
        # Only put the legend on the second subplot so it doesn't clutter
        if i == 1:
            ax.legend(fontsize=12, loc="lower left")

    plt.suptitle("Minimum Entropy Routing vs. Baseline Hallucination on POPE", fontsize=16, y=1.05, fontweight="bold")
    plt.tight_layout()
    
    out_path = Path("results/auroc_degradation_plot.png")
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nMaster figure saved successfully to {out_path}")

if __name__ == "__main__":
    main()