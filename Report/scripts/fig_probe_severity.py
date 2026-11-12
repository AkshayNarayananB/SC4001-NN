"""Figure: TEST AUROC vs severity for the clean-teacher probe readouts against ViewLayer.
    python Report/scripts/fig_probe_severity.py   (reads results/probe_views_dae_cells.csv)
Hue = method family (blue: ViewLayer's layer-20 readout, green: clean-teacher probe readout);
line style and marker = variant, so identity never rests on colour alone."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
d = pd.read_csv(ROOT / "results/probe_views_dae_cells.csv")
base = d[d.method == "ViewLayer*"].groupby(["kind", "level"]).auroc_base.mean()

SERIES = [  # label, method key in the CSV (None = baseline), colour, style, marker, width
    ("Baseline (layer 32)", None, "#8a8986", (0, (4, 2)), "o", 1.6),
    ("ViewLayer (5 passes)", "ViewLayer*", "#2a78d6", "-", "o", 2.0),
    ("CT probe, 1 pass", "Clean-teacher probe, 1 view*", "#008300", (0, (4, 2)), "^", 1.6),
    ("CT probe × ViewLayer views (5)", "Clean-teacher probe, × ViewLayer views*", "#008300", "-", "D", 2.0),
    ("CT probe × ViewLayer+DAE views (6)", "Clean-teacher probe, × ViewLayer+DAE views*", "#008300", (0, (1, 1.2)), "s", 2.0),
]
for f in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.linewidth": 0.8,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True)
xs = range(5)
for ax, kind, title in zip(axes, ("blur", "noise"), ("Gaussian blur", "Gaussian noise")):
    keys = [("clean", 0)] + [(kind, l) for l in (1, 2, 3, 4)]
    for label, m, col, ls, mk, lw in SERIES:
        if m is None:
            y = [base[k] for k in keys]
        else:
            g = d[d.method == m].groupby(["kind", "level"]).auroc.mean()
            y = [g[k] for k in keys]
        ax.plot(xs, y, color=col, ls=ls, marker=mk, ms=4.2, lw=lw, label=label,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    ax.set_xticks(list(xs), ["clean", "s1", "s2", "s3", "s4"])
    ax.set_xlabel("Severity level")
    ax.grid(True, color="#e6e5e1", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("AUROC (TEST, mean of 3 splits)")
axes[0].set_ylim(0.725, 0.96)
axes[1].legend(loc="lower left", frameon=False, fontsize=7.6, handlelength=2.6, labelspacing=0.3)
fig.tight_layout(w_pad=1.2)
out = ROOT / "Report/latex/figures/fig4_probe_severity"
fig.savefig(f"{out}.pdf")
fig.savefig(f"{out}.png", dpi=250)
print("wrote", out)
