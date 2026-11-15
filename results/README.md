# Results

These are small tables that are tracked in git. The raw per-question JSONL files stay in `$VCD_OUTPUT_ROOT` on ASPIRE scratch and are not in git.

Every number in the report is a TEST-image number (400 images; 100 DEV images held out), unless the table says otherwise.

| Folder | Written by | Contents | Report |
|---|---|---|---|
| `metrics/` | `experiments.vcd.run_pope` | One CSV per run with acc / prec / recall / F1 / yes-ratio per split (see the prefixes below). `archive/levels_v1/` holds the first, too-weak degradation ladder. | Table A.2, Stage 1 calibration |
| `vcd/` | `experiments.vcd.analyze --stage stage2` | Baseline vs VCD: `stage2_comparison` (F1, yes-ratio), `stage2_margins` (AUROC, matched-threshold F1), `stage2_diagnostics` (JS divergence, contrast gap) | §4.2, Table 2 |
| `dae/` | `experiments.dae.analyze_fixb` | Fix B0 / B1 / B2 on noise. `_test` = the 400 TEST images only. | §4.4, Table 3 |
| `nulls/` | `experiments.nulls.analyze`, `show_nulls` | Phase / mean null: DEV α sweep, comparison, example images | Table 3, Fig. A.2 |
| `ritual/` | `experiments.ritual.analyze` | RITUAL: DEV α sweep, comparison, per-transform ablation | Table 3, App. E |
| `deco/` | `experiments.deco.analyze` | DeCo: DEV α sweep, comparison, per-layer AUROC | Table 3 |
| `viewlayer/` | `experiments.viewlayer.analyze` | ViewLayer: DEV sweep (views × layer), comparison | §4.4, Tables 3, A.3–A.5 |
| `viewlayer_dae/` | `experiments.viewlayer_dae.analyze` | ViewLayer + DAE view, with the VCD-contrast ablation | §4.4, Table 3 |
| `readouts/` | `experiments.readouts.analyze_*`, `compare_viewlayer`, `plot_probe_severity` | See the list below the table. | §4.5, Tables 3, 4, A.6, Fig. 3, Fig. 4 |
| `routing/` | early entropy-routing analysis (scripts removed in `3c54c3c`) | Online DoLa / entropy-routing diagnostics and plots. Superseded by the offline rescoring in `readouts/layers_*`. | (background for §4.6) |
| `contact_sheet.png` | `scripts.contact_sheet` | The nine conditions at the model's 336 px input size | Fig. 1 |

What `readouts/` holds:
- **Layer readouts, entropy routing and probes:** `layers_*`. `layers_scores.npz` holds the per-question scores reused by `analyze_selective` and `compare_viewlayer`.
- **Zero/shuffle hooks:** `contrast_*`.
- **Patch logit-lens:** `patches_*`.
- **Selective answering:** `selective_*`.
- **Clean-teacher probe vs ViewLayer:** `compare_viewlayer*`.
- **Probe × views × DAE:** `probe_views_dae_*`, plus `pilot_dae_s4/`, the disclosed s4 pilot.
- **Figure 4:** `fig4_probe_severity.*`.

Prefixes in `metrics/`:
- `stage0_`: clean reproduction.
- `calib_`: level calibration.
- `stage1_`: degradation pilot.
- `stage2_`: baseline/VCD grid.
- `fixb_`: DAE restoration.
- `dola_`: online entropy-routing runs.
