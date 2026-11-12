# Probe × views × DAE

This experiment combines three results that each fix part of the problem.

| Source | What it adds | Noise s4 AUROC | Blur s4 AUROC |
|---|---|---|---|
| ViewLayer | test-time views (orig, blur σ 1/2/3, h-flip), read at layer 20 | .817 | .782 |
| ViewLayer+DAE | the `dae_v1`-restored image as a sixth view | .833 | .782 |
| DAE alone (Fix B, B0) | baseline on the restored image | .836 | – |
| Clean-teacher probe, 1 pass (`docs/probe_results.md`) | a linear readout fit to the model's own clean-image answers | .801 | .830 |
| Clean-teacher probe × 5 ViewLayer views (pilot, §7) | probe readout of every view, averaged | .852 | .832 |

The probe-results diagnosis explains why the parts should add up:

- **Blur** is a read-out problem: the evidence is in the hidden state but the Yes/No unembedding mis-reads it. The probe fixes this.
- **Noise** is an input problem: the evidence is lost before the decoder. The views fix this, and the restored image is the strongest single view we have on noise.

**The method.** Score each question as `mean_{v ∈ V} probe(h_v)`, where `h_v` is the last-token hidden state for view `v` and `V` is a set of the six views (orig, blur1–3, hflip, dae). The probe replaces ViewLayer's fixed layer-20 Yes−No margin. Nothing else changes.

## Protocol

The protocol is unchanged from Table 4.

- **Split:** 100 DEV / 400 TEST images, the same split as every other method.
- **Probes:**
  - Fit on the **original view** of the DEV images only, across all nine conditions.
  - Layers and C are the earlier DEV choices: gold L24 with C = 1e-3, and clean-teacher L16+24 with C = 1e-4. They are not re-tuned here.
  - The same probe scores every view.
- **View set:** chosen on DEV from all 63 non-empty subsets, as one global choice.
  - Selection uses **out-of-fold** probe scores from 5 image-grouped folds, so a DEV image is always scored by a probe that never saw it.
  - Without this, selection would favour the original view, which the probe was trained on.
- **Threshold:** one global DEV F1 threshold, from the same out-of-fold scores.
- **TEST metrics:**
  - AUROC with significance against the baseline, and F1 against the matched-threshold baseline.
  - A paired bootstrap against **ViewLayer+DAE** on identical questions.
- **Reference rows:** ViewLayer and ViewLayer+DAE are recomputed from the same logs.

## Running it

Run this on the machine that wrote the existing `layers_<cond>/` logs (Akshay's RTX 4500). The probes are fit on those logs, and switching machines changes the fp16 numerics.

1. Copy the DAE weights from ASPIRE to `$VCD_CHECKPOINT_ROOT/dae/dae_v1/best.pt`:
   ```bash
   scp c230202@aspire2antu.nscc.sg:/scratch/users/ntu/c230202/checkpoints/dae/dae_v1/best.pt ...
   ```
2. Run the whole queue. It is resumable: the smoke test comes first, then the GPU runs, then the analysis.
   ```bash
   VCD_CHECKPOINT_ROOT=... bash scripts/queue_probe_views_dae.sh
   ```
   The steps inside are:
   ```bash
   python -m scripts.run_layers --views dae                       # 9 runs
   python -m scripts.run_layers --views blur1 blur2 blur3 hflip   # 28 new runs; blur_s4/noise_s4 already done
   python -m scripts.analyze_probe_views_dae                      # CPU, minutes
   ```

**Cost and disk.** That is 37 single-pass runs. At the pilot's rate (about 13 min per run), this is roughly 8 GPU hours. The hidden states take about 0.35 GB per run, so about 13 GB in total.

**Partial results.** The analysis runs on whatever conditions already have all six views and marks the table as partial. Once the DAE view exists for blur s4 and noise s4, the 6-cell pilot can be extended with the DAE view after about 30 minutes of GPU time:
```bash
python -m scripts.run_layers --conditions blur_s4 noise_s4 --views dae
```

**Outputs** go to `results/`:
- `probe_views_dae_table.md` (the Table 4-style table plus the head-to-head)
- `probe_views_dae_cells.csv`
- `probe_views_dae_dev_sweep.csv`

**Tests** (CPU, fake logs):
```bash
python -m pytest scripts/tests -q
```

## What to look for

- **Does the DAE view add to probe × views on noise?** Compare "× ViewLayer+DAE views" against "× ViewLayer views". The pilot without the DAE view reached .852 at noise s4.
- **Is the clean-teacher version close to the gold one?** If it is, the headline method needs no human labels.
- **Does the DEV-chosen view set beat ViewLayer+DAE in most cells?** The cheap subsets are especially interesting, for example orig + dae: 2 passes against ViewLayer+DAE's 6.

## Caveats

- **Not training-free.** The probes are a small fitted head. The clean-teacher probe needs clean/degraded pairs at training time, but no labels.
- **Distribution shift.** The probes are fit on original-view hidden states and applied to blurred, flipped and restored views. The pilot shows this works, but fitting on all views is an untested alternative.
- **Optimistic intervals.** The bootstrap resamples questions, and questions within an image are correlated. As in the report, this applies equally to every method.
