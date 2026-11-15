# Does Visual Contrastive Decoding Still Help When the Image Is Already Degraded?

SC4001 Neural Networks and Deep Learning, Group Project (Topic E), Group 115.

We test Visual Contrastive Decoding (VCD) on LLaVA-1.5-7B with POPE, using images degraded by Gaussian blur and by Gaussian noise (four levels each).

- **VCD recalibrates; it does not improve grounding.** Its F1 gain grows with severity, from +1.5 to +18.9 points. A baseline whose Yes/No threshold is shifted to VCD's yes-ratio reaches the same F1, and VCD's AUROC is unchanged.
- **Variants that subtract something do no better.** This covers zero/shuffle hooks, phase/mean nulls, RITUAL, DeCo and entropy layer routing: all stay within |ΔAUROC| ≤ 0.005.
- **What helps is adding evidence and reading it better:**
  - **ViewLayer** averages the Yes−No margin over blurred and flipped views, read out at decoder layer 20.
  - The **DAE-restored view** adds a denoised copy of the image.
  - The **clean-teacher linear probe** is trained on the model's own clean-image answers.
- **Best result:** the clean-teacher probe over three DEV-selected views raises TEST AUROC from .750 to .833 at the strongest blur, and from .737 to .868 at the strongest noise.

| Condition | Baseline | VCD | ViewLayer | ViewLayer+DAE | CT probe (1 pass) | CT probe × DEV views |
|---|---|---|---|---|---|---|
| Clean | .940 | .935 | .945 | .945 | .949 | .950 |
| Blur s4 | .750 | .749 | .782 | .782 | .830 | .833 |
| Noise s4 | .737 | .737 | .817 | .833 | .801 | .868 |

All numbers are TEST AUROC, the mean of the three POPE splits. The full tables are in `results/`; see `results/README.md`.

## Layout

```
src/                    shared library
  paths.py              data/output/checkpoint roots (env: VCD_DATA_ROOT, VCD_OUTPUT_ROOT, VCD_CHECKPOINT_ROOT)
  model.py              LLaVA-1.5 loading, prompt, encoding
  degradations.py       blur / noise levels (LEVELS is the single source of truth)
  vcd.py                diffusion noise, contrastive logits, plausibility mask
  decode.py             shared generation loop (baseline, VCD, entropy routing)
  hooks.py              zero / shuffle visual-contrast hooks
  eval_pope.py          POPE metrics (matches the VCD reference)
  dae.py                7.2M-parameter residual U-Net denoiser (Fix B)
scripts/                data preparation and ASPIRE infrastructure
  download_pope.py, download_coco_train.py, make_degraded.py, contact_sheet.py
  setup_aspire.sh, submit.sh, pbs/
experiments/
  protocol.py           shared evaluation: DEV/TEST split, decision-preserving AUROC, matched F1, bootstrap
  vcd/                  §4.1–4.2  baseline vs VCD (Stage 0–2); online entropy-routing runs
  dae/                  §4.4      Fix B: train the DAE, restore POPE images, score B0/B1/B2
  nulls/                §4.3      phase / mean null contrast images
  ritual/               §4.3      RITUAL re-implementation (+ per-transform ablation)
  deco/                 §4.3      DeCo port (checked against the official code)
  viewlayer/            §3.5/4.4  ViewLayer
  viewlayer_dae/        §4.4      ViewLayer + DAE-restored view
  readouts/             §3.6/4.5  per-layer logging, routing, hooks, patch/token readouts, probes, probe × views
configs/                run configs for experiments.vcd.run_pope (stage0, calib, stage1, stage2, fixb)
results/                metric tables, one folder per experiment (see results/README.md)
checkpoints/dae/dae_v1/ trained DAE weights used by Fix B and the probe × views × DAE analysis
docs/                   running findings log and per-analysis notes
logs/                   stdout of the readout / probe runs
```

Each method package (`nulls`, `ritual`, `deco`, `viewlayer`, `viewlayer_dae`) contains:
- `run_*.py`, the GPU runner;
- `run_*.pbs`, a PBS array job over the nine conditions;
- `analyze.py`, which selects settings on DEV, scores on TEST, and writes to `results/<method>/` and the README's Results section;
- `tests/`, which run on CPU without model weights;
- a `README.md` with the method and its full results.

## Setup

- Python 3.11. Pinned dependencies are in `requirements.txt` (torch 2.5.1, transformers 4.46.3).
  - Two CPU tests that compare against the HF model internals (`deco`, `viewlayer`) need transformers 4.46.x.
- On NSCC ASPIRE 2A, `bash scripts/setup_aspire.sh` builds the conda env.
- Download the weights once on a login node with `huggingface-cli download llava-hf/llava-1.5-7b-hf`. Jobs then run with `HF_HUB_OFFLINE=1`.
- Set the data and output roots:
  ```bash
  export VCD_DATA_ROOT=/path/to/data          # default ./data
  export VCD_OUTPUT_ROOT=/path/to/raw_outputs  # default ./results/raw (per-question JSONL; not in git)
  ```

Run every entry point from the repo root as a module, e.g. `python -m experiments.viewlayer.analyze`.

## Reproducing the report

| Report | Run | Analyse → output |
|---|---|---|
| Data, Fig. 1 | `python -m scripts.download_pope` (login node), then `python -m scripts.make_degraded`, then `python -m scripts.contact_sheet` | `$VCD_DATA_ROOT/pope/`, `results/contact_sheet.png` |
| §4.1 VCD reproduction (Table A.2) | `python -m experiments.vcd.make_configs stage0`, then `bash scripts/submit.sh configs/stage0/` | `python -m experiments.vcd.summarize --prefix stage0` |
| §4.2 Baseline vs VCD (Table 2) | `python -m experiments.vcd.make_configs stage2`, then `bash scripts/submit.sh configs/stage2/` | `python -m experiments.vcd.analyze --stage stage2` → `results/vcd/` |
| Phase/mean nulls | `qsub experiments/nulls/run_nulls.pbs` | `python -m experiments.nulls.analyze` → `results/nulls/` |
| RITUAL (+ App. E) | `qsub experiments/ritual/run_ritual.pbs` | `python -m experiments.ritual.analyze` → `results/ritual/` |
| DeCo | `qsub experiments/deco/run_deco.pbs` | `python -m experiments.deco.analyze` → `results/deco/` |
| ViewLayer | `qsub experiments/viewlayer/run_viewlayer.pbs` | `python -m experiments.viewlayer.analyze` → `results/viewlayer/` |
| ViewLayer+DAE | `qsub experiments/viewlayer_dae/run_viewlayer_dae.pbs` | `python -m experiments.viewlayer_dae.analyze` → `results/viewlayer_dae/` |
| Fix B (DAE) | `python -m scripts.download_coco_train`; `qsub -v NAME=dae_v1 scripts/pbs/train_dae.pbs`; `qsub scripts/pbs/make_restored.pbs`; `python -m experiments.vcd.make_configs fixb` and `bash scripts/submit.sh configs/fixb/...` | `python -m experiments.dae.analyze_fixb --test-only` → `results/dae/` |
| Hooks, layer routing, probes (§4.5, Tables 3–4, A.6) | `python -m experiments.readouts.run_layers`; `python -m experiments.readouts.run_contrast`; `bash experiments/readouts/queue_probe_views_dae.sh` | `python -m experiments.readouts.analyze_layers`, `analyze_contrast`, `analyze_selective`, `analyze_probe_views_dae` → `results/readouts/` |
| Fig. 4 | | `python experiments/readouts/plot_probe_severity.py` → `results/readouts/fig4_probe_severity.{pdf,png}` |

For a 30-question smoke test of a method, add `-J 4-8:4 -v LIMIT=30` to `qsub`.

Run the CPU tests with:

```bash
python -m pytest experiments -q
```

### Old → new paths

The report's Appendix F was written before this reorganisation. Its paths map as follows:

| Report (Appendix F) | Now |
|---|---|
| `model_fix_2/` | `experiments/nulls/` |
| `RITUAL/`, `DeCo/`, `ViewLayer/`, `ViewLayerDAE/` | `experiments/ritual/`, `experiments/deco/`, `experiments/viewlayer/`, `experiments/viewlayer_dae/` |
| `fix_eval/common.py` | `experiments/protocol.py` |
| `src/fixes/dae.py` | `src/dae.py` |
| `scripts.run_pope`, `make_configs`, `analyze`, `summarize` | `experiments.vcd.*` |
| `scripts.train_dae`, `make_restored`, `analyze_fixb` | `experiments.dae.*` |
| `scripts.run_layers`, `run_contrast`, `analyze_layers`, `analyze_contrast`, `analyze_selective`, `analyze_probe_views_dae`, `scripts/queue_probe_views_dae.sh` | `experiments.readouts.*` |
| `Report/scripts/fig_probe_severity.py` | `experiments/readouts/plot_probe_severity.py` |
| `results/<method>_*.csv` | `results/<method>/<method>_*.csv` |

The data-prep scripts (`scripts.download_pope`, `make_degraded`, `contact_sheet`, `download_coco_train`) and `scripts/submit.sh` did not move.
