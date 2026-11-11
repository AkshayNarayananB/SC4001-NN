# CLAUDE.md

Context for Claude Code (and humans) working in this repo.

## Project summary

Small research project (SC4001 Neural Networks). We study whether **Visual Contrastive Decoding (VCD)** still reduces object hallucination in vision-language models (VLMs) when the input image is **already degraded** (blurred, noisy, dark, compressed), and propose **two fixes** to the VCD decoding step.

### Background: VCD (Leng et al., CVPR 2024)

- Training-free; only changes how the next token is chosen at decode time.
- At each step, the model runs twice:
  - with the real image `v` → logits `l(y | v, x)`
  - with a heavily noised copy `v'` (forward-diffusion noise) → logits `l(y | v', x)`. The model can barely see anything here, so it falls back on language priors.
- Contrastive logits: `l_vcd = l(y|v,x) + α · (l(y|v,x) − l(y|v',x))` = `(1+α)·l(y|v,x) − α·l(y|v',x)`
- **Adaptive plausibility constraint:** only tokens with `p(y|v,x) ≥ β · max_w p(w|v,x)` are kept; the rest are masked to `-inf` before sampling.
- Paper defaults: `α = 1`, `β = 0.1`, noise step `T = 500` (of 1000 diffusion steps).
- Reference implementation: https://github.com/DAMO-NLP-SG/VCD

### Hypothesis

VCD assumes the "real" image is clean. If the input is already degraded, the real branch is already partly guessing from language priors. Both branches then produce similar logits, `l(y|v) − l(y|v')` shrinks, and VCD reduces to almost the baseline. That happens exactly when hallucination is most likely.

**Expected result of Phase 1:** baseline and VCD perform about the same on degraded images, and VCD's gain shrinks as degradation severity increases.

**Actual result (Stage 2, 2026-11-01): the F1 prediction is refuted, but the mechanism holds.** Details are in `docs/findings.md`, Stage 2.
- VCD's POPE F1 gain *grows* with severity: greedy +1.2 on clean, +19.5 at blur s4, +20.1 at noise s4.
- The gain is entirely a Yes/No threshold shift. VCD lowers AUROC in every cell, and a baseline thresholded to VCD's yes-ratio matches or beats VCD's F1.
- Degradation pushes the baseline toward "no", so VCD's roughly constant push toward "yes" is worth more as severity rises.
- The two branches do converge, as hypothesised: JS halves, and the discriminative part of the contrast halves.
- **Phase 2 fixes must therefore be judged on AUROC and against the matched-threshold baseline**, not on F1 alone. Those numbers are in `results/stage2_margins.csv`.

## Research plan

### Phase 1: Reproduce the failure mode
Compare **baseline decoding** and **VCD** on clean images and on several degradation types and severities.

Degradations (decided 2026-10-30): **Gaussian blur** and **additive Gaussian noise**, 4 levels each plus clean. Blur is what the project brief names. Noise is the closest match to VCD's own diffusion noise, so it's the most direct test of the hypothesis. Low light is an optional third type if time allows. JPEG and downsampling were dropped: downsampling behaves much like blur, and JPEG barely matters until the quality is extreme. Fewer types keeps the grid (types × levels × methods × splits) manageable.

- Levels live in `LEVELS` in `src/degradations.py`; that's the single source of truth.
  - Current levels (calibrated 2026-11-01): **blur σ 5/7/10/16, noise std 0.3/0.5/0.7/1.0**. These give greedy-baseline accuracy drops of about −3/−7/−14/−23 and −2/−12/−14/−20.
  - The first ladder (blur 2/4/8/16, noise .05/.1/.2/.4) was too weak: noise s1–s3 had no effect. See `docs/findings.md`, Stage 1.
  - Changing `LEVELS` means rebuilding the dataset (`scripts/pbs/make_degraded.pbs`) and archiving any results made under the old levels.
  - Runs record `level_param`, so stale run dirs fail loudly instead of being resumed.
- Degrade at the image's **native resolution**, before the model's image preprocessing, so the degraded image is the model's "real" input. LLaVA's resize to 336 px softens blur and averages out noise, so the levels are set stronger than they'd be at 336 px.
- Degradations are deterministic: noise is seeded by `crc32(image|type|level)`.
- VCD's own diffusion noise is applied on top of the degraded image at runtime, never saved to disk.

### Dataset

- POPE COCO: all three splits (random/popular/adversarial) use the **same 500 COCO val2014 images**, with 3000 questions per split, half of them "yes". Question files come from `AoiDragon/POPE` (`output/coco/coco_pope_{split}.json`, JSONL: `question_id, image, text, label`).
- Layout under `$VCD_DATA_ROOT` (on ASPIRE, `/scratch/users/ntu/c230202/data`; locally, `./data`):
  ```
  pope/questions/coco_pope_{split}.json
  pope/coco_val2014/COCO_val2014_*.jpg         # original downloads
  pope/images/clean/COCO_val2014_*.png         # lossless PNG for every version
  pope/images/{blur,noise}/s{1..4}/COCO_val2014_*.png
  pope/manifest.csv                            # image, type, level, param, seed, psnr, ssim
  ```
- **Built on ASPIRE:** 4500 PNGs plus `manifest.csv` in `/scratch/users/ntu/c230202/data/pope/`. This is the **canonical copy**; every experiment reads it.
  - First built 2026-10-30. Rebuilt with the calibrated levels 2026-11-01 (job `26074795`).
  - Per-level PSNR/SSIM: see `manifest.csv`, or the summary printed in `logs/make_degraded.log`.
  - The local `data/pope/` copy was refreshed on 2026-11-01 and checksum-verified against ASPIRE (all 5004 files match). The shareable zip is `data/pope_dataset_v2.zip`.
  - `data/pope.zip` (2026-10-30) and `data/pope_old_levels/` have the **old** levels; don't share them.
  - To refresh the local copy, stream a tar over ssh (`ssh aspire 'tar -C <pope> -cf - .' | tar -xf -`). macOS's rsync stalled on this 2 GB tree.
- **Sharing:** the dataset is **not** in git (`data/` is gitignored). The user shares it with teammates as a zip outside git. The code remote will be GitHub, which isn't set up yet.
- **Rebuilding elsewhere:** a rebuild is reproducible on the same machine, but not byte-identical across Pillow versions. The JPEG decoder differs between Pillow 11.0 and 11.1, so pixels can differ by ±1 on fewer than 0.005% of them. Don't mix locally built images into results.
- Pipeline:
  - `scripts/download_pope.py` downloads the question files and images. It needs internet, so run it on the login node.
  - `scripts/make_degraded.py` builds every version plus the manifest (CPU job: `scripts/pbs/make_degraded.pbs`).
  - `scripts/contact_sheet.py` makes a visual check grid at the model's 336 px input size.

### Stage 0: Reproduce VCD on clean images (gate for everything else)

The implementation must reproduce the paper's clean-image POPE gain before any degraded-image results mean anything.

- **Reference details** (from the DAMO-NLP-SG/VCD repo; checked 2026-11-01):
  - Prompt: llava_v1 template plus `" Please answer this question with one word."`
  - Noise is added to the normalized `pixel_values`, on a sigmoid β schedule, at `noise_step=500`.
  - Plausibility cutoff: `logits < log β + max logits`.
  - Decoding: plain sampling (T=1, no top-p/top-k), `max_new_tokens=1024` (we use 16, since answers are one word).
  - Scoring: gold "yes" is correct if `'yes' in answer`, gold "no" if `'no' in answer`.
  - The paper uses α=1, β=0.1 and averages 5 runs. "Regular" in its tables is **sampling**, not greedy. Note the repo's bash script passes β=0.2; we follow the paper's 0.1.
- **Verified locally:**
  - `src/vcd.py` noise is bit-identical to the reference given the same RNG stream.
  - The VCD logit step is identical.
  - `src/eval_pope.py` matches the reference `eval_pope.py` exactly.
  - The cached two-branch generation loop matches a no-cache recompute (tested with a tiny LLaVA).
- **Paper targets** (LLaVA-1.5, POPE MSCOCO, acc / F1):

  | split | Regular | VCD |
  |---|---|---|
  | random | 83.29 / 81.33 | 87.73 / 87.16 |
  | popular | 81.88 / 80.06 | 85.38 / 85.06 |
  | adversarial | 78.96 / 77.57 | 80.88 / 81.33 |

  Pass criterion: our sampling numbers land within ~1.5 points of these, and VCD beats baseline on all three splits.

- **Result (2026-11-01): PASSED with `noise_step=999`.** This is the VCD setting used for all later stages, and it's now the default in `scripts/run_pope.py`.
  - **Baseline sampling:** F1 81.19 / 80.20 / 77.81 (Δ vs paper −0.14 / +0.14 / +0.24). The greedy baseline (85.51 F1, random) matches the paper's Appendix B.4 (85.54).
  - **VCD, T=999, sampling:** F1 87.09 / 85.48 / 81.77 (Δ −0.07 / +0.42 / +0.44). VCD greedy on random gives 88.47 / 91.81 / 84.47 / 87.99, against the paper's 88.49 / 91.78 / 84.56 / 88.02.
  - **VCD, T=500** (the repo script's default): gains are only +2.3 / +2.3 / +1.8 F1 with sampling, and ≈0 or negative with greedy. This matches the reproduction complaints in VCD GitHub issues #24, #16 and #9. Issue #24 is where T=999 for POPE was pointed out.
  - **Greedy VCD (T=999):** helps on random (+2.5 F1) but barely on popular (+1.1) and not on adversarial (−0.1). Most of VCD's POPE gain comes from a recall/yes-ratio increase (yes-ratio 39% → 46–52%).
  - Metrics are in `results/metrics/stage0_*.csv`. View them with `python -m scripts.summarize --prefix stage0`.
- **Runs:** `python -m scripts.make_configs stage0` writes `configs/stage0/`:
  - baseline and VCD × sampling seeds 55/56/57
  - baseline and VCD × greedy
  - `stage0_smoke` (30 questions)

  Submit on ASPIRE with `bash scripts/submit.sh configs/stage0/<cfg>.yaml`.
- **Outputs:**
  - Per-question JSONL (answer, first-token Yes/No logits and log-probs for both branches, JS divergence) goes to `$VCD_OUTPUT_ROOT/<name>/` (on ASPIRE, `/scratch/users/ntu/c230202/results/raw`). Runs resume from it.
  - Per-split metrics go to `results/metrics/<name>.csv`.
- **Contrast-image strength:**
  - At T=500, VCD's noised image is still strongly correlated (≈0.86) with the input, about as noisy as our noise s2–s3. That likely explains why T=500 gives weak gains.
  - At T=999 (what we use), the correlation is ≈0.29, far heavier than any of our input degradations. The degraded-input hypothesis is therefore tested against a contrast image that is always much "blinder" than the input.
  - Contrast strength clearly matters (T=500 → T=999 roughly doubles VCD's gain), which supports Phase 2 fix #2 (degradation-aware negative).

### Phase 2: Propose and evaluate two fixes
Pick two from the candidates below (or our own), implement them as drop-in variants of the VCD sampling step, and evaluate them on the same grid as Phase 1.

Candidate directions:
1. **Adaptive α (contrast-strength scaling):** increase α when the real and noised distributions are too similar, e.g. `α_t = f(JS/KL divergence between p(·|v) and p(·|v'))`, or scale α by an image-quality estimate (blur/variance-of-Laplacian, BRISQUE/NIQE).
2. **Stronger / degradation-aware negative:** make the contrast branch more "blind" than the input, so the gap reopens. Options: a higher noise step that depends on input quality, a blank or mean image, or text-only (no image) logits as the prior.
3. **Cleaner positive branch:** restore the input before the real pass (unsharp mask, Wiener deconvolution, denoising, gamma correction, or a pretrained restoration net), so `v` is clearer than what the user supplied.
4. **Multi-level contrast:** contrast against several noise levels and weight them, instead of a single `v'`.
5. **Re-tuned plausibility constraint:** adapt β to input quality, since a degraded positive branch has a flatter distribution and the β cutoff behaves differently.

Report how each fix changes F1 relative to VCD and to baseline at each severity level, plus its compute cost (number of extra forward passes).

#### Fix B: DAE restoration before the real pass (DAE trained 2026-11-02)

- **What it is:** a small U-Net DAE (`src/fixes/dae.py`, 7.2M params). It restores the degraded image at native resolution, once per image, before LLaVA. Both VCD branches then use the restored image; LLaVA itself is never trained.
  - Residual output with a zero-initialised last layer, so it starts as the exact identity. 20% of training samples are clean→clean, so it should learn to leave clean images alone.
- **Data:** COCO **train2014** only (POPE is val2014, so there's no leakage). `scripts/download_coco_train.py` (login node) writes `$VCD_DATA_ROOT/coco_train2014/{images/,train.txt,val.txt}`: 10k train and 100 val images, a seeded shuffle of images with a short side ≥ 320.
- **Training:** `scripts/train_dae.py` (on ASPIRE, `qsub -v NAME=dae_v1 scripts/pbs/train_dae.pbs`).
  - Blind model: blur σ ~ U(2, 20) and noise std ~ U(0.1, 1.1), applied with `src.degradations.apply` (same code and uint8 rounding as the POPE images). 256 px crops, L1 loss, 60k steps.
  - Crops are degraded via a margin-padded region; this was checked to be bit-identical to degrading the full image.
  - Validation uses held-out train2014 images at the exact test `LEVELS`, reporting PSNR in → out. Outputs go to `$VCD_CHECKPOINT_ROOT/dae/<name>/` (on ASPIRE, `/scratch/users/ntu/c230202/checkpoints`). Same name and args = resume.
- **Local smoke test (36 images, 600 steps):** noise PSNR +4–5 dB, blur unchanged so far, clean passes through unchanged. **Watch blur in the full run:** noise dominates the L1 loss. If blur stays flat, reweight the loss per sample.
- **Trained: `dae_v1`** (job `26091677`, 60k steps, 0.036 s/step, about 40 min on an A100). Weights: `/scratch/users/ntu/c230202/checkpoints/dae/dae_v1/best.pt`. Final validation PSNR, degraded → restored:
  - **Noise:** s1 12.1 → 25.4, s2 9.2 → 23.6, s3 7.8 → 22.4, s4 6.8 → 21.1 dB.
  - **Blur:** unchanged at every level (identity). Clean images also pass through unchanged.
  - **Blur at these levels is unrecoverable, not a training bug.** Non-blind Wiener deconvolution given the true σ *lowers* PSNR at every level (−0.02 dB at best for s1, −6 dB for s4). Gaussian σ ≥ 5 removes the detail above ~0.07 cycles/px, below 8-bit precision.
  - **So Fix B is a noise-only fix.** On blur cells, DAE + baseline = baseline and DAE + VCD = VCD, so run the Fix B variants on the noise cells only.
- **POPE runs (submitted 2026-11-02):** greedy, noise s1–s4, real branch = `dae_v1`-restored PNGs.
  - `scripts/make_restored.py` (job `26091951`) writes `pope/images/restored/dae_v1/noise/s*/` plus `pope/restored_dae_v1_manifest.csv`. A local 3-image check gave PSNR +13.5–15 dB, matching validation. At s4 the output looks smooth and painterly, without visible invented objects.
  - `run_pope.py` keys: `restore: dae_v1` (real-branch input) and `contrast: diffusion|degraded`. `python -m scripts.make_configs fixb` makes 3 groups plus `fixb_smoke`.
  - Jobs are chained: restore `26091951` → smoke `26091952` → b0/b1/b2 `26091953`–`26091955`. Each step needs the previous one to succeed (`DEPEND=` in `submit.sh`).
  - Analysis: `python -m scripts.analyze_fixb` (on ASPIRE) → `results/fixb_comparison.csv`, reporting AUROC, matched F1, JS/|dm| and bootstrap CIs for the restore effect, b1 − VCD, and the interaction.
- **Result (2026-11-02; details in `docs/findings.md`, Phase 2 fix B): restoration works, but as an input fix, not a VCD fix.**
  - **B0 (baseline + DAE)** is the first intervention to raise AUROC above the baseline (+0.011 / +0.024 / +0.049 / +0.097 for s1–s4). It also beats its matched-threshold baseline (+1.4 to +6.5 F1).
  - **B1 (VCD + DAE)** has the best F1 (+0.8 to +7.4 over VCD). But AUROC(B1) < AUROC(B0) everywhere, the interaction is ≈ 0, and B0 thresholded to B1's yes-ratio beats B1. So VCD still only recalibrates.
  - **Restoring the positive doesn't reopen the contrast gap:** B1's JS is slightly *below* VCD's.
  - **B2 is dominated;** drop it.
- **Evaluation plan:** variants B0 = baseline + DAE (control), B1 = VCD + DAE, B2 = DAE-restored vs degraded as the contrast. Judge B1 by whether (B1 − B0) > (VCD − baseline), on AUROC and against the matched-threshold baseline, and check that the clean cell doesn't drop.

## Models and benchmarks

- **Primary model:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf` via HuggingFace `transformers`). Optional second model: InstructBLIP or Qwen-VL, to check that results generalise.
- **Primary benchmark:** POPE (yes/no object-existence probing on MSCOCO val2014; random / popular / adversarial splits). Metrics: accuracy, precision, recall, **F1**, yes-ratio.
- **Optional:** CHAIR (CHAIR_s / CHAIR_i) on free-form COCO captions, and MME perception subsets.
- Start with a **subset** (for example 500 POPE questions per split) while iterating, and run full splits only for final numbers.

## Compute notes

- **Local:** macOS (Apple Silicon), used for writing code and small MPS smoke tests only. A 7B VLM in fp16 needs about 15 GB, and VCD doubles the forward passes.
- **Remote:** NSCC **ASPIRE 2A** (user `c230202`), scheduled with PBS Pro. GPU nodes have NVIDIA A100 40GB cards, and one GPU is enough for LLaVA-1.5-7B in fp16.
- Code must work on `cuda`, `mps`, and `cpu`. Select the device automatically and never hard-code `.cuda()`.
- Batch-of-1 greedy or sampled decoding is fine. Cache degraded images on disk so they are not regenerated on every run.

### ASPIRE 2A workflow

Checked on this account (2026-10-30): project `personal-c230202` (~100k SU), home quota 50 GB, scratch quota 100 TB, and the login node has internet access.

- **GPU check:** passed on job `26057994`. The node had an A100-SXM4-40GB with driver 570.124.06 (supports CUDA ≤12.8), and torch could see and use the GPU.
- **Weights:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) is cached in `/scratch/users/ntu/c230202/hf_cache` (14 GB), so jobs can run with `HF_HUB_OFFLINE=1`.
- **Env:** conda env at `~/envs/vcd` (Python 3.11, torch 2.5.1+cu121, transformers 4.46.3), built by `scripts/setup_aspire.sh`. The conda/pip caches live in scratch.
- **Repo copy:** the repo lives at `~/SC4001-NN` on ASPIRE and is synced from the Mac with `rsync -az --exclude .git --exclude data --exclude results ./ aspire:~/SC4001-NN/`. There's no git remote yet.
- **Long commands:** run long remote commands with `nohup ... > $SCRATCH/logs/x.log 2>&1 &` so they survive a dropped connection.

- Login: `ssh c230202@aspire2antu.nscc.sg` (NTU login host). The server only accepts password/GSSAPI auth, with no public-key login, so the first login must be typed in a real terminal. Claude's `!`/Bash shell has no TTY and can't answer the password prompt.
  - The local `~/.ssh/config` has a `Host aspire` alias with ControlMaster multiplexing (ControlPersist 8h). After the user runs `ssh aspire` once in their own terminal, Claude can run `ssh aspire '<cmd>'` non-interactively over the shared connection.
  - If `ssh aspire` from Claude fails with permission denied, the master connection has expired. Ask the user to log in again; don't retry.
- Storage:
  - Home `/home/users/ntu/c230202`: small quota. Put code and the env here.
  - Scratch `/scratch/users/ntu/c230202`: large but purged periodically. Put the HF cache, datasets, degraded images, and raw outputs here, and copy final metric tables back to the repo.
- Set `export HF_HOME=/scratch/users/ntu/c230202/hf_cache` so model weights don't fill home.
- Compute nodes may have no internet access. Download model weights and datasets **on the login node** first (`huggingface-cli download llava-hf/llava-1.5-7b-hf`), then run offline in jobs (`HF_HUB_OFFLINE=1`).
- Don't run heavy compute on login nodes. Submit jobs with `qsub`.
- Check jobs with `qstat -u c230202`, and cancel them with `qdel <jobid>`.
- Sync code with `git push` locally and `git pull` on ASPIRE, or with `rsync`. Never sync `data/` or model weights from the Mac.

PBS job template (`scripts/pbs/run.pbs`):

```bash
#!/bin/bash
#PBS -N vcd-pope
#PBS -P personal-c230202          # project code; replace if a course project code is assigned
#PBS -q normal
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o /scratch/users/ntu/c230202/logs/

cd $PBS_O_WORKDIR
module load miniforge3/25.3.1
eval "$(conda shell.bash hook)"
conda activate $HOME/envs/vcd
export HF_HOME=/scratch/users/ntu/c230202/hf_cache
export HF_HUB_OFFLINE=1

python -m scripts.run_pope --config configs/$CONFIG
```

Submit with `qsub -v CONFIG=blur_s3_vcd.yaml scripts/pbs/run.pbs`. For sweeps, submit one job per config, or use a PBS job array (`#PBS -J 0-N`, indexed by `$PBS_ARRAY_INDEX`).

## Conventions

- Python 3.11. Dependencies go in `requirements.txt` (pin major versions of `torch`, `transformers`, `accelerate`, `pillow`, `numpy`).
- Every experiment is driven by a config (YAML or CLI args) recording model, decoding method, α, β, noise step, degradation type and severity, seed, and number of samples.
- Write results as one JSON/CSV row per (method, degradation, severity, split) under `results/`, and never overwrite: include a timestamp or run id.
- Fix seeds (`torch`, `numpy`, `random`) for every run. With sampling, report the mean over ≥3 seeds.
- Keep the decoding method pluggable. Baseline, VCD, and each fix should share one generation loop and differ only in how the next-token logits are computed.
- Do not commit datasets, model weights, or large outputs. Keep `data/`, `checkpoints/`, and bulky `results/` artifacts in `.gitignore`.

## Suggested layout (create as needed)

```
src/
  degradations.py   # blur, noise, low-light, jpeg, downsample: (PIL image, severity) -> PIL image
  vcd.py            # diffusion-noise helper, contrastive logits, plausibility mask
  fixes/            # one module per proposed fix
  decode.py         # shared generation loop with pluggable logit-processing strategy
  eval_pope.py      # run POPE, compute acc/prec/recall/F1/yes-ratio
scripts/            # entry points / sweep launchers
configs/            # experiment configs
notebooks/          # analysis + plots
results/            # metric tables (small CSV/JSON only)
```

## Things to watch for

- Make sure VCD's noise is applied to the **same preprocessed tensor** the model sees (after resize/normalize choices match the reference implementation). Otherwise the comparison is unfair.
- With greedy decoding, POPE answers are often a single token ("Yes"/"No"). Check that the contrastive step actually changes that first token, and log the raw logit gap `l(y|v) − l(y|v')` for "Yes" and "No". This gap is the main diagnostic for the hypothesis.
- Report the yes-ratio alongside F1. Hallucination often shows up as yes-bias.
