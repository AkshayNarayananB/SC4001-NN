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

## Research plan

### Phase 1: Reproduce the failure mode
Compare **baseline decoding** and **VCD** on clean images and on several degradation types and severities.

Degradations (decided 2026-10-30): **Gaussian blur** and **additive Gaussian noise**, 4 levels each plus clean. Blur is what the project brief names. Noise is the closest match to VCD's own diffusion noise, so it's the most direct test of the hypothesis. Low light is an optional third type if time allows. JPEG and downsampling were dropped: downsampling behaves much like blur, and JPEG barely matters until the quality is extreme. Fewer types keeps the grid (types × levels × methods × splits) manageable.

- Levels live in `LEVELS` in `src/degradations.py`; that's the single source of truth. Recalibrate there if the contact sheet or a quick baseline run shows they're too weak or too strong.
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
- **Built on ASPIRE (2026-10-30, job `26060930`):** 4500 PNGs (1.5 GB) plus `manifest.csv` in `/scratch/users/ntu/c230202/data/pope/`. This is the **canonical copy**; every experiment reads it. Mean PSNR/SSIM per level:
  blur σ 2/4/8/16 → 24.3/22.0/19.9/18.0 dB, SSIM .70/.57/.49/.45; noise std .05/.10/.20/.40 → 26.3/20.5/15.0/10.4 dB, SSIM .61/.38/.20/.09.
- **Sharing:** the dataset is **not** in git (`data/` is gitignored). The user shares it with teammates as a zip outside git. The code remote will be GitHub, which isn't set up yet.
- **Rebuilding elsewhere:** a rebuild is reproducible on the same machine, but not byte-identical across Pillow versions. The JPEG decoder differs between Pillow 11.0 and 11.1, so pixels can differ by ±1 on fewer than 0.005% of them. Don't mix locally built images into results.
- Pipeline:
  - `scripts/download_pope.py` downloads the question files and images. It needs internet, so run it on the login node.
  - `scripts/make_degraded.py` builds every version plus the manifest (CPU job: `scripts/pbs/make_degraded.pbs`).
  - `scripts/contact_sheet.py` makes a visual check grid at the model's 336 px input size.

### Phase 2: Propose and evaluate two fixes
Pick two from the candidates below (or our own), implement them as drop-in variants of the VCD sampling step, and evaluate them on the same grid as Phase 1.

Candidate directions:
1. **Adaptive α (contrast-strength scaling):** increase α when the real and noised distributions are too similar, e.g. `α_t = f(JS/KL divergence between p(·|v) and p(·|v'))`, or scale α by an image-quality estimate (blur/variance-of-Laplacian, BRISQUE/NIQE).
2. **Stronger / degradation-aware negative:** make the contrast branch more "blind" than the input, so the gap reopens. Options: a higher noise step that depends on input quality, a blank or mean image, or text-only (no image) logits as the prior.
3. **Cleaner positive branch:** restore the input before the real pass (unsharp mask, Wiener deconvolution, denoising, gamma correction, or a pretrained restoration net), so `v` is clearer than what the user supplied.
4. **Multi-level contrast:** contrast against several noise levels and weight them, instead of a single `v'`.
5. **Re-tuned plausibility constraint:** adapt β to input quality, since a degraded positive branch has a flatter distribution and the β cutoff behaves differently.

Report how each fix changes F1 relative to VCD and to baseline at each severity level, plus its compute cost (number of extra forward passes).

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
