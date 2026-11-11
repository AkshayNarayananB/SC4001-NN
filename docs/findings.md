# Findings

Running log of experimental findings, newest stage last. Metrics live in `results/metrics/`; regenerate any table with `python -m scripts.summarize --prefix <stage>`.

## Stage 0: Reproducing VCD on clean images (2026-11-01)

**Goal:** confirm our VCD implementation reproduces the paper (Leng et al., CVPR 2024, Table 1) on clean POPE images before testing degraded images.

**Setup**
- **Model:** LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`), fp16, one A100-40GB on ASPIRE 2A.
- **Benchmark:** POPE MSCOCO, with random, popular and adversarial splits. Each split has 3000 questions over the same 500 val2014 images, half of them "yes".
- **Prompt, noise and scoring:** ported from the official repo (DAMO-NLP-SG/VCD). Before running anything we checked that our code matches theirs:
  - the noise function gives bit-identical output for the same random stream
  - the VCD logit step is identical
  - the metrics match their `eval_pope.py` exactly
- **VCD settings:** α = 1 and β = 0.1, as in the paper. We tried noise steps T = 500 (the repo script's default) and T = 999.
- **Decoding:**
  - **Sampling** (temperature 1, no top-p/top-k), averaged over 3 seeds (55, 56, 57). This is the paper's "Regular" setting.
  - **Greedy**, run once per method.

### Results: F1 (%), sampling, mean ± std over 3 seeds

| Split | Baseline (ours) | Baseline (paper) | VCD T=500 (ours) | VCD T=999 (ours) | VCD (paper) |
|---|---|---|---|---|---|
| Random | 81.19 ± 0.45 | 81.33 | 83.53 ± 0.30 | **87.09 ± 0.25** | 87.16 |
| Popular | 80.20 ± 0.78 | 80.06 | 82.48 ± 0.81 | **85.48 ± 0.33** | 85.06 |
| Adversarial | 77.81 ± 0.29 | 77.57 | 79.62 ± 0.27 | **81.77 ± 0.31** | 81.33 |

### Accuracy, F1 and yes-ratio (%), sampling (3 seeds) and greedy

| Split | Method | Acc (sample) | F1 (sample) | Yes-ratio (sample) | Acc (greedy) | F1 (greedy) | Yes-ratio (greedy) |
|---|---|---|---|---|---|---|---|
| Random | Baseline | 83.20 | 81.19 | 39.3 | 87.07 | 85.51 | 39.3 |
| Random | VCD T=500 | 85.03 | 83.53 | 40.9 | 86.63 | 85.37 | 41.4 |
| Random | VCD T=999 | 87.64 | 87.09 | 45.7 | 88.47 | 87.99 | 46.0 |
| Popular | Baseline | 82.00 | 80.20 | 40.9 | 85.90 | 84.41 | 40.4 |
| Popular | VCD T=500 | 83.72 | 82.48 | 42.9 | 84.83 | 83.67 | 42.9 |
| Popular | VCD T=999 | 85.74 | 85.48 | 48.2 | 85.73 | 85.53 | 48.6 |
| Adversarial | Baseline | 79.23 | 77.81 | 43.6 | 83.67 | 82.37 | 42.7 |
| Adversarial | VCD T=500 | 80.37 | 79.62 | 46.3 | 82.23 | 81.38 | 45.4 |
| Adversarial | VCD T=999 | 81.34 | 81.77 | 52.3 | 81.73 | 82.29 | 53.1 |

The paper's Appendix B.4 reports greedy results on random only:
- baseline: 87.10 acc / 85.54 F1, against our 87.07 / 85.51
- VCD: 88.49 acc / 88.02 F1, against our 88.47 / 87.99

### Findings

1. **Our implementation reproduces the paper.** With T = 999, VCD is within 0.5 F1 of the paper on all three splits. The baseline is within 0.25 F1. Greedy results on random match the paper's appendix to within 0.1 on every metric.
2. **The paper's POPE numbers need noise step T = 999, not the repo default of 500.** At T = 500, VCD's gain is roughly halved (+2.3 / +2.3 / +1.8 F1, against +5.9 / +5.3 / +4.0 at T = 999). With greedy decoding at T = 500, VCD gives no gain or slightly hurts.
   - This explains several open reproduction complaints on the VCD GitHub repo (issues #24, #16, #9). Issue #24 is where a user pointed out T = 999 for POPE.
   - All later stages use T = 999.
3. **How strongly the contrast image is destroyed controls VCD's effect.**
   - At T = 500, the noised image still correlates ≈0.86 with the input. That's about as noisy as our own noise levels s2–s3 once LLaVA resizes the image.
   - At T = 999 the correlation is ≈0.29.
   - A "blinder" contrast image gives a larger gain. This is direct evidence for Phase 2 candidate fix #2: a stronger, degradation-aware negative.
4. **VCD's POPE gain comes mainly from answering "yes" more.**
   - The yes-ratio rises from about 39–44% to 46–52%.
   - Recall jumps (72.5 → 83.3 on random) while precision drops a little (92.2 → 91.2 on random, 83.5 → 80.0 on adversarial).
   - On the adversarial split, VCD's yes-ratio overshoots 50% (the true rate). Precision then pays for it, and that's why the adversarial gain is smallest.
5. **With greedy decoding, VCD helps mainly on the random split.** The gains are +2.5 F1 on random, +1.1 on popular and −0.1 on adversarial. The paper's headline gains use sampling, where VCD's plausibility cutoff also removes some of the sampling noise. Part of VCD's sampling gain over "Regular" may therefore come from that truncation rather than from the contrast itself.
6. **Greedy is a much stronger baseline than sampling.** On random, greedy baseline F1 is 85.5 and sampling baseline F1 is 81.2. Greedy VCD (88.0) beats greedy baseline by less than sampling VCD (87.1) beats sampling baseline (81.2).

### Implications for the main experiment

- Use **T = 999, α = 1, β = 0.1** as "VCD" in all later stages. T = 500 can be kept as a weaker-contrast reference point.
- Report **both greedy and sampling**. Sampling matches the paper's setting. Greedy is the stronger, deterministic baseline.
- At T = 999 the contrast image is always far blinder than even our strongest degradation. So if VCD fails on degraded inputs, the cause can't be a weak contrast image. It has to be the real branch drifting toward language priors, which is the project's hypothesis.
- Track yes-ratio and precision, not just F1. VCD's gain is largely a yes-bias shift, and degraded inputs may change that bias on their own.

### Implementation notes

- **transformers 4.46:** LLaVA mis-detects "legacy" image processing on a 1-token decode step. We pass `inputs_embeds` for decode steps to avoid it (`src/decode.py`).
- **One-off debug check:** on one question, the real and contrast branches gave the same Yes/No margin. A debug job confirmed the branches do receive different images (mean pixel difference 0.42). The noise just didn't move that particular margin at fp16 resolution, so it's not a bug.
- **Speed on an A100:** baseline about 0.08 s/question, VCD about 0.16 s/question (two branches). A full 9000-question VCD run takes about 25 minutes.

## Stage 1: Severity-ladder pilot (2026-11-01)

**Goal:** check that the degradation levels actually hurt LLaVA, and hurt it steadily more from s1 to s4, before running the full baseline-vs-VCD grid.

**Setup**
- **Decoding:** greedy baseline only (no VCD).
- **Questions:** a fixed random subset of 100 questions per split (90 distinct images, 55 yes / 45 no), the same subset for every level.
- **Levels:** clean, blur σ = 2/4/8/16 and noise std = 0.05/0.10/0.20/0.40, all applied at native resolution.
- **Run:** one job (`26074656`).
- With 100 questions, one answer is worth 1 point of accuracy, so differences of a few points are within noise.

### Results (mean over the 3 splits, %)

| Level | Accuracy | F1 | Yes-ratio | Precision | Recall |
|---|---|---|---|---|---|
| Clean | 80.0 | 79.6 | 43.0 | 90.9 | 70.9 |
| Blur s1 (σ 2) | 78.7 | 78.1 | 42.3 | 89.9 | 69.1 |
| Blur s2 (σ 4) | 78.7 | 77.6 | 40.3 | 91.8 | 67.3 |
| Blur s3 (σ 8) | 69.7 | 63.2 | 27.3 | 95.2 | 47.3 |
| Blur s4 (σ 16) | 56.7 | 37.5 | 14.3 | 91.4 | 23.6 |
| Noise s1 (0.05) | 79.7 | 79.4 | 43.3 | 90.1 | 70.9 |
| Noise s2 (0.10) | 80.0 | 79.6 | 43.0 | 90.9 | 70.9 |
| Noise s3 (0.20) | 80.0 | 79.6 | 43.0 | 90.9 | 70.9 |
| Noise s4 (0.40) | 71.7 | 68.7 | 35.3 | 88.0 | 56.4 |

Recall is identical across splits because the three splits share the same "yes" questions. Only the negatives differ.

### Findings

1. **The noise ladder is too weak.**
   - LLaVA is almost unaffected by noise up to std 0.20 (PSNR 15 dB). Per question, the Yes/No logit margin shifts (mean |Δ| 0.23 → 0.37 → 0.63 for s1–s3), but only 2–4 of 100 answers flip. Those flips cancel out, so the metrics come out identical to clean.
   - Only s4 (std 0.40) hurts (−8 acc). The likely reason is that LLaVA's resize from ~640 px to 336 px averages away much of the per-pixel noise.
2. **The blur ladder is uneven.** s1 and s2 (σ 2, 4) are nearly harmless (about −1 acc), while s3 and s4 (σ 8, 16) cause large drops (−10, −23 acc). The interesting range is between σ 4 and 16.
3. **Degradation causes misses, not hallucinations, on POPE.**
   - As images degrade, LLaVA says "no" more: the yes-ratio falls from 43% to 14% at blur s4.
   - Recall collapses (71% → 24%) while precision stays around 90%.
   - So the dominant failure on degraded images is *missing objects that are there*, not *inventing objects that aren't*.
   - This matters for the hypothesis. On clean images, VCD's POPE gain came mostly from pushing answers toward "yes" (Stage 0). On degraded images that push might *help* recall, so VCD's gain need not shrink the way the hypothesis predicts. Stage 2 will show which effect wins.

### Calibration sweep and new levels (2026-11-01)

**Setup:** the same greedy baseline and 100-question subset, with images degraded on the fly at explicit parameters (`configs/calib/`, job `26074709`). Clean accuracy is 80.0.

**Blur:** mean accuracy and the drop from clean, by σ

| σ | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 80.0 | 78.7 | 76.7 | 72.7 | 73.0 | 69.7 | 66.0 | 62.7 | 58.0 | 56.7 | 52.0 |
| Drop | 0.0 | −1.3 | −3.3 | −7.3 | −7.0 | −10.3 | −14.0 | −17.3 | −22.0 | −23.3 | −28.0 |

**Noise:** mean accuracy and the drop from clean, by std

| std | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| Accuracy | 78.3 | 71.7 | 67.7 | 72.3 | 66.3 | 62.3 | 60.0 |
| Drop | −1.7 | −8.3 | −12.3 | −7.7 | −13.7 | −17.7 | −20.0 |

**Findings**
- Blur hurts smoothly and almost linearly between σ 4 and 20.
- Noise is bumpier: 0.6 beats 0.5, which is within the ±4-point noise of 100 questions. Noise also tops out lower. Even std 1.0 (pixels close to saturated random values) only costs 20 points, so LLaVA is much more robust to noise than to blur.
- The yes-ratio falls along with accuracy in both cases: from 43% down to 9% at σ 20, and to 19% at std 1.0. Degradation pushes the model toward "no".

**New levels** (in `LEVELS`, `src/degradations.py`)
- blur σ = 5 / 7 / 10 / 16, a drop of about −3 / −7 / −14 / −23
- noise std = 0.3 / 0.5 / 0.7 / 1.0, a drop of about −2 / −12 / −14 / −20

**Bookkeeping**
- The first-ladder Stage 1 results are archived in `results/metrics/archive/levels_v1/`. The raw outputs are on scratch as `levels_v1_stage1_*`.
- Run configs now record each cached level's parameter (`level_param`). A future change to `LEVELS` then makes an old run dir fail the config check instead of being silently resumed.

### Stage 1 rerun on the rebuilt dataset (2026-11-01)

**Setup:** the dataset was rebuilt with the new levels (job `26074795`), then the Stage 1 pilot was rerun on it (job `26074796`): greedy baseline, the same 100-question subset per split.

**Results:** mean over the 3 splits (%)

| Level | Accuracy | F1 | Yes-ratio | Precision | Recall | PSNR (dB) | SSIM |
|---|---|---|---|---|---|---|---|
| Clean | 80.0 | 79.6 | 43.0 | 90.9 | 70.9 | — | — |
| Blur s1 (σ 5) | 76.7 | 75.0 | 38.3 | 91.4 | 63.6 | 21.3 | .54 |
| Blur s2 (σ 7) | 73.0 | 69.0 | 32.0 | 93.8 | 54.5 | 20.3 | .50 |
| Blur s3 (σ 10) | 66.0 | 57.5 | 25.0 | 92.3 | 41.8 | 19.3 | .47 |
| Blur s4 (σ 16) | 56.7 | 37.5 | 14.3 | 91.4 | 23.6 | 18.0 | .45 |
| Noise s1 (0.3) | 75.7 | 73.7 | 37.3 | 91.1 | 61.8 | 12.1 | .13 |
| Noise s2 (0.5) | 69.0 | 64.4 | 32.0 | 87.9 | 50.9 | 9.2 | .07 |
| Noise s3 (0.7) | 63.7 | 52.4 | 21.3 | 93.8 | 36.4 | 7.8 | .05 |
| Noise s4 (1.0) | 58.3 | 43.5 | 18.7 | 86.7 | 29.1 | 6.8 | .03 |

**Findings**
1. **Both ladders now decline monotonically and fairly evenly.** Accuracy steps are −3/−7/−14/−23 for blur and −4/−11/−16/−22 for noise. The damage is comparable between the two types at each level, which makes blur and noise directly comparable in Stage 2.
2. **Pixel metrics don't predict model damage across types.** Noise s4 has PSNR 6.8 and blur s4 has 18.0, yet both cost about 22 accuracy points. Plots of VCD's gain should therefore use the **level** (or baseline accuracy) on the x-axis, not PSNR or SSIM.
3. **The "degradation → more no, not more yes" pattern holds for both types.**
   - Yes-ratio falls from 43% to 14–19%.
   - Precision stays around 87–94% while recall falls from 71% to 24–29%.
   - On POPE, a degraded LLaVA mostly *misses* objects rather than hallucinating them.

## Stage 2: Baseline vs VCD on degraded images (2026-11-01)

**Setup**
- Full POPE splits (3000 questions each), all 9 conditions (clean, blur s1–s4, noise s1–s4).
- Baseline vs VCD (α=1, β=0.1, T=999). Greedy decoding plus sampling with seeds 55/56/57.
- 24 jobs, `26075007`–`26075030`.
- Analysis: `python -m scripts.analyze --stage stage2`, run on ASPIRE, writes three files:
  - `results/stage2_comparison.csv`: per-split F1 / accuracy / yes-ratio, with a paired-bootstrap 95% CI on ΔF1, VCD fixes vs breaks, and McNemar p for greedy.
  - `results/stage2_diagnostics.csv`: the contrast term.
  - `results/stage2_margins.csv`: AUROC and the matched-threshold control.
- **AUROC scoring corrected 2026-11-11.** The first version scored answers forced by VCD's plausibility mask as ±∞, which ties 60–70% of the questions and lowers AUROC even when no answer changes. They are now ranked by the real-branch margin (`decision_score` in `scripts/analyze.py`, the same convention as `fix_eval/common.py`). The old numbers are kept in the `auroc_vcd_tied` column.
- **Status:** complete. All 72 runs finished: greedy, plus sampling with 3 seeds in every cell.

### Results: mean over the 3 splits (%)

| Condition | Greedy F1 base → VCD | ΔF1 | Sampling F1 base → VCD | ΔF1 | Yes-ratio base → VCD (greedy) |
|---|---|---|---|---|---|
| Clean | 84.1 → 85.3 | +1.2 | 79.7 → 84.8 | +5.1 | 40.8 → 49.2 |
| Blur s1 | 75.1 → 82.1 | +7.0 | 72.3 → 81.1 | +8.8 | 33.7 → 47.0 |
| Blur s2 | 70.0 → 79.3 | +9.3 | 67.8 → 78.1 | +10.3 | 30.6 → 45.2 |
| Blur s3 | 60.3 → 72.7 | +12.5 | 58.8 → 71.7 | +13.0 | 24.8 → 40.2 |
| Blur s4 | 40.3 → 59.8 | +19.5 | 43.0 → 58.5 | +15.5 | 15.1 → 31.2 |
| Noise s1 | 76.1 → 81.7 | +5.6 | 73.4 → 80.9 | +7.5 | 34.6 → 46.1 |
| Noise s2 | 69.6 → 78.1 | +8.5 | 66.9 → 77.4 | +10.5 | 30.3 → 46.0 |
| Noise s3 | 57.6 → 72.9 | +15.3 | 57.5 → 71.6 | +14.1 | 23.5 → 44.0 |
| Noise s4 | 43.0 → 63.1 | +20.1 | 45.5 → 61.7 | +16.2 | 16.8 → 38.8 |


Every degraded cell has a ΔF1 CI above zero. Clean greedy reproduces Stage 0: +2.5 / +1.1 / −0.1 F1 on random / popular / adversarial.

### First-token margin analysis (greedy, mean over splits)

Margin = logit(Yes) − logit(No) at the first answer token. The VCD margin is recomputed from the logged logits, including the plausibility mask; it agrees with the decoded answer on ≥99.96% of questions.
- **AUROC:** how well a margin separates present from absent objects, independent of where the threshold sits.
  - **AUROC VCD:** VCD's actual decisions, with mask-forced answers ranked by the real-branch margin.
  - **AUROC VCD unmasked:** the raw contrastive margin `2·real − contrast`, before the plausibility mask.
- **"Matched":** the baseline margin, thresholded so its yes-ratio equals VCD's. In other words, a pure bias shift with no contrast at all.

| Condition | AUROC real (= baseline) | AUROC contrast branch | AUROC VCD | AUROC VCD unmasked | F1 baseline | F1 matched | F1 VCD |
|---|---|---|---|---|---|---|---|
| Clean | .938 | .694 | .933 | .922 | 84.1 | 86.3 | 85.3 |
| Blur s1 | .907 | .579 | .905 | .897 | 75.1 | 82.8 | 82.1 |
| Blur s2 | .881 | .555 | .878 | .874 | 70.0 | 79.5 | 79.3 |
| Blur s3 | .838 | .535 | .836 | .834 | 60.3 | 73.1 | 72.7 |
| Blur s4 | .754 | .508 | .753 | .758 | 40.3 | 59.9 | 59.7 |
| Noise s1 | .911 | .607 | .907 | .899 | 76.1 | 82.6 | 81.7 |
| Noise s2 | .878 | .528 | .875 | .871 | 69.6 | 78.9 | 78.1 |
| Noise s3 | .826 | .481 | .823 | .827 | 57.6 | 73.0 | 72.9 |
| Noise s4 | .736 | .459 | .735 | .744 | 43.0 | 63.4 | 63.1 |

### Findings

1. **The hypothesis is refuted as stated: VCD's F1 gain *grows* with severity.** Greedy goes from +1.2 (clean) to +19.5 (blur s4) and +20.1 (noise s4), and sampling shows the same trend. Every degraded cell is significant: paired-bootstrap CI > 0, and McNemar p < 0.05 in 21 of 24 greedy cells; the exceptions are adversarial at blur s1, noise s1 and noise s2.

2. **The gain is entirely a Yes/No threshold shift, not better discrimination.**
   - VCD does not raise AUROC. Its decisions rank within −0.006 to +0.002 of the baseline: the 95% CI is below zero in 21 of 27 cells and above zero in one (noise s4, random, +0.002).
   - The first version of this analysis reported a drop of −0.016 to −0.054. Most of that was the tie artifact described above, not a loss of discrimination.
   - A baseline that simply moves its threshold to VCD's yes-ratio matches or beats VCD's F1 everywhere. VCD − matched ranges from −1.4 to +0.7 F1; the CI excludes zero only on the negative side (12 of 27 cells) and never on the positive side.
   - On POPE, VCD acts as a recalibration toward "yes". It is worth the most where degradation has pushed the baseline furthest toward "no" (Stage 1: precision stays at ~90% while recall collapses), and that is why its gain scales with severity.

3. **The contrast branch isn't "blind" on clean images, so subtracting it removes real signal there. On the worst images it carries none, and the subtraction stops hurting.**
   - At T=999 the contrast-branch margin still separates present from absent objects with AUROC 0.69 on clean images.
   - That falls to ≈0.5 at blur s4, and to below 0.5 at noise s3–s4.
   - The unmasked VCD margin follows the same pattern. It is below the baseline on clean and mildly degraded images (−0.016 clean; CI below zero in 18 of 27 cells) and above it in 4 cells at the highest severities: blur s4 random (+0.013), noise s3 random (+0.007), noise s4 random (+0.018) and noise s4 adversarial (+0.007).
   - The contrast's overall offset (the language prior leans "no" less than the real branch does) supplies the yes-shift at every level.

4. **The mechanism the hypothesis predicted is real, but it isn't what drives F1.**
   - JS(real ‖ contrast) falls from 0.135 (clean) to 0.066 (blur s4) and 0.083 (noise s4): the two branches do converge as the input degrades.
   - Mean |dm| (the real-minus-contrast Yes/No margin) falls from 2.6 to 1.8 (blur s4).
   - The *discriminative* part of the contrast, dm on gold-yes minus dm on gold-no, halves: 3.79 → 1.83 (blur s4) and 1.81 (noise s4).
   - For gold-no questions, dm turns positive under degradation (−0.38 → +1.07 at noise s4). VCD then pushes absent objects toward "yes" as well, i.e. it adds hallucinations along with the recall it recovers.
   - So the contrast carries less and less image information. It just keeps a large constant yes-push, which on POPE happens to correct the degradation-induced no-bias.

### Implications for Phase 2

- Plain F1 on POPE can't tell a better decoder from a recalibrated one. Every fix should be reported against two references: (a) the **matched-threshold baseline** and (b) **AUROC**, alongside F1 and yes-ratio.
- A fix that genuinely restores contrast should raise AUROC above the baseline's, which VCD's decisions do in only 1 of 27 cells. That makes the bar clear and measurable.
- Fixes aimed at making the negative "blinder" (degradation-aware T, blank image, text-only prior) are cheap to test offline first:
  - Run only the contrast branch, log its margin, and recompute AUROC and F1 from the stored real-branch logits.
  - The real branch is already logged for every question.

## Phase 2, fix B: DAE restoration before decoding (2026-11-02)

**Setup**
- **DAE:** `dae_v1`, a 7.2M-param U-Net trained on COCO train2014 (job `26091677`, 60k steps). Final validation PSNR, degraded → restored: noise s1–s4 12.1/9.2/7.8/6.8 → 25.4/23.6/22.4/21.1 dB.
- **Blur is left unchanged at every level**, and that's not a training bug: non-blind Wiener deconvolution given the true σ *lowers* PSNR at every level. Fix B is therefore evaluated on noise only.
- **Restored POPE images:** `pope/images/restored/dae_v1/noise/` (job `26092158`), with the same PSNR gains as validation.
- **Runs:** greedy, full splits, noise s1–s4 (jobs `26092160`–`26092162`).
  - **B0:** baseline on the restored image.
  - **B1:** VCD on the restored image (contrast = restored + diffusion noise, T=999).
  - **B2:** VCD contrasting the restored image against the original degraded image.
- **Comparison:** against the Stage 2 greedy baseline and VCD on the same questions. Analysis: `python -m scripts.analyze_fixb` → `results/fixb_comparison.csv`.
- **Rescored 2026-11-11** with the corrected AUROC convention (mask-forced answers ranked by each variant's own real-branch margin). Only the AUROCs of the masked variants (VCD, B1, B2) changed; F1 and yes-ratios are identical.
  - `python -m scripts.analyze_fixb --test-only` → `results/fixb_comparison_test.csv` scores the 400 TEST images of `fix_eval`, for comparison with the other fixes. B0 there is .921 / .904 / .875 / .836.

### Results: mean over the 3 splits

| Noise level | AUROC base / VCD / **B0** / B1 / B2 | F1 base / VCD / **B0** / B1 / B2 | Yes-ratio base / VCD / B0 / B1 / B2 |
|---|---|---|---|
| s1 | .911 / .907 / **.922** / .916 / .913 | 76.1 / 81.7 / 80.0 / **82.9** / 79.5 | 34.6 / 46.1 / 37.4 / 46.7 / 39.8 |
| s2 | .878 / .875 / **.902** / .896 / .891 | 69.6 / 78.1 / 74.6 / **78.9** / 75.5 | 30.3 / 46.0 / 34.4 / 44.9 / 39.1 |
| s3 | .826 / .823 / **.875** / .868 / .863 | 57.6 / 72.9 / 69.2 / **75.7** / 72.8 | 23.5 / 44.0 / 31.6 / 43.7 / 38.8 |
| s4 | .736 / .735 / **.833** / .825 / .823 | 43.0 / 63.1 / 61.0 / **70.5** / 67.9 | 16.8 / 38.9 / 26.8 / 41.9 / 38.3 |

**Matched-threshold controls (F1, %)**

| Noise level | B0 | Baseline at B0's yes-ratio | B1 | **B0 at B1's yes-ratio** |
|---|---|---|---|---|
| s1 | 80.0 | 78.6 | 82.9 | **83.8** |
| s2 | 74.6 | 73.0 | 78.9 | **80.6** |
| s3 | 69.2 | 65.4 | 75.7 | **77.1** |
| s4 | 61.0 | 54.5 | 70.5 | **71.7** |

**Contrast size (VCD → B1 → B2)**
- JS: .132/.131/.113/.083 → .120/.110/.099/.082 → .014/.024/.040/.052.
- Mean |dm|: 2.56/2.56/2.41/2.13 → 2.40/2.26/2.11/1.88 → 0.63/0.88/1.19/1.42.

### Findings

1. **Restoration alone (B0) is the first intervention that improves discrimination, not just calibration.**
   - AUROC rises above the baseline at every level, by +0.011 / +0.024 / +0.049 / +0.097 (s1–s4), and the gain grows with severity.
   - The 95% CI is above zero in all 12 (level, split) cells.
   - B0 beats the baseline thresholded to B0's yes-ratio by +1.4 / +1.6 / +3.8 / +6.5 F1.
   - VCD never cleared either bar in Stage 2.
2. **VCD on top of restoration (B1) gives the best F1, but VCD's own contribution is still only a threshold shift.**
   - B1 has the highest F1 at every level: +1.2 / +0.8 / +2.8 / +7.4 over VCD.
   - But AUROC(B1) < AUROC(B0) in every cell, by −0.004 to −0.008. VCD costs slightly more AUROC on a restored input (−0.006 to −0.008 by level) than on a degraded one (−0.001 to −0.004), so the interaction is zero or slightly negative: −0.002 / −0.003 / −0.004 / −0.006 for s1–s4, with the CI below zero in 7 of 12 cells and never above.
   - Against the baseline, B1's AUROC is higher in 10 of 12 cells (CI > 0) and lower in none. That gain is the restoration's: B1 stays below B0.
   - B0 thresholded to B1's yes-ratio beats B1 by +0.9 to +1.7 F1.
   - **B1's gain over VCD therefore comes entirely from restoration.** The honest description is "restore before decoding", with VCD as an optional recalibration. It is not a fix to VCD's contrast.
3. **A cleaner positive does not reopen the contrast gap.** B1's JS and |dm| are slightly *smaller* than VCD's. The "converging branches" problem from Stage 2 isn't caused by the positive branch being degraded; it's the T=999 negative that dominates.
4. **B2 (restored vs degraded as the contrast) is dominated.** Its contrast is small (JS 0.01–0.05), its AUROC is below B0's, and its F1 is below B1's. Drop it.
5. **Cost:** one DAE forward pass per image (7.2M params, negligible next to the 7B LLM) and no extra LLM forward passes. B0 costs the same as the baseline; B1 costs the same as VCD.

### Implications
- For the report: Fix B works, but as an **input fix that helps any decoder**. VCD's degraded-input problem is better described as "VCD only recalibrates, and restoration supplies the missing information".
- Blur can't be helped by restoration at these levels, so blur cells need a different fix: decoding-side (Fix A) or recalibration.
- Optional follow-ups:
  - B0/B1 with sampling seeds 55/56/57, for the sampling table.
  - The clean-image cell, to confirm that an always-on DAE doesn't hurt clean images. Validation passed clean images through exactly unchanged, so the POPE result should be identical to the baseline.

## Phase 2, ViewLayer+DAE: a denoised view on top of ViewLayer, and VCD's contrast on top of that (2026-11-11)

**Setup**
- Started as "ViewVCD": VCD's formula with a stronger real branch, `score = (1+α) · mean over views of m_L(v) − α · m_L(v_cd)`.
  - Views: the degraded image, blur σ 1/2/3, horizontal flip (all from the ViewLayer records) and the `dae_v1`-restored copy.
  - Contrast `v_cd`: VCD's diffusion-noised image at T=999, with the same noise as the Stage 2 runs. No plausibility mask.
- Views, layer, α ∈ {0, 0.25, 0.5, 1, 2} and the threshold are chosen on the 100 DEV images; results are on the 400 TEST images (`fix_eval` scoring).
- Runs: jobs `26199460` (smoke) and `26199954` (9 conditions). Code, full tables and caveats: `ViewLayerDAE/README.md`. Tables: `results/viewlayer_dae_*.csv`.

### Results: mean over the 3 splits (TEST)

| Condition | AUROC baseline | AUROC VCD | AUROC ViewLayer | AUROC ViewLayer+DAE | F1 VCD | F1 ViewLayer | F1 ViewLayer+DAE |
|---|---|---|---|---|---|---|---|
| Clean | .940 | .935 | .945 | .945 | 85.7 | 84.5 | 85.2 |
| Blur s1 | .907 | .904 | .915 | .915 | 82.3 | 83.4 | 83.3 |
| Blur s2 | .877 | .875 | .892 | .892 | 79.0 | 80.8 | 80.9 |
| Blur s3 | .836 | .834 | .857 | .857 | 73.1 | 77.8 | 77.3 |
| Blur s4 | .750 | .749 | .782 | .782 | 59.7 | 68.6 | 66.9 |
| Noise s1 | .910 | .906 | .925 | .927 | 81.9 | 83.7 | 84.1 |
| Noise s2 | .878 | .874 | .903 | .909 | 78.0 | 81.7 | 82.7 |
| Noise s3 | .823 | .821 | .873 | .882 | 72.5 | 79.0 | 79.7 |
| Noise s4 | .737 | .737 | .817 | .833 | 62.7 | 73.9 | 75.5 |

### Findings

1. **DEV switched the contrast off.** The selected setting is all six views at layer 20 with α = 0. DEV AUROC falls monotonically with α: 0.882 (α = 0), 0.878 (0.25), 0.861 (2).
2. **The DAE view adds to ViewLayer on noise and changes nothing elsewhere.**
   - AUROC is above ViewLayer's in all 12 noise cells (+0.002 to +0.018, CI > 0). On blur and clean the difference is within ±0.001.
   - Against the baseline: AUROC is higher in 26 of 27 cells and lower in none. F1 beats the matched-threshold baseline in 13 cells (ViewLayer: 11).
   - F1 on blur s3–s4 is slightly below ViewLayer's (66.9 vs 68.6 at s4). The ranking is the same; the single DEV threshold landed differently.
3. **VCD's contrast does not help a strong real branch.**
   - Forced on at α = 0.25 with the six views, AUROC drops by about 0.005 on clean and blur and is level at noise s3–s4.
   - Plain VCD without the mask at layer 20 beats the baseline in 17 of 27 cells, but ViewLayer's layer-only variant does better with no subtraction, so the gain is the layer-20 readout.
   - The lead from the unmasked-VCD AUROC (above the baseline at blur s4 and noise s3–s4) does not carry over.
4. **Against the DAE alone (same TEST images, same scorer).**
   - Versus Fix B's B0 (the DAE view at layer 32): AUROC is higher in 18 of 27 cells and lower in none. The gain is +0.005 on clean, +0.009 to +0.031 on blur and +0.005 to +0.007 at noise s1–s3.
   - **At noise s4 the DAE alone is as good or better:** .836 at layer 32 and .840 at layer 20, against .833 for ViewLayer+DAE (differences not significant). F1 there is 76.9 / 77.0 against 75.5.
   - Versus the DAE view alone at its best layer (20): AUROC is higher in 6 cells (clean and blur s4) and lower in none.
   - So the combination is never significantly worse than either part. The views and the layer-20 readout carry blur; the DAE carries noise, and on heavy noise the extra views add nothing to it.
5. **Sanity checks.** The `viewlayer` variant reproduces `ViewLayer/README.md` exactly. The contrast view at layer 32 matches the Stage 2 contrast logits to within 0.11. The DAE view at layer 32 reproduces Fix B's B0 on the TEST images (.921 / .904 / .875 / .836).

### Follow-ups done (2026-11-11)
- Fix B was rescored through the shared convention and on the TEST images (see the Fix B section and finding 4 above).
- The Stage 2 AUROC was recomputed with the corrected scoring (see the Stage 2 margin analysis).
