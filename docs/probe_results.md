# Linear probes vs ViewLayer: final results and validation

All numbers below come from TEST images only, with the same protocol as the group report (§3.3).

- **Split:** 100 DEV and 400 TEST images, chosen by `random.Random(0).sample(sorted(images), 100)`. This is identical to `fix_eval/common.py`.
- **Cells:** 27 condition × split cells, with 2,400 questions each.
- **Hyperparameters:** every hyperparameter, threshold and probe weight is fit on DEV only.
- **Uncertainty:** 95% paired bootstrap over questions with 500 resamples. "Sig ↑/↓" counts the cells whose interval lies entirely above or below zero. The comparison is against the baseline for AUROC and against the matched-threshold baseline for F1.
- **Thresholds:** rows marked * use one DEV-tuned global threshold, the same rule as the report's starred rows.
- **Hardware:** the runs used the local RTX 4500 Ada with transformers 5.x. That is a different machine from ASPIRE.
  - Baseline AUROCs match the report to 3 dp.
  - Threshold-0 baseline F1 differs from the report by ≤ 0.3, because of fp16 differences in the forward pass.

## 1. The two probes

Both probes are a logistic regression on the decoder's hidden state at the last prompt token. That is the position whose output is the answer token. The state comes from the forward pass the baseline already runs, so neither probe needs an extra pass.

| | **Gold probe** (`Probe L24`) | **Clean-teacher probe** (`Clean-teacher probe L16+24`) |
|---|---|---|
| Features | layer-24 hidden state (4,096 dims) | layer-16 and layer-24 hidden states, concatenated (8,192 dims) |
| Training rows | 9,234 unique DEV questions: 100 images × 9 conditions, deduplicated across splits | the same 9,234 rows |
| **Label for each row** | **POPE gold answer** (human annotation) | **the model's own answer on the clean version of the same image** (sign of the layer-32 Yes−No margin on the clean image). No human labels. |
| What it learns | "is the object really there?" | "would I say yes if I could see this image clearly?" (self-distillation from clean to degraded) |
| Regularisation | C = 1e-3 | C = 1e-4 |
| How C was chosen | 5-fold image-grouped CV on DEV, scored by gold AUROC | 5-fold image-grouped CV on DEV, scored by gold AUROC |
| Threshold | DEV F1-optimal | DEV F1-optimal |
| Ceiling | gold labels, so it can in principle exceed the clean model | the clean model's knowledge: teacher labels agree with gold on **89.3%** of DEV questions |
| Inference cost | 1 pass + one 4,096-d dot product | 1 pass + one 8,192-d dot product |

**Use of labels, stated precisely.** The clean-teacher probe's weights never see a human label. Two DEV-level choices do use DEV gold, exactly as ViewLayer's view, layer and threshold choices do:

- **The value of C.** The candidates were 1e-4, 3e-4 and 1e-3, with CV AUROC .8867, .8843 and .8773. The choice is nearly irrelevant.
- **The decision threshold.**

AUROC does not depend on the threshold. So the teacher probe's AUROC is label-free apart from a near-irrelevant choice of C.

**Why one probe beats the other.** Two factors separate them:

- **Label quality.** The gold probe uses true labels. The teacher inherits the clean model's 10.7% disagreement with gold.
- **Ceiling.** The teacher's target is the clean model. The clean baseline's AUROC is .940, and the teacher probe on clean images reaches .949. On degraded images it approaches, but cannot exceed, what the model would see with a clean image.

## 2. Main table (TEST, 27 cells)

| Method | Passes | Mean ΔAUROC | Sig ↑/↓ AUROC | Sig ↑/↓ F1 | AUROC Clean | AUROC Blur s4 | AUROC Noise s4 | F1 Blur s4 | F1 Noise s4 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (threshold 0) | 1 | – | – | – | .940 | .750 | .737 | 40.8 | 43.9 |
| Baseline (DEV threshold)* | 1 | – | – | – | .940 | .750 | .737 | 67.9 | 68.7 |
| Layer 20 only* | 1 | +.0133 | 25/0 | 3/1 | .943 | .779 | .754 | 67.4 | 67.8 |
| ViewLayer* (report, Table 4) | 5 | +.0277 | 26/0 | 10/0 | .945 | .782 | .817 | 68.6 | 73.9 |
| Lens-LR (64 logit-lens scalars)* | 1 | +.0272 | 24/0 | 17/0 | .948 | .803 | .778 | 72.2 | 71.5 |
| **Clean-teacher probe L16+24 (no labels)*** | **1** | **+.0383** | **26/0** | **22/0** | .949 | .830 | .801 | 76.9 | 73.6 |
| **Gold probe L24*** | **1** | **+.0503** | **26/0** | **24/0** | .952 | .848 | .820 | 78.1 | 75.3 |
| Object prior only (no image)* | 1 | −.1465 | 2/23 | 5/19 | .704 | .704 | .704 | 70.7 | 70.7 |

- **Lens-LR** is a logistic regression on the 32 per-layer Yes−No margins plus the 32 per-layer entropies. It is a probe on 64 scalars rather than 4,096 dimensions. It already matches ViewLayer's mean gain with a single pass.
- **Object prior only** is the control described in §4.
- **ViewLayer** was recomputed here on blur s4 and noise s4. It gives .781 / .817, matching the report.

### 2.1 Per condition (TEST, mean of three splits)

ViewLayer's columns are taken from report Table B.1. All F1 values use DEV thresholds.

| Condition | AUROC Base | AUROC L20 | AUROC ViewLayer (5 passes) | AUROC Teacher (1 pass) | AUROC Gold probe (1 pass) | F1 Base (DEV thr.) | F1 ViewLayer | F1 Teacher | F1 Gold probe |
|---|---|---|---|---|---|---|---|---|---|
| Clean | .940 | .943 | .945 | .949 | **.952** | 84.6 | 84.5 | 87.0 | **88.1** |
| Blur s1 | .907 | .915 | .915 | .928 | **.936** | 83.2 | 83.4 | 85.0 | **86.0** |
| Blur s2 | .877 | .891 | .892 | .910 | **.922** | 80.4 | 80.8 | 83.3 | **84.1** |
| Blur s3 | .836 | .857 | .857 | .886 | **.900** | 77.2 | 77.8 | 80.8 | **82.0** |
| Blur s4 | .750 | .779 | .782 | .830 | **.848** | 67.9 | 68.6 | 76.9 | **78.1** |
| Noise s1 | .910 | .917 | .925 | .927 | **.935** | 83.0 | 83.7 | 84.7 | **85.5** |
| Noise s2 | .878 | .885 | .903 | .905 | **.916** | 80.5 | 81.7 | 82.2 | **83.2** |
| Noise s3 | .823 | .836 | .873 | .867 | **.883** | 75.5 | 79.0 | 78.9 | **80.1** |
| Noise s4 | .737 | .754 | .817 | .801 | **.820** | 68.7 | 73.9 | 73.6 | **75.3** |

The pattern by degradation type:

- **Blur:** both probes beat ViewLayer at every level, and the gap grows with severity (blur s4: +.048 for the teacher, +.066 for the gold probe).
- **Noise:**
  - At s1–s2 the teacher roughly equals ViewLayer. At s3–s4 the teacher is below it (−.006 / −.016).
  - The gold probe is above ViewLayer at every noise level. The margin at s4 (+.003) is within noise.

### 2.2 Per split (TEST, mean over the nine conditions)

| Split | Base | L20 | Teacher | Gold probe |
|---|---|---|---|---|
| Random | .883 | .902 | .916 | .923 |
| Popular | .859 | .870 | .913 | .927 |
| Adversarial | .811 | .821 | .839 | .854 |

The probes gain most on *popular*: +.054 for the teacher and +.068 for the gold probe. Popular negatives are frequent objects, so the final-layer "Yes" readout is biased toward them. The probe direction is less tied to object frequency.

### 2.3 Head-to-head with ViewLayer on identical TEST questions

Here ViewLayer was recomputed from its own views at layer 20, with a DEV threshold of −13.11. The table shows Δ = method − ViewLayer with paired-bootstrap 95% CIs (1,000 resamples). AURC is the area under the risk–coverage curve; lower is better.

| Cell | ViewLayer | Teacher | Δ teacher [95% CI] | Gold probe | Δ gold [95% CI] | AURC ViewLayer / teacher / gold |
|---|---|---|---|---|---|---|
| blur s4 random | .835 | .863 | +.029 [+.019, +.037] | .868 | +.033 [+.021, +.044] | .130 / .115 / .098 |
| blur s4 popular | .789 | .862 | +.073 [+.063, +.083] | .885 | +.096 [+.084, +.109] | .187 / .119 / .088 |
| blur s4 adversarial | .721 | .765 | +.044 [+.034, +.054] | .790 | +.069 [+.056, +.084] | .256 / .221 / .183 |
| noise s4 random | .868 | .832 | −.035 [−.046, −.023] | .837 | −.030 [−.044, −.018] | .125 / .153 / .120 |
| noise s4 popular | .821 | .834 | +.014 [+.001, +.025] | .864 | +.043 [+.027, +.057] | .180 / .149 / .105 |
| noise s4 adversarial | .762 | .736 | −.025 [−.039, −.012] | .759 | −.002 [−.019, +.015] | .241 / .249 / .207 |
| **blur s4 mean** | .781 | .830 | **+.048** | .848 | **+.066** | .191 / .151 / .123 |
| **noise s4 mean** | .817 | .801 | **−.016** | .820 | **+.003** | .182 / .183 / .144 |

F1 at s4, with DEV thresholds for all three methods:

| | ViewLayer | Teacher | Gold probe |
|---|---|---|---|
| Blur s4 | 70.9 | 76.9 | 78.1 |
| Noise s4 | 74.7 | 73.6 | 75.3 |

### 2.4 Selective answering: recognising uncertainty (topic E)

In selective answering, the method abstains on its least-confident questions, ranked by distance from its own threshold.

| Method | Passes | AURC ↓ (27 cells) | Acc @100% | @80% | @50% | AURC blur s4 | AURC noise s4 |
|---|---|---|---|---|---|---|---|
| Baseline (threshold 0) | 1 | .176 | 73.4 | 78.2 | 84.5 | .320 | .314 |
| Baseline (DEV thr.)* | 1 | .131 | 76.9 | 81.3 | 85.3 | .210 | .226 |
| Layer 20 only* | 1 | .121 | 77.5 | 82.3 | 86.8 | .186 | .211 |
| ViewLayer (recomputed, s4 only) | 5 | – | – | – | – | .191 | .182 |
| Clean-teacher probe* | 1 | .097 | 79.3 | 84.7 | 90.6 | .151 | .183 |
| Gold probe L24* | 1 | .077 | 81.4 | 86.9 | 93.2 | .123 | .144 |

When the gold probe answers only its most confident half of questions, it is right 93.2% of the time. The baseline is right 85.3% at the same coverage.

## 3. Layer profile of the gold probe

| Layer | 8 | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|---|
| Mean ΔAUROC | +.013 | +.046 | +.048 | +.050 | **+.050** | +.049 | +.049 |
| Noise s4 | .786 | .813 | .819 | .820 | .820 | .819 | .817 |
| Blur s4 | .815 | .844 | .845 | .847 | .848 | .846 | .845 |
| Sig ↑/↓ AUROC | 10/7 | 25/0 | 26/0 | 26/0 | 26/0 | 26/0 | 26/0 |

The curve is flat from layer 12 to layer 32. Layer 24 is therefore not a cherry-picked peak, and the result does not depend on the choice of layer. Concatenating layers 24, 28 and 32 does slightly worse (+.046), which is a sign of overfitting with 12k features.

## 4. Validation checklist

| # | Check | Result | What it rules out |
|---|---|---|---|
| 1 | **Image-level DEV/TEST split.** Probes are fit on the 100 DEV images. Every number is on the 400 TEST images. No image appears in both. | All tables above | Memorising images or questions |
| 2 | **C chosen by 5-fold image-grouped CV on DEV.** Folds split by image, so CV never scores a question whose image was in training. | CV AUROC .898 for gold L24 and .887 for the teacher. TEST mean AUROC is consistent with these values. | Overfitting the regularisation to TEST |
| 3 | **Object-prior control.** Logistic regression on a one-hot of the queried object only, with no image. | AUROC .704 in every condition. Mean ΔAUROC −.147, below baseline in 23 cells. | "The probe just learns which objects are usually present." The probe gains on top of the image evidence; it does not replace it with a prior. |
| 4 | **Matched-threshold F1.** Each probe's F1 is compared with the baseline thresholded to the same yes-ratio. | Gold 24/0 cells, teacher 22/0 cells. ViewLayer 10/0. | "The gain is just a Yes/No bias shift", which is the VCD failure from report §4.2 |
| 5 | **Discrimination, not calibration.** The headline metric is AUROC, which no threshold or constant shift can change. | 26/0 cells for both probes | Same as #4 |
| 6 | **Held-out degradation type (teacher).** The probe is trained on clean + blur DEV only and tested on noise, then trained on clean + noise and tested on blur. | Unseen noise: +.035, 12/0, noise s4 .795 (baseline .737). Unseen blur: +.041, 12/0, blur s4 .822 (baseline .750). | "The probe only works because it saw the same corruption in training." It transfers across corruption types. |
| 7 | **Clean-only training (teacher).** The probe is trained on clean DEV images only. | ≈0 gain on degraded TEST (scratchpad run; not in the tables) | Shows *why* #6 works: the probe needs to have seen *some* degraded inputs to learn which directions survive degradation, but not the specific corruption. |
| 8 | **Layer robustness.** Seven layers were probed. | Flat from L12 to L32 (§3) | Layer cherry-picking |
| 9 | **Paired comparison with ViewLayer** on identical questions, with ViewLayer recomputed from logged view logits. | Reproduces the report's .781 / .817 exactly (§2.3) | Comparing against a mis-implemented ViewLayer |
| 10 | **Teacher label quality.** | 89.3% agreement with gold on DEV | Explains the gap between the teacher and the gold probe |
| 11 | **Offline readout is consistent with the decoder.** The layer-32 logit-lens margin is the logged baseline margin, and per-layer readouts go through the model's own final RMSNorm and LM head. | The offline top-k re-score reproduces the earlier GPU run of the top-k fix to within 0.004 logits | Probing a different representation from the one the model decodes |

### Limitations

These apply to both probes.

- **Not training-free, unlike ViewLayer.** The probes need a small fitted head: 4,096 or 8,192 weights, fit in seconds on CPU from about 9k cached hidden states. The teacher needs no labels, but it does need clean/degraded image pairs at training time. Those come free with synthetic degradations, but not with real-world degraded photos.
- **Optimistic intervals.** The bootstrap resamples questions, and questions within an image are correlated. This affects every method in the report equally.
- **First token only.** Like ViewLayer, the probes score only the first token, so they apply to yes/no probing. CHAIR-style captioning would need a per-token head.
- **Teacher operating point under noise.** With one global DEV threshold, the teacher's yes-ratio rises to 62–66% at s4, against 53% for the DEV-threshold baseline and 56% for ViewLayer. A deployment worried about hallucination should use a stricter threshold, as the report says for ViewLayer.
- **Prior work.** Object-existence probes on VLM hidden states are established: arXiv 2508.04567, ASD (ACL 2025), TruthPrInt 2503.10602, HALP 2603.05465, VIB-Probe 2601.05547. The gold probe must be cited as an application of that idea, not claimed as new. We found no prior work for the clean-teacher, label-free self-distilled probe for degraded inputs. The closest are arXiv 2604.22529 and LP-CLIP 2309.10361. That search covered abstracts only.

## 5. How the probe strengthens the case for ViewLayer

ViewLayer's report makes three claims that it supports only indirectly:

- **(a)** The final layer *ignores* evidence that intermediate layers have (§4.4, §5).
- **(b)** Layer 20 is the right place to read.
- **(c)** On noise, evidence is *destroyed* before the decoder, so only extra views can recover it (§4.5 (v)).

The gold probe is the best linear readout of a single pass, so it acts as an upper bound for any single-pass readout. It tests all three claims directly.

1. **"Computed but ignored" is confirmed, and quantified.**
   - Read with the Yes/No unembedding, layer 32 gives AUROC .737 on noise s4 and .750 on blur s4.
   - A linear probe on *the same layer-32 state* gives .817 and .845.
   - So the evidence is in the final hidden state; the fixed `w_Yes − w_No` direction just does not read it. ViewLayer's layer-20 readout recovers part of the missing evidence on blur (.779 of the probe's .845). This is direct evidence for ViewLayer's mechanism.
2. **The layer choice is validated.**
   - Probe accuracy saturates from layer 12 onward, and the logit-lens margin peaks at layers 19–20 (report Fig. 4a).
   - Together these are consistent with layer 20 being a good *fixed* readout. By layer 20 the evidence is fully formed. After layer 20, what degrades is how well the `w_Yes − w_No` direction lines up with that evidence, not the evidence itself, which persists to layer 32.
3. **The view component is shown to be necessary for noise.**
   - Every single-pass readout we tried plateaus at about .80–.82 on noise s4. That includes the best linear probe on any layer, multi-layer probes, object-token and image-token pooled features, regularised and ensembled probes, self-training, and patch-level internal confidence.
   - ViewLayer's five views reach .817 without any training. They match the supervised single-pass ceiling, and on noise s4 random they exceed it (.868 vs .837).
   - This is strong support for ViewLayer's claim that blurred views add information that is not present in one pass of the noisy image. The views denoise before the vision encoder, and no readout of a single pass can replace that.
4. **The ablation in report §4.4 is explained.**
   - **Blur:** the evidence survives in a single pass but is mis-read. The fix is a better readout: layer 20, or a probe.
   - **Noise:** the evidence is partly lost at the encoder. The fix is more input: the views.
   - The probe results reproduce this split exactly. The probe gains most over ViewLayer on blur (+.066) and ties it on noise (+.003).

**Suggested sentence for the report:** "A supervised linear probe on a single forward pass, an upper bound for any single-pass readout, matches ViewLayer on noise s4 (.820 vs .817) but exceeds it on blur s4 (.848 vs .781). This confirms that on blur the evidence is present but mis-read, while on noise ViewLayer's views recover information that no single-pass readout can."

## 6. The clean-teacher probe as a low-compute alternative to ViewLayer

| | ViewLayer | Clean-teacher probe |
|---|---|---|
| Image forward passes per question | 5 (batched) | **1** (the baseline's own pass) |
| Latency (A100, report §5) | 0.33 s | ≈0.08 s (baseline) + a 8,192-d dot product |
| Human labels | DEV gold for view, layer and threshold selection | DEV gold only for C (near-irrelevant) and the threshold. **None for the weights.** |
| Training | none | logistic regression on 9,234 cached DEV hidden states (CPU, seconds). Needs clean + synthetically degraded DEV pairs. |
| Mean ΔAUROC (27 cells) | +.0277 | **+.0383** |
| Sig ↑/↓ AUROC | 26/0 | 26/0 |
| Sig ↑/↓ F1 vs matched baseline | 10/0 | **22/0** |
| Clean / blur s4 / noise s4 AUROC | .945 / .782 / **.817** | **.949 / .830** / .801 |
| F1 blur s4 / noise s4 | 68.6 / **73.9** | **76.9** / 73.6 |
| AURC blur s4 / noise s4 | .191 / **.182** | **.151** / .183 |
| Unseen corruption type | n/a (training-free) | transfers: +.035 on unseen noise, +.041 on unseen blur |

**In short:** for one fifth of the inference compute and no labelled training data, the clean-teacher probe:

- beats ViewLayer on average (+.038 vs +.028 mean ΔAUROC);
- beats it clearly on blur (+.048 AUROC, +6 F1 at s4);
- beats it on clean images;
- comes close on heavy noise (−.016 AUROC, −1.1 F1 at s4, roughly equal AURC).

It is a different mechanism, not a ViewLayer extension. ViewLayer adds information at the input. The teacher learns, from the model's own clean-image behaviour, which hidden directions still carry the answer once the image is degraded.

**Where it loses.** It loses on noise s3–s4, the regime where the input itself has lost information (§5, point 3). There, ViewLayer's views remain the better choice when 5× compute is affordable.

## 7. Pilot: probes on ViewLayer's views (blur s4 and noise s4 only)

**Question.** The probe fixes the readout, and ViewLayer's views fix the input. Do the two gains add up?

**Setup.**
- The probes are **not retrained**. These are the same gold and clean-teacher probes as in §1, fit on the original view of the DEV images.
- Each probe scores every one of ViewLayer's views: original, blur σ=1/2/3 and h-flip. The scores are then averaged, as ViewLayer averages margins.
- The views come from the view runs already made for the head-to-head in §2.3. Those exist only for **blur s4 and noise s4**, so this is a 6-cell pilot, not the full 27-cell table.
- Script: `python -m scripts.probe_views`.

### 7.1 TEST AUROC, mean of three splits

| Method | Passes | Blur s4 | Noise s4 |
|---|---|---|---|
| Baseline | 1 | .750 | .737 |
| ViewLayer | 5 | .781 | .817 |
| Clean-teacher probe | 1 | .830 | .801 |
| Clean-teacher probe, original + blur σ=1 | 2 | .830 | .833 |
| **Clean-teacher probe × 5 views** | 5 | **.832** | **.852** |
| Gold probe L24 | 1 | .848 | .820 |
| Gold probe, original + blur σ=1 | 2 | .848 | .849 |
| **Gold probe × 5 views** | 5 | **.849** | **.866** |

### 7.2 Per split, with the paired-bootstrap Δ against ViewLayer

The intervals are 95% paired-bootstrap intervals over 500 resamples.

| Cell | ViewLayer | Teacher (1 pass) | Teacher × 5 views | Δ teacher×5 − ViewLayer [95% CI] | Gold × 5 views | Δ gold×5 − ViewLayer [95% CI] |
|---|---|---|---|---|---|---|
| blur s4 random | .835 | .863 | .866 | [+.021, +.039] | .870 | [+.022, +.045] |
| blur s4 popular | .789 | .862 | .864 | [+.065, +.084] | .887 | [+.085, +.109] |
| blur s4 adversarial | .721 | .765 | .766 | [+.035, +.056] | .791 | [+.057, +.084] |
| noise s4 random | .868 | .832 | .883 | [+.006, +.024] | .888 | [+.009, +.030] |
| noise s4 popular | .821 | .834 | .882 | [+.051, +.071] | .903 | [+.070, +.094] |
| noise s4 adversarial | .762 | .736 | .790 | [+.018, +.038] | .808 | [+.033, +.059] |

### 7.3 What it shows

- **It breaks the noise ceiling.** Every single-pass readout plateaued at about .80–.82 on noise s4 (§5, point 3). With views added, the label-free teacher reaches .852 and the gold probe .866. Both beat ViewLayer significantly in **all 6 cells**.
- **The gains add as the diagnosis predicted.**
  - On noise, the views lift the probe by +.05: they restore evidence the noise destroyed.
  - On blur, the views add almost nothing (+.002): the evidence was present but mis-read, and the probe already fixes that. ViewLayer's own ablation found the same, with views-only giving .753 on blur s4.
- **A cheaper variant also beats ViewLayer.** With two passes, original plus blur σ=1, the teacher reaches .830 / .833, beating ViewLayer's .781 / .817 on both degradations at 2/5 of the compute.
- **The labelled probe is not needed.** The label-free teacher × 5 views is within .014–.017 of the gold probe × 5 views.

### 7.4 Caveats before this goes in the report

- **Pilot scale.** These are 6 cells (s4 only). Clean, s1–s3, and the full Sig ↑/↓ counts need the view runs for the other 7 conditions: 28 GPU runs, about 6 h on the local GPU.
- **Post-hoc 2-view subset.** The view set for "× 5 views" is ViewLayer's own DEV-chosen set, so it involves no new selection. The 2-view subset (original + blur σ=1) was picked after looking at these TEST numbers. For the report, choose the view set on DEV.
- **No F1 yet.** F1 and thresholds are not computed here. AUROC only.
- **Framing.** This reuses ViewLayer's views, so credit them. The fair statement is that the probe replaces ViewLayer's fixed layer-20 readout. With the views kept, that one change takes noise s4 from .817 to .852 and blur s4 from .781 to .832.

**Options for the team:**

| Option | Compute | Status |
|---|---|---|
| (a) Clean-teacher probe as the low-cost single-pass alternative | 1 pass | Results complete (§2, §6) |
| (b) Add "probe + views" as the best-accuracy operating point | 2 or 5 passes | Needs the ~6 h run for the full table |

## Reproducing

```bash
export HF_HOME=/mnt/Data/akshay/hf_cache HF_HUB_OFFLINE=1
python -m scripts.run_layers                       # per-layer logits + hidden states, 9 conditions (1 pass/question)
python -m scripts.run_layers --views blur1 blur2 blur3 hflip   # ViewLayer views for the head-to-head (blur/noise s4)
python -m scripts.analyze_layers                   # main table, probes, held-out degradation -> results/layers_*
python -m scripts.analyze_selective                # results/selective_table.*
python -m scripts.compare_viewlayer                # teacher vs ViewLayer -> results/compare_viewlayer.md
COMPARE_METHOD="Probe L24*" python -m scripts.compare_viewlayer   # -> results/compare_viewlayer_goldprobe.md
```
