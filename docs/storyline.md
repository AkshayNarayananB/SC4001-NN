# Project storyline: draft for the group report

This is a mock narrative arc for the whole project, built on the current draft (`SC4001_Group_Report.pdf`). It slots in the new results from `docs/probe_results.md` and the missing Table 4 rows.

ViewLayer stays the headline method. The probes add two things:

1. a **diagnostic** that explains *why* ViewLayer works, and
2. a **low-compute alternative**: the clean-teacher probe.

## Working title options

- *Does Visual Contrastive Decoding Still Help When the Image Is Already Degraded?* (current)
- Subtitle, updated: *Training-free hallucination mitigation for LLaVA-1.5 under blur and noise, and why the evidence is there but mis-read*

## The story in one paragraph

We set out to show that VCD breaks on degraded images. It doesn't: its F1 gain *grows* with degradation. But we show the gain is a bias shift, not better grounding. A threshold change reproduces it, and its discrimination (AUROC) never moves. Every variant that subtracts a "blind" branch fails the same way, because there is no reliably blind image to subtract: zero hook, shuffle hook, phase null, mean null, RITUAL, DeCo, and entropy routing. What does work is reading the model *differently*. ViewLayer reads layer 20 instead of layer 32 and averages over lightly blurred views. It improves discrimination in 26/27 cells. A linear probe explains why. On blur, the answer is present in the hidden state but the Yes/No unembedding mis-reads it. On noise, the evidence is lost before the decoder, and only extra views recover it. Finally, a probe trained only on the model's own clean-image answers matches ViewLayer's average gain with one forward pass instead of five, and with no human labels.

## Act by act

### Act 1: The question (Introduction, §1)

- **Hook.** VCD is the go-to training-free hallucination fix, and it assumes the input is clean. Real images are blurry and noisy.
- **Hypothesis.** On degraded images the real and noised branches converge, so VCD collapses to the baseline exactly when hallucination is likeliest.
- **Research question.** Do training-free decoding fixes still help on degraded inputs, and do they improve grounding or just shift the Yes/No bias?
- **Contributions.** There are now six:
  1. VCD reproduction (T=999)
  2. degradation study
  3. evaluation protocol
  4. negative results for 7+ modifications
  5. ViewLayer
  6. **probe analysis and clean-teacher probe** (new)

### Act 2: Setting up a fair test (§2–3)

- Related work (§2): unchanged. **Add one paragraph on probing**:
  - "Linear probes on VLM hidden states detect object hallucination (2508.04567; ASD; TruthPrInt; HALP; VIB-Probe), but all are trained with gold labels on clean images."
  - This sets up the gap the clean-teacher probe fills.
- Degradation ladder and reasoning (§3.1): unchanged. Blur and noise, four levels each, calibrated to drop accuracy evenly.
- **Evaluation protocol (§3.3): the backbone of the story.** Three components:
  - an image-level DEV/TEST split;
  - AUROC as the main metric, with the decision-preserving tie rule;
  - a matched-threshold baseline.
- Every later claim hangs on the protocol's distinction between discrimination and bias.

### Act 3: The surprise (§4.1–4.2)

- **Reproduction (Arushi):** VCD reproduces within 0.5 F1, but only at T=999. That explains the open GitHub issues.
- **The twist:** the F1 gain *grows* with degradation (+1.5 → +18.9). The hypothesis looks refuted.
- **The resolution:**
  - The matched-threshold baseline equals VCD's F1 in every condition, and AUROC is unchanged.
  - Degradation makes LLaVA say "no" (yes-ratio 41% → 16%), and VCD's constant push toward "yes" compensates.
  - The hypothesised mechanism is real (JS halves, the discriminative contrast halves), but the outcome is the opposite of what we predicted.
- **Takeaway line:** "On degraded images VCD recalibrates rather than grounds."

### Act 4: Trying to fix VCD, and why nothing subtractive works (§3.4, §4.3, §4.5)

Group the methods by idea, as Table 1 already does, and give each group a one-line verdict:

| Idea | Methods (owner) | Verdict |
|---|---|---|
| Blinder negative | Zero hook, shuffle hook (Akshay) | Zero hook ≈ baseline (−.0005; the mask forces ~70% of answers and the zero branch is just the text prior). Shuffle hook is *worse* in 26/27 cells, because shuffled patches still carry content, so subtracting them removes evidence. |
| Content-free null | Phase null, mean null (Sanjana) | Phase null keeps scene gist, so it is worse in 17 cells. The mean null is slightly *below* chance, giving tiny gains. |
| Restoration | Fix B: DAE + VCD ([owner]) | [fill in] |
| Extra views | RITUAL (Sanjana) | Amplifies the "no" bias. Only the blur view helps on noise, and that observation seeded ViewLayer. |
| Layer readout | DeCo (Sanjana); JSD routing, entropy top-k, gradient exit (Akshay) | DeCo's mixing weight is too small. JSD routing collapses. Entropy routing ≈ baseline (top-k as run −.0003; DEV-tuned +.005). Selecting by *low* entropy hurts (hard min-entropy −.021, gradient exit −.036): on degraded inputs, confident layers are confidently wrong. |

**Unifying failure analysis.** Every method that keeps the final-layer readout of the original image moves AUROC by less than ±.005. Subtracting any "blind" branch fails because no branch is reliably blind. Only methods that leave **the final layer** or **the original view** give real gains. That observation leads into Act 5.

### Act 5: ViewLayer (§3.5, §4.4; Sanjana)

- **Method.** The mean Yes−No margin over 5 views (original, blur σ 1/2/3, h-flip), read out at layer 20. Every choice is made on DEV.
- **Results.** It beats the baseline in 26/27 cells, gaining +.032 at blur s4 and +.080 at noise s4, and up to +5.1 F1 over an equally tuned baseline.
- **Ablation.** The layer fixes blur and the views fix noise, and the two gains add.
- **Leaves open:** *why* does reading layer 20 help, and is there more evidence to recover?

### Act 6 (new): Is the evidence there? A probe analysis (§4.6; Akshay)

This is a new subsection placed after ViewLayer. Use `docs/probe_results.md` §2, §3 and §5.

- **Setup.** The best *linear* readout of one forward pass: logistic regression on the last-token hidden state, fit on DEV gold. It works as an upper bound for any single-pass readout.
- **Finding 1: computed but ignored.** The layer-32 *hidden state* gives AUROC .845 / .817 at blur s4 / noise s4. The layer-32 *Yes/No logits* give .750 / .737. The final layer has the answer, but the unembedding direction doesn't read it. This is direct evidence for ViewLayer's premise.
- **Finding 2: blur and noise fail differently.**
  - On blur, the probe beats ViewLayer by +.066, because there is more evidence left to read.
  - On noise, every single-pass readout we tried caps at about .82, and ViewLayer's views reach that cap without training.
  - So blur is a *read-out* problem and noise is an *input* problem. This explains the report's ablation.
- **Controls:**
  - object prior alone: .704;
  - flat layer profile from L12 to L32;
  - image-grouped CV;
  - 24/0 cells above the matched-threshold baseline.

### Act 7 (new): A one-pass, label-free alternative (§4.7; Akshay)

- **Idea.** Self-distillation from clean to degraded. Train the probe to predict *the model's own answer on the clean image* from its hidden state on the degraded image. No human labels are needed, and synthetic degradation gives the pairs for free.
- **Results:**
  - mean ΔAUROC +.038 vs ViewLayer's +.028, in 26/0 cells;
  - F1 above the matched baseline in 22 cells vs ViewLayer's 10;
  - better than ViewLayer on blur s4 by +.048, slightly worse on noise s4 by −.016;
  - **1 pass vs 5.**
- **Generalisation.** Trained without noise, it still gains +.035 on noise. Trained without blur, it gains +.041 on blur.
- **Uncertainty (topic E).** In risk–coverage, AURC is .097 vs .131 for the baseline. At 50% coverage, accuracy is 90.6% vs 85.3%.
- **Optional extension, pending the team's decision and a ~6 h run.** The pilot is in `docs/probe_results.md` §7. Applying the same probe to ViewLayer's views reaches .852 on noise s4 (ViewLayer .817) and .832 on blur s4 (ViewLayer .781). It beats ViewLayer in all 6 s4 cells. This would become the best-accuracy operating point, with the 1-pass probe as the cheap one.
- **Novelty claim (hedged).** "To our knowledge, self-distilled probes from clean to degraded inputs have not been used for VLM hallucination."

### Act 8: Discussion and conclusion (§5–6)

- **What we learned about VCD.** Unchanged: it recalibrates rather than grounds. Report AUROC and matched thresholds.
- **What we learned about degradation.** There are two failure types:
  - *Mis-read evidence* (blur): fix the readout, with a layer or a probe.
  - *Lost evidence* (heavy noise): fix the input, with views.
- **Practical recommendation:**

  | Situation | Recommended method |
  |---|---|
  | Compute-bound, or blur-like degradation | clean-teacher probe (1 pass) |
  | Heavy noise, 5× compute affordable, training-free required | ViewLayer |

- **Limitations:**
  - one model, one benchmark, two synthetic degradations;
  - first token only;
  - small DEV set;
  - the teacher needs clean/degraded pairs at training time;
  - bootstrap intervals are optimistic.
- **Future work:**
  - a probe on ViewLayer's views (does it break the noise cap?);
  - per-token probes for CHAIR;
  - real-world degradations without clean pairs.

## Fills for the current draft

**Table 4, new and filled rows.** These are local runs; baseline AUROC is identical and threshold-0 F1 is within 0.3.

| Method | Passes | Mean ΔAUROC | Sig ↑/↓ AUROC | Sig ↑/↓ F1 | Clean | Blur s4 | Noise s4 | F1 Blur s4 | F1 Noise s4 |
|---|---|---|---|---|---|---|---|---|---|
| Zero-hook VCD | 2 | −.0005 | 1/18 | 0/0 | .940 | .750 | .737 | 40.6 | 43.2 |
| Shuffle-hook VCD | 2 | −.0032 | 0/26 | 0/10 | .936 | .747 | .735 | 39.6 | 43.5 |
| Entropy routing, as run (top-k K=5, T=0.5) | 1 | −.0003 | 3/7 | 1/0 | .936 | .753 | .739 | 59.5 | 62.3 |
| Entropy routing (DEV K=6, T=5)* | 1 | +.0048 | 17/0 | 1/2 | .941 | .762 | .741 | 67.5 | 67.8 |
| Clean-teacher probe (no labels)* | 1 | +.0383 | 26/0 | 22/0 | .949 | .830 | .801 | 76.9 | 73.6 |
| Linear probe L24 (DEV gold)* | 1 | +.0503 | 26/0 | 24/0 | .952 | .848 | .820 | 78.1 | 75.3 |

Other fixes needed in the draft:

- **Entropy routing row:** replace the "‡ all 500 images / n/a" row with the TEST rows above.
- **Table 1:** split the zero/shuffle row into two rows, since their results differ. Add the two probe rows under a new idea, "Learned readout", with 1 pass each.
- **§3.4:** update the entropy-routing description to "top-k entropy-weighted layers from {18…30}". Mention the gradient early-exit variant and its negative result.
- **§4.5:** add the shuffle-hook explanation (shuffled patches still carry content, like the phase null). Add the entropy finding: confident early layers are confidently wrong on degraded input.
- **Abstract:**
  - Change the opening of the sentence "We propose ViewLayer…" to "We propose ViewLayer … and show with a linear probe that …".
  - Add: "A probe trained on the model's own clean-image answers matches ViewLayer's average gain with a single forward pass and no labels."
- **Appendix A:** replace the `scripts.analyse_dola` command, which was deleted in the cleanup, with the commands in `docs/probe_results.md` § Reproducing.
- **AI use statement (Akshay's part):**
  - Claude Code was used to write the layer-logging, probe and contrast pipelines and to run the GPU queue.
  - Verification: the offline top-k re-score matches the earlier GPU run within 0.004 logits; zero-hook contrast logits match the earlier `vcd_zero` runs exactly; ViewLayer, recomputed independently, matches the report's .781 / .817.

## Contribution line (Akshay)

> Zero- and shuffle-hook contrast branches; JSD, entropy and gradient layer routing; per-layer logging pipeline; linear-probe analysis of ViewLayer; clean-teacher (label-free) probe.
