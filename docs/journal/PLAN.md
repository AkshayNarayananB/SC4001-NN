# Journal extension: plan and log

Living document. Newest log entries at the bottom.

## Decision (2026-11-25)

- **Target venue:** IEEE Robotics and Automation Letters (RA-L; 6+2 pages, ICRA/IROS presentation option).
  Fallback: TMLR (judged on correctness and evidence, no page limit), using the same experiments.
- **Authors:** Akshay Narayanan Balajee Viswanath (first, corresponding author; does the extension), Arushi Verma,
  Sanjana Yalamanchili (co-authors for the original course study).
- **Thesis:** Under camera and sensor degradation, decoding-time hallucination fixes for VLMs (VCD and relatives)
  only shift the Yes/No threshold. They do not improve grounding. The evidence survives in the hidden state, and a
  single-pass readout trained only on the model's own clean-image answers recovers it at almost no extra compute.
  Every method is reported as accuracy per unit of compute (passes, latency, memory, and INT8/INT4 precision).
- **Contributions:**
  1. A degraded-perception benchmark protocol: sensor-realistic degradations, image-level held-out test,
     AUROC with decision-preserving ties, and a matched-threshold baseline.
  2. Multi-model evidence that contrastive and other decoding fixes recalibrate rather than ground.
  3. The clean-teacher readout, a label-free single-pass probe, plus the readout-vs-input diagnosis
     (blur is a readout failure; heavy noise and low light are input failures, and restoration views help there).
  4. Compute-aware evaluation, run on an edge-class GPU and under weight quantization.

## Roadmap (set 2026-11-25)

Target: IEEE RA-L, submitted with the IROS 2027 presentation option (RA-L+IROS deadline is usually ~1 March; verify).
Hard constraint: ASPIRE access may end ~2026-12-02..09, so every GPU-heavy run happens first (Phase 1).

| Phase | When | Deliverable |
|---|---|---|
| 1. Collect (GPU) | now -> ~Dec 4 | all forward passes logged for 3 models: baseline/VCD/layers/hidden (running); views; extra baselines; INT8/INT4; real-degradation set; CHAIR subset |
| 2. Analyse (CPU, local) | ~Nov 30 -> Dec 14 | every table and figure from logged data; probes; transfer tests; compute Pareto; VCD decomposition |
| 3. Write | Dec 14 -> Jan 10 | RA-L draft (6+2 pages) + supplement; co-author review |
| 4. Harden | Jan 10 -> Feb | internal review, figure polish, anonymised code + data release, submit |

### Phase 1 runs (priority order)
1. [running] Baseline + VCD + logit lens + hidden states: LLaVA, Qwen2.5-VL, InstructBLIP x TEST(21 conditions) + DEV(21).
2. Views (blur 1/2/3, hflip, DAE) for ViewLayer and probe x views: LLaVA on all 21 conditions; Qwen/InstructBLIP on clean + s2 + s4.
   Needs batched multi-view passes in run_readout first.
3. Missing baselines that reviewers will ask for, all first-token computable: ICD (instruction-disturbance contrast),
   VDD/blank-image contrast, PAI (image-attention amplification, one hooked pass). M3ID / OPERA only if time allows.
4. Quantisation: LLaVA and Qwen in INT8 and INT4 on TEST (all conditions, 1000 images); latency and memory benchmarks on A100 and RTX 4500.
5. Real degradations (key for RA-L): ExDark (real low-light photos with boxes for 12 COCO classes) -> POPE-style questions;
   optionally ACDC (night/fog/rain driving). No tuning on these: pure transfer test.
6. Stretch: CHAIR captioning (500 images x clean/s2/s4 x baseline/VCD) + per-token probe.

### Phase 2 analyses
- VCD recalibration across models and degradations (AUROC, matched threshold, image-clustered CIs).
- Short theory: decompose VCD's score into a bias shift and a discriminative term; predict when AUROC can rise (needs
  the contrast margin to correlate with the label beyond the real margin) and verify the prediction on all cells.
- Readouts per model: best layer, gold probe, clean-teacher probe; probe x views; DAE view.
- Generalisation: held-out degradation types (train 3 kinds, test 2), held-out object classes, DEV->TEST images,
  synthetic->real (COCO DEV -> ExDark). Teacher without clean pairs: synthetic pairs only.
- Compute: AUROC vs passes / latency / memory / precision Pareto; INT4 effect on readouts.

### Decisions and open items
- Edge hardware (2026-11-25): user can get a Jetson Nano, but only much later. The paper must not depend on it:
  report latency + energy per question (NVML, no root) on A100 and RTX 4500, and add Qwen2.5-VL-3B as the
  edge-sized model (tasks appended on ASPIRE). Jetson numbers (if an Orin Nano 8 GB; an original 4 GB Nano cannot
  run a 7B VLM, only the DAE/probe add-ons) go into the revision.
- Faculty co-author: user is considering it. Main value: recommendation letter, reviewer-aware framing, IROS
  registration/travel funds. Prefer an NTU robotics/embedded-vision faculty member with overlap. Claude drafts the
  outreach email + 1-page summary once the first full tables exist.
- CHAIR stretch: undecided (default: in, if ASPIRE time remains).

## Gaps from the course report, and fixes

| Gap | Fix | Status |
|---|---|---|
| Only 500 images, some TEST peeking | Fresh POPE-style questions on COCO val2014 images disjoint from POPE's 500, never inspected during design | todo |
| Gaussian noise/blur only, extreme levels | Sensor model: Poisson-Gaussian shot/read noise, low light, motion blur, JPEG; realistic severity | todo |
| One model | LLaVA-1.5-7B, Qwen2.5-VL-7B, plus a third model (InstructBLIP or LLaVA-NeXT) | todo |
| Question-level bootstrap | Image-clustered bootstrap | todo |
| Mixed hardware/software stacks | Re-run every reported number on one stack (local RTX 4500 Ada, transformers 5.x) | todo |
| No compute accounting | Latency, memory, passes; INT8/INT4 | todo |
| Probe leakage worries | Held-out object classes, held-out degradation types, cross-dataset (fresh images) | todo |
| Yes/No only | Stretch goal: per-token readout + CHAIR on captions | later |

## Log

**2026-11-25**
- Branch `journal`. Fresh TEST set: `scripts/journal/make_fresh_pope.py` builds 2000 COCO val2014 images disjoint
  from POPE's 500 with POPE's recipe (seed 2026); 12,000 questions per split, 19,704 unique per condition.
  The old 500 images become DEV (probe fitting, all hyperparameters). `POPE_SET=pope|pope_fresh` switches sets.
- Sensor degradations in `src/degradations.py` (`SENSOR_LEVELS`): low light (Poisson shot + 3 e- read noise,
  auto-gain, sRGB), motion blur (seeded-angle line kernel), JPEG. blur/noise outputs verified bit-identical.
- `src/vlm.py`: one adapter for LLaVA-1.5 / Qwen2.5-VL / InstructBLIP (+ int8/int4 via bitsandbytes);
  `scripts/journal/run_readout.py`: one logger (margin, per-layer logit lens, hidden states, VCD branch batched);
  `scripts/journal/protocol.py`: image-clustered bootstrap, decision-preserving AUROC, matched F1;
  `scripts/journal/bench_compute.py`: latency/memory per method and precision. Tests: `scripts/tests/test_journal.py`.
- GPU run 1: level calibration on 600 DEV questions (results/journal/calibration.csv). Chosen levels:
  lowlight 20/10/5/3 e- (~-4/-9/-17/-25 acc), motion 25/40/60/90 px (-4/-8/-13/-17), jpeg q10/5/3/1 (-1/-5/-8/-11).
  References agree with the course pilot (blur sigma 10: -12.7; noise 0.7: -14.5).
- Gate (scripts/journal/check_smoke.py): unbatched logger reproduces the course logs at all 32 layers (max 0.0006 logit).
  Batched real+VCD pass differs only by fp16 kernel noise (mean 0.01, max 0.30 logit at mid layers, 100% decision
  agreement), the same size as A100-vs-RTX noise, so batching is kept (~1.6x faster). Disclose in the paper.
- Infrastructure: local `vcd-sched` (GPU-aware queue; llava_dev started 13:08) and `vcd-pull` (pulls ASPIRE results
  every 15 min). ASPIRE account `akshayna` (project personal-akshayna, ~99.5k SU): env `~/envs/vcdj` pinned to the
  local stack, models + data in /scratch/users/ntu/akshayna; `submit_all.sh` (nohup, login node) waits for bootstrap,
  then submits per-model smoke jobs and the TEST arrays for all 3 models (+ DEV arrays for Qwen/InstructBLIP),
  gated by afterok.

**2026-11-25 (afternoon)**
- ASPIRE bootstrap done (env `~/envs/vcdj` = local stack; LLaVA, Qwen2.5-VL, InstructBLIP in scratch HF cache).
  Smoke OK on A100 for LLaVA and Qwen. InstructBLIP failed: transformers 5.x processor returns 5-D pixel_values
  (1,1,3,224,224); fixed in `src/vlm.py` encode(), re-smoke job 26362970.
- ASPIRE limits: one running job per user per queue; queues routed by GPU count (g1 1, g2 2-3, g3 4, glong >24 h,
  gdev <=2 h). Arrays in g1 ran one condition at a time (LLaVA TEST: 37 min/condition at 8.8 q/s with VCD).
  Replaced by worker-pool jobs (`scripts/journal/aspire/pool.pbs` + `pool_worker.py`, atomic claims + heartbeats)
  in g1 + g2 (3 GPU) + g3 (4 GPU) + glong: up to 9 GPUs at once. Task list: `configs/journal/aspire_tasks.tsv`.
- Local llava_dev: 4.2 q/s with VCD on the RTX 4500 (~20 min/condition, ~7 h total).
- First held-out result (LLaVA, fresh TEST, clean, 12,000 questions per split): baseline AUROC .972/.962/.934
  (rand/pop/adv), VCD (decision-preserving) .968/.957/.927; VCD F1 90.5/88.6/83.7 vs matched-threshold baseline
  91.5/89.5/84.6. The course finding (VCD recalibrates, does not ground) replicates on unseen images.
  The fresh set is easier than POPE's 500 (clean AUROC .972 vs .940 on random): report both, never pool.
