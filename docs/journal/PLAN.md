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
- GPU run 1: level calibration on 600 DEV questions (results/journal/calibration.csv).
