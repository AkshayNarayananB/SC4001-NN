ViewLayer recomputed on the same TEST questions (views orig+blur1+blur2+blur3+hflip @ layer 20, DEV threshold -13.11) vs the clean-teacher probe. AURC: lower is better.

| Cell | AUROC baseline | AUROC ViewLayer (5 passes) | AUROC Probe L24* (1 pass) | Δ method − ViewLayer [95% CI] | F1 ViewLayer | F1 method | AURC ViewLayer | AURC method |
|---|---|---|---|---|---|---|---|---|
| blur s4 random | 0.793 | 0.835 | 0.868 | +0.033 [+0.021, +0.044] | 74.1 | 79.0 | 0.130 | 0.098 |
| blur s4 popular | 0.759 | 0.789 | 0.885 | +0.096 [+0.084, +0.109] | 71.6 | 81.6 | 0.187 | 0.088 |
| blur s4 adversarial | 0.698 | 0.721 | 0.790 | +0.069 [+0.056, +0.084] | 67.1 | 73.7 | 0.256 | 0.183 |
| noise s4 random | 0.784 | 0.868 | 0.837 | -0.030 [-0.044, -0.018] | 78.3 | 75.7 | 0.125 | 0.120 |
| noise s4 popular | 0.740 | 0.821 | 0.864 | +0.043 [+0.027, +0.057] | 75.0 | 79.0 | 0.180 | 0.105 |
| noise s4 adversarial | 0.687 | 0.762 | 0.759 | -0.002 [-0.019, +0.015] | 70.7 | 71.3 | 0.241 | 0.207 |
| **blur s4 mean** | 0.750 | 0.781 | 0.848 | +0.066 | 70.9 | 78.1 | 0.191 | 0.123 |
| **noise s4 mean** | 0.737 | 0.817 | 0.820 | +0.003 | 74.7 | 75.3 | 0.182 | 0.144 |
