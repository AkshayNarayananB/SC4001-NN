ViewLayer recomputed on the same TEST questions (views orig+blur1+blur2+blur3+hflip @ layer 20, DEV threshold -13.11) vs the clean-teacher probe. AURC: lower is better.

| Cell | AUROC baseline | AUROC ViewLayer (5 passes) | AUROC Clean-teacher probe L16+24 (no labels)* (1 pass) | Δ method − ViewLayer [95% CI] | F1 ViewLayer | F1 method | AURC ViewLayer | AURC method |
|---|---|---|---|---|---|---|---|---|
| blur s4 random | 0.793 | 0.835 | 0.863 | +0.029 [+0.019, +0.037] | 74.1 | 78.5 | 0.130 | 0.115 |
| blur s4 popular | 0.759 | 0.789 | 0.862 | +0.073 [+0.063, +0.083] | 71.6 | 79.7 | 0.187 | 0.119 |
| blur s4 adversarial | 0.698 | 0.721 | 0.765 | +0.044 [+0.034, +0.054] | 67.1 | 72.4 | 0.256 | 0.221 |
| noise s4 random | 0.784 | 0.868 | 0.832 | -0.035 [-0.046, -0.023] | 78.3 | 75.0 | 0.125 | 0.153 |
| noise s4 popular | 0.740 | 0.821 | 0.834 | +0.014 [+0.001, +0.025] | 75.0 | 76.0 | 0.180 | 0.149 |
| noise s4 adversarial | 0.687 | 0.762 | 0.736 | -0.025 [-0.039, -0.012] | 70.7 | 69.7 | 0.241 | 0.249 |
| **blur s4 mean** | 0.750 | 0.781 | 0.830 | +0.048 | 70.9 | 76.9 | 0.191 | 0.151 |
| **noise s4 mean** | 0.737 | 0.817 | 0.801 | -0.016 | 74.7 | 73.6 | 0.182 | 0.183 |
