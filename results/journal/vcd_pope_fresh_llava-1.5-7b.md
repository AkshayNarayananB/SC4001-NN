# VCD vs baseline: llava-1.5-7b on pope_fresh

3 cells (1 conditions x splits), 300 image-clustered bootstrap resamples. VCD AUROC uses decision-preserving ties. Mean dAUROC -0.0051; cells with CI above/below zero: AUROC 0/3, F1 vs matched-threshold baseline 0/3.

Per condition, mean of the three splits (F1, yes-ratio, FPR in %):

| condition   |   f1_base |   f1_vcd |   f1_matched |   yes_base |   yes_vcd |   fpr_base |   fpr_vcd |   auroc_base |   auroc_vcd |   d_auroc | auroc_sig_up/down   | f1_vs_matched_up/down   |
|:------------|----------:|---------:|-------------:|-----------:|----------:|-----------:|----------:|-------------:|------------:|----------:|:--------------------|:------------------------|
| clean_s0    |      87.7 |     87.6 |         88.5 |       44.3 |        53 |          6 |      15.9 |        0.956 |       0.951 |   -0.0051 | 0/3                 | 0/3                     |
