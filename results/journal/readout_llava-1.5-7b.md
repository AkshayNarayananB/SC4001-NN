# Readout make-or-break: llava-1.5-7b

DEV conditions used for fitting: clean_s0, blur_s1, blur_s2, blur_s3, blur_s4, noise_s1, noise_s2. Probe layers [16, 24], mid layer 20. Thresholds fit on DEV: base 0.00, vcd 0.00, base_tuned -2.72, layer_mid -11.62, gold -1.12, teacher -1.88

## pope_fresh (mean of 3 splits; F1 in %)

| condition   |   auroc_base |   auroc_vcd |   auroc_layer_mid |   auroc_teacher |   auroc_gold |   f1_base0 |   f1_thr_const |   f1_vcd |   f1_layer_mid |   f1_teacher |   f1_gold |   aurc_thr_const |   aurc_teacher | teacher AUROC up/down   | teacher F1 vs thr_const up/down   |
|:------------|-------------:|------------:|------------------:|----------------:|-------------:|-----------:|---------------:|---------:|---------------:|-------------:|----------:|-----------------:|---------------:|:------------------------|:----------------------------------|
| clean_s0    |        0.956 |       0.951 |             0.958 |           0.963 |        0.966 |     87.657 |         84.178 |   87.597 |         83.607 |       89.054 |    89.57  |            0.072 |          0.031 | 3/0                     | 3/0                               |
| blur_s1     |        0.913 |       0.91  |             0.919 |           0.936 |        0.942 |     78.023 |         82.515 |   82.595 |         82.341 |       85.364 |    85.877 |            0.086 |          0.05  | 3/0                     | 3/0                               |
| blur_s2     |        0.887 |       0.885 |             0.896 |           0.92  |        0.928 |     72.129 |         80.714 |   79.262 |         81.022 |       84.059 |    84.365 |            0.1   |          0.062 | 3/0                     | 3/0                               |
