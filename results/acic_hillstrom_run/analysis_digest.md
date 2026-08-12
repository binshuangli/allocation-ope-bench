# ACIC hardening digest

## Accuracy (median relative RMSE per DGP setting)

| estimator | acic_hi_s1 | acic_hi_s2 | acic_hi_s3 | acic_hi_s4 | acic_hi_s5 | acic_hi_s6 | mean_exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dm | 0.002 | 0.019 | 0.001 | 0.003 | 0.002 | 0.006 | 0.006 |
| dr | 0.003 | 0.021 | 0.001 | 0.004 | 0.002 | 0.007 | 0.006 |
| switch_dr | 0.003 | 0.021 | 0.001 | 0.004 | 0.002 | 0.007 | 0.006 |
| snips | 0.008 | 0.021 | 0.005 | 0.007 | 0.005 | 0.01 | 0.009 |
| ips | 0.014 | 0.026 | 0.016 | 0.017 | 0.014 | 0.018 | 0.018 |
| bips | 0.014 | 0.026 | 0.016 | 0.017 | 0.014 | 0.018 | 0.018 |

Best mean-across-settings accuracy: **dm**


Mean |true value| across cells: 2.1683


## Diagnostics vs IPS error (Spearman)

| diagnostic | spearman_rho | p_value |
| --- | --- | --- |
| diag_ess_fraction | -0.217 | 0.0 |
| diag_support_deficiency | 0.147 | 0.0 |
| diag_max_weight | 0.189 | 0.0 |