# ACIC hardening digest

## Accuracy (median relative RMSE per DGP setting)

| estimator | acic_s1 | acic_s2 | acic_s3 | acic_s4 | acic_s5 | acic_s6 | mean_exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dm | 0.019 | 0.073 | 0.013 | 0.071 | 0.016 | 0.026 | 0.036 |
| dr | 0.023 | 0.09 | 0.015 | 0.081 | 0.016 | 0.029 | 0.042 |
| switch_dr | 0.023 | 0.09 | 0.015 | 0.081 | 0.016 | 0.029 | 0.042 |
| snips | 0.036 | 0.099 | 0.019 | 0.087 | 0.025 | 0.035 | 0.05 |
| ips | 0.061 | 0.11 | 0.057 | 0.102 | 0.057 | 0.062 | 0.075 |
| bips | 0.061 | 0.108 | 0.057 | 0.099 | 0.057 | 0.061 | 0.074 |

Best mean-across-settings accuracy: **dm**


Mean |true value| across cells: 2.1717


## Diagnostics vs IPS error (Spearman)

| diagnostic | spearman_rho | p_value |
| --- | --- | --- |
| diag_ess_fraction | -0.167 | 0.0 |
| diag_support_deficiency | 0.105 | 0.0 |
| diag_max_weight | 0.104 | 0.0 |