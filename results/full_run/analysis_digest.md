# Analysis digest

## RQ1 — accuracy (median relative RMSE)

| estimator | hillstrom | ihdp | jobs | lenta | synthetic | mean_exact | mean_rct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dm | 0.022 | 0.014 | 0.066 | 0.026 | 0.044 | 0.029 | 0.038 |
| dr | 0.023 | 0.015 | 0.059 | 0.028 | 0.046 | 0.03 | 0.037 |
| switch_dr | 0.023 | 0.015 | 0.058 | 0.028 | 0.046 | 0.03 | 0.037 |
| snips | 0.022 | 0.025 | 0.063 | 0.03 | 0.058 | 0.042 | 0.038 |
| ips | 0.022 | 0.052 | 0.068 | 0.03 | 0.062 | 0.057 | 0.04 |
| bips | 0.022 | 0.051 | 0.068 | 0.03 | 0.062 | 0.057 | 0.04 |

Best mean-across-datasets accuracy: **dm**


## RQ2 — trust: do logged-data diagnostics predict IPS error?

| diagnostic | spearman_rho | p_value |
| --- | --- | --- |
| diag_ess_fraction | -0.149 | 0.0 |
| diag_support_deficiency | 0.187 | 0.0 |
| diag_max_weight | 0.184 | 0.0 |

Spearman ρ of each diagnostic vs IPS |relative bias|. Expect ρ<0 for ESS fraction (more support → less error) and ρ>0 for support deficiency / max weight (more risk → more error).


## RQ3 — optimizer's curse (per dataset)

| dataset | curse_present | dr_abs_bias | dr_rel_bias | dr_signed_rel_bias | cross_fitted_dr_pct_removed | cross_fitted_dr_algo_pct_removed |
| --- | --- | --- | --- | --- | --- | --- |
| hillstrom | False | 0.0 | 0.0 | -0.0 | nan | nan |
| ihdp | True | 0.24 | 0.07 | 0.07 | -18.3 | 68.5 |
| jobs | False | 0.03 | 0.04 | -0.0 | nan | nan |
| lenta | False | 0.01 | 0.04 | -0.04 | nan | nan |
| synthetic | True | 0.18 | 0.24 | 0.24 | -36.03 | 91.66 |

Curse present on: **['ihdp', 'synthetic']** (continuous / known-effect datasets). Pooling across all datasets understates de-biasing — report per dataset.


## RQ4 — selection quality

| estimator | correct_rate | mean_regret_norm | mean_sharpe_k2plus | n |
| --- | --- | --- | --- | --- |
| dm | 0.656 | 0.153 | 11.909 | 750 |
| dr | 0.633 | 0.148 | 11.879 | 750 |
| switch_dr | 0.631 | 0.149 | 11.881 | 750 |
| snips | 0.589 | 0.16 | 11.734 | 750 |
| bips | 0.58 | 0.172 | 11.023 | 750 |
| ips | 0.573 | 0.179 | 11.019 | 750 |