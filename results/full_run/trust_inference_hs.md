# Trust-map clustered inference

## Spearman rho with cluster-bootstrap 95% CIs (main run)

| diagnostic | rho | CI (dataset clusters) | CI (dataset x seed) |
| --- | --- | --- | --- |
| diag_ess_fraction | -0.477 | [-0.535, -0.412] | [-0.523, -0.431] |
| diag_support_deficiency | +0.463 | [+0.439, +0.491] | [+0.438, +0.488] |
| diag_max_weight | +0.430 | [+0.345, +0.502] | [+0.377, +0.484] |

## Within-dataset rho (cross-dataset confounds removed)

```
           diag_ess_fraction  diag_support_deficiency  diag_max_weight
dataset                                                               
hillstrom             -0.571                    0.509            0.460
ihdp                  -0.487                    0.435            0.466
jobs                  -0.498                    0.462            0.266
lenta                 -0.443                    0.496            0.497
synthetic             -0.383                    0.450            0.394
```

## Leave-one-dataset-out pooled rho

```
           diag_ess_fraction  diag_support_deficiency  diag_max_weight
synthetic             -0.506                    0.467            0.435
hillstrom             -0.448                    0.452            0.442
lenta                 -0.487                    0.457            0.393
ihdp                  -0.468                    0.474            0.412
jobs                  -0.475                    0.466            0.463
```

## Threshold rule (fit on main run, validated out-of-DGP)

Rule (delta=0.1): flag when ESS fraction < 0.482 OR support deficiency > 0.008

Held-out run: results/acic_hillstrom_run

- n_cells: 1080
- base_error_rate: 0.268
- flag_rate: 0.268
- error_rate_flagged: 0.581
- error_rate_unflagged: 0.153
- recall_of_errors: 0.581
- auc_diag_ess_fraction: 0.864
- auc_diag_support_deficiency: 0.679
- auc_diag_max_weight: 0.812
