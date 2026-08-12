"""WP5 — Output validation and anomaly detection tests."""

import numpy as np
import pandas as pd

from allocation_ope_bench.experiments.validate import (
    ESS_FRAC_MIN,
    REL_RMSE_FLAG,
    SUPPORT_DEF_MAX,
    print_validation_report,
    validate_estimates,
)


def _make_clean_df(n: int = 10) -> pd.DataFrame:
    """Minimal clean estimates DataFrame — should produce zero flags."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "dataset": ["synthetic"] * n,
            "estimator": ["ips"] * n,
            "seed": list(range(n)),
            "budget_k": [0.3] * n,
            "overlap_temperature": [2.0] * n,
            "candidate_policy": ["t_learner"] * n,
            "true_value": rng.uniform(1.0, 2.0, n),
            "value_hat": rng.uniform(1.0, 2.0, n),
            "ci_low": rng.uniform(0.5, 1.0, n),
            "ci_high": rng.uniform(2.0, 3.0, n),
            "bias": rng.uniform(-0.1, 0.1, n),
            "abs_bias": rng.uniform(0.0, 0.1, n),
            "rel_bias": rng.uniform(-0.1, 0.1, n),
            "diag_ess_fraction": rng.uniform(0.1, 0.9, n),
            "diag_support_deficiency": rng.uniform(0.0, 0.3, n),
            "selection_compatible": [True] * n,
        }
    )


def test_clean_df_produces_no_flags():
    df = _make_clean_df()
    flagged = validate_estimates(df)
    assert flagged.empty, f"Expected no flags, got:\n{flagged[['flag']].value_counts()}"


def test_nonfinite_value_hat_flagged():
    df = _make_clean_df()
    df.loc[2, "value_hat"] = float("nan")
    flagged = validate_estimates(df)
    assert not flagged.empty
    assert flagged["flag"].str.contains("non-finite value_hat").any()


def test_nonfinite_true_value_flagged():
    df = _make_clean_df()
    df.loc[3, "true_value"] = float("inf")
    flagged = validate_estimates(df)
    assert not flagged.empty
    assert flagged["flag"].str.contains("true_value").any()


def test_ci_order_violation_flagged():
    df = _make_clean_df()
    df.loc[0, "ci_low"] = df.loc[0, "value_hat"] + 1.0  # ci_low > value_hat
    flagged = validate_estimates(df)
    assert not flagged.empty
    assert flagged["flag"].str.contains("CI ordering").any()


def test_high_rel_bias_flagged():
    df = _make_clean_df()
    df.loc[1, "rel_bias"] = REL_RMSE_FLAG + 1.0
    flagged = validate_estimates(df)
    assert not flagged.empty
    assert flagged["flag"].str.contains("rel_bias").any()


def test_low_ess_flagged():
    df = _make_clean_df()
    df.loc[4, "diag_ess_fraction"] = ESS_FRAC_MIN / 2
    flagged = validate_estimates(df)
    assert not flagged.empty
    assert flagged["flag"].str.contains("ess_fraction").any()


def test_high_support_deficiency_flagged():
    df = _make_clean_df()
    df.loc[5, "diag_support_deficiency"] = SUPPORT_DEF_MAX + 0.05
    flagged = validate_estimates(df)
    assert not flagged.empty
    assert flagged["flag"].str.contains("support_deficiency").any()


def test_print_validation_report_clean(capsys):
    flagged = pd.DataFrame()
    print_validation_report(flagged)
    captured = capsys.readouterr()
    assert "No anomalies" in captured.out


def test_print_validation_report_flagged(capsys):
    df = _make_clean_df()
    df.loc[0, "value_hat"] = float("nan")
    flagged = validate_estimates(df)
    print_validation_report(flagged)
    captured = capsys.readouterr()
    assert "anomalous" in captured.out
