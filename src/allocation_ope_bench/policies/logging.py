"""Construct logged bandit feedback under a synthetic logging policy.

Two construction paths, chosen by what the source dataset records:

1. **RCT rejection sampling** (datasets with a known propensity). An RCT records
   each unit's outcome only under its *randomized* action, so we cannot freely
   reassign actions. We keep unit i with probability proportional to
   ``pi_b(a_i | x_i) / pi_rct(a_i | x_i)``: the retained units keep their real
   (action, outcome) tuples, but the retained sample follows ``pi_b`` — valid
   logged feedback with a known pscore.

2. **Semi-synthetic surface sampling** (datasets with known potential-outcome
   means ``mu0, mu1`` but *no* RCT propensity — e.g. IHDP, whose real treatment
   assignment is confounded). We draw logged actions ``a_i ~ Bernoulli(pi_b(1|x))``
   and the reward from the known response surface, ``Y_i = mu_{a_i}(x_i) + eps``,
   with ``eps`` calibrated to the factual residual std so the reward scale matches
   the real outcome. This is the standard semi-synthetic OPE construction and is
   exactly consistent with the exact-effect oracle in
   :func:`~allocation_ope_bench.data.ground_truth.true_allocation_value`.

In both paths ``temperature`` (softmax sharpness over the uplift score) is the
overlap knob: HIGH temperature => smooth logging => good common support; LOW
temperature => sharp logging => worse common support (the WP4 stress dimension).
``mixture_alpha`` mixes in uniform exploration (pi_b <- (1-a)*pi_b + a*0.5), the
coupon-paper "mixed logging" that BIPS is designed to exploit.
"""

from __future__ import annotations

import numpy as np

from allocation_ope_bench.estimators.base import LoggedData


def _logging_treat_prob(
    eval_dataset,
    score: np.ndarray,
    *,
    temperature: float,
    mixture_alpha: float,
    clip: float,
    rct_fallback: float = 0.5,
    center: float | None = None,
) -> np.ndarray:
    """Logging treat-probability pi_b(1 | x) from the uplift score.

    Softmax over the standardized score; temperature 0 falls back to the known
    RCT propensity if available, else a constant ``rct_fallback``.
    """
    score = np.asarray(score, dtype=float)
    if temperature <= 0.0:
        if eval_dataset.propensity is not None:
            p_treat = eval_dataset.propensity.copy()
        else:
            p_treat = np.full(eval_dataset.n, float(rct_fallback))
    else:
        # ``center=None`` (default) reproduces the historical construction: the logistic
        # is centered at the score MEAN. Passing a raw-score cutoff instead yields an
        # ACTION-aligned logger centered at the target's budget boundary -- the
        # distinction the corrected overlap limit turns on (Appendix mechanics).
        mid = score.mean() if center is None else float(center)
        s = (score - mid) / (score.std() + 1e-12)
        p_treat = 1.0 / (1.0 + np.exp(-s / temperature))
    if mixture_alpha > 0.0:
        p_treat = (1.0 - mixture_alpha) * p_treat + mixture_alpha * 0.5
    return np.clip(p_treat, clip, 1.0 - clip)


def make_logged_data(
    eval_dataset,
    score: np.ndarray,
    *,
    temperature: float = 0.0,
    mixture_alpha: float = 0.0,
    seed: int = 42,
    clip: float = 0.02,
    center: float | None = None,
) -> LoggedData:
    """Build LoggedData under a synthetic logging policy.

    Dispatches on the source dataset: rejection sampling when an RCT propensity
    is known, else semi-synthetic surface sampling when potential-outcome means
    are known. Raises if neither is available.

    Parameters
    ----------
    eval_dataset  : a Dataset to draw logged units from (RCT propensity OR known
                    mu0/mu1).
    score         : per-unit uplift score driving the logging policy, shape (n,).
    temperature   : softmax sharpness; HIGH => smooth/good overlap, LOW => sharp.
    mixture_alpha : uniform-exploration mix in [0, 1) for mixed logging (BIPS).
    clip          : floor/ceiling on logging treat-probability for stability.
    """
    if eval_dataset.propensity is None:
        if eval_dataset.has_ground_truth_effect:
            return _make_logged_from_surfaces(
                eval_dataset,
                score,
                temperature=temperature,
                mixture_alpha=mixture_alpha,
                seed=seed,
                clip=clip,
                center=center,
            )
        raise ValueError(
            "logged-data construction requires either a known RCT propensity or "
            "known potential-outcome means (has_ground_truth_effect)"
        )

    rng = np.random.default_rng(seed)
    n = eval_dataset.n

    p_treat = _logging_treat_prob(
        eval_dataset, score, temperature=temperature, mixture_alpha=mixture_alpha, clip=clip,
        center=center,
    )

    action = eval_dataset.treatment
    pi_b_obs = np.where(action == 1, p_treat, 1.0 - p_treat)  # pi_b(a_i | x_i)
    pi_rct_obs = np.where(action == 1, eval_dataset.propensity, 1.0 - eval_dataset.propensity)

    # Rejection sampling: accept ratio normalized so the max is 1.
    ratio = pi_b_obs / pi_rct_obs
    accept_prob = ratio / ratio.max()
    keep = rng.random(n) < accept_prob
    idx = np.flatnonzero(keep)
    if idx.size == 0:  # degenerate (extreme temperature on tiny data)
        idx = np.arange(n)

    return LoggedData(
        context=eval_dataset.X[idx],
        action=action[idx],
        reward=eval_dataset.outcome[idx],
        pscore=pi_b_obs[idx],
        logging_prob_treat=p_treat[idx],
        name=f"{eval_dataset.name}_logged",
    )


def _make_logged_from_surfaces(
    eval_dataset,
    score: np.ndarray,
    *,
    temperature: float,
    mixture_alpha: float,
    seed: int,
    clip: float,
    center: float | None = None,
) -> LoggedData:
    """Semi-synthetic logged feedback from known response surfaces (mu0, mu1).

    Draws logged actions under the logging policy and rewards from the surfaces
    plus calibrated noise. Uses all units (no rejection sampling needed, since we
    can synthesize the counterfactual reward).
    """
    if eval_dataset.mu0 is None or eval_dataset.mu1 is None:
        raise ValueError("surface sampling requires known mu0 and mu1")

    rng = np.random.default_rng(seed)
    n = eval_dataset.n
    mu0 = np.asarray(eval_dataset.mu0, dtype=float)
    mu1 = np.asarray(eval_dataset.mu1, dtype=float)

    p_treat = _logging_treat_prob(
        eval_dataset, score, temperature=temperature, mixture_alpha=mixture_alpha, clip=clip,
        center=center,
    )

    # Sample logged actions under the logging policy.
    action = (rng.random(n) < p_treat).astype(int)
    pi_b_obs = np.where(action == 1, p_treat, 1.0 - p_treat)

    # Reward from the response surface; noise calibrated to the factual residual
    # so the synthetic reward scale matches the real outcome (if outcomes exist).
    mu_a = np.where(action == 1, mu1, mu0)
    if eval_dataset.outcome is not None:
        factual_mu = np.where(eval_dataset.treatment == 1, mu1, mu0)
        sigma = float(np.std(np.asarray(eval_dataset.outcome, dtype=float) - factual_mu))
    else:
        sigma = 0.0
    reward = mu_a + rng.normal(scale=sigma, size=n)

    return LoggedData(
        context=eval_dataset.X,
        action=action,
        reward=reward,
        pscore=pi_b_obs,
        logging_prob_treat=p_treat,
        name=f"{eval_dataset.name}_logged",
    )
