# Contributing

Thank you for your interest in `allocation-ope-bench`.

## Setup

```bash
git clone <repo-url>
cd allocation-ope-bench
pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
make test           # all offline tests (~30 s)
make test-fast      # excludes slow/network marks
```

All 101 offline tests must pass before opening a PR.

## Code style

```bash
make fmt    # auto-format (ruff + black, line length 100)
make lint   # check only
```

Pre-commit hooks enforce the same checks on every commit.

## Adding a dataset

1. Add a loader function in `src/allocation_ope_bench/data/loaders.py` that
   returns a `Dataset` namedtuple.
2. Implement `true_allocation_value` logic — either exact potential-outcome
   means or a known-propensity IPS oracle on a held-out RCT split.
3. Wire it into `_build_dataset()` in `experiments/runner.py`.
4. Add a row to `paper/tables/datasets.tex` and update `datasets` in
   `conf/config.yaml`.
5. Ensure the loader uses **only public data** — no proprietary datasets.

## Adding an estimator

1. Add an `estimate(logged, policy_scores)` function in
   `src/allocation_ope_bench/estimators/`.
2. Register it in `registry.py` — decide whether it belongs in
   `fixed_target_estimators()` or `optimization_bias_estimators()`.
3. Add the estimator name to `_PALETTE` in `analysis/figures.py`.
4. Add unit tests under `tests/`.

## Public data only

This benchmark is built entirely on public datasets and personal research
resources. Do **not** add code, data, or insight derived from any employer or
proprietary source.

## Pull request checklist

- [ ] `make test` passes (101+ tests, 0 failures)
- [ ] `make lint` clean
- [ ] New dataset/estimator has a unit test
- [ ] `paper/tables/datasets.tex` updated if dataset added
- [ ] No proprietary data or code referenced
