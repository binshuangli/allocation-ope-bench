.PHONY: install repro repro-smoke repro-medium repro-full repro-optbias analyze smoke test test-fast lint fmt clean

PY := ~/.pyenv/versions/py312/bin/python
RESULTS_DIR := results
DATA_DIR := data

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"
	pre-commit install

# ── Reproduction ──────────────────────────────────────────────────────────────

repro: install test repro-medium
	@echo "=== allocation-ope-bench: medium reproduction complete ==="

repro-smoke: install test smoke
	@echo "=== Smoke reproduction complete ==="

smoke:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=smoke \
		smoke=true

repro-medium:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=medium_run \
		n_jobs=4 \
		"datasets=[{name: synthetic},{name: hillstrom},{name: ihdp}]"

# x5 is omitted pending a proper feature loader (its raw bunch needs a
# purchases-table aggregation / categorical encoding — see load_x5 docstring).
repro-full:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=full_run \
		n_jobs=4 \
		results_dir=$(RESULTS_DIR)/full_run \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# Hardening on a LARGER real covariate set (Hillstrom, n=10k fixed subsample).
repro-acic-hillstrom:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=acic_run \
		n_jobs=4 \
		results_dir=$(RESULTS_DIR)/acic_hillstrom_run \
		"datasets=[{name: acic, setting: 1, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 2, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 3, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 4, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 5, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 6, covariates: hillstrom, n_rows: 10000}]"

repro-optbias:
	$(PY) -m allocation_ope_bench.experiments.optimization_bias \
		experiment=opt_bias \
		n_jobs=4 \
		results_dir=$(RESULTS_DIR)/full_run \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# RQ3 extension: optimizer's-curse comparison on the six ACIC exact-value DGPs
# (shows the honest-splitting reduction is not specific to synthetic + IHDP).
repro-optbias-acic:
	$(PY) -m allocation_ope_bench.experiments.optimization_bias \
		experiment=opt_bias \
		n_jobs=4 \
		results_dir=$(RESULTS_DIR)/optbias_acic \
		"datasets=[{name: acic, setting: 1},{name: acic, setting: 2},{name: acic, setting: 3},{name: acic, setting: 4},{name: acic, setting: 5},{name: acic, setting: 6}]"

# Known-effect hardening: six ACIC-2017-style DGPs on real IHDP covariates
# (1,080 config cells; conf/experiment/acic_run.yaml).
repro-acic:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=acic_run \
		n_jobs=4 \
		results_dir=$(RESULTS_DIR)/acic_run \
		"datasets=[{name: acic, setting: 1},{name: acic, setting: 2},{name: acic, setting: 3},{name: acic, setting: 4},{name: acic, setting: 5},{name: acic, setting: 6}]"

# Reference-dependence check: shared vs disjoint RCT reference split, ALL three RCTs
# (Appendix F). Jobs matters most — it is the one dataset where IPS is nominally best
# in Table 2, and that win does not survive this design.
repro-rct-disjoint:
	$(PY) -m allocation_ope_bench.experiments.reference_dependence \
		experiment=refdep_run \
		max_n=50000 \
		results_dir=$(RESULTS_DIR)/refdep_run \
		"datasets=[{name: hillstrom},{name: lenta},{name: jobs}]"

# RQ4 under a COMMON logger: every candidate scored on ONE shared logged dataset.
# The main runner logs from each candidate's own score, which conflates candidate
# quality with the candidate-specific data-collection process. Two logger regimes
# (aligned with one candidate / candidate-independent).
repro-commonlog:
	$(PY) -m allocation_ope_bench.experiments.common_logger_selection \
		experiment=selection_run \
		max_n=50000 \
		results_dir=$(RESULTS_DIR)/commonlog_run \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# RQ1 robustness: out-of-fold vs in-sample outcome nuisance (Section 5.1). Checks that
# the model-based-over-IPS gap does not depend on mu-hat being fit in-sample, and that
# in-sample fitting is not what produces the DM-DR null.
repro-nuisance-crossfit:
	$(PY) -m allocation_ope_bench.experiments.nuisance_crossfit \
		experiment=selection_run \
		results_dir=$(RESULTS_DIR)/nuisance_crossfit \
		"datasets=[{name: synthetic},{name: ihdp},{name: acic, setting: 3},{name: acic, setting: 3, covariates: hillstrom, n_rows: 10000}]" \
		"experiment.candidate_policies=[t_learner,s_learner]"

# Is "tail control never helps" a fact about OPE or an artifact of the 0.02 propensity
# floor? Sweeps eps in {0.02, 0.005, 0.001, 0.0002} (weight ceilings 50 -> 5000) with
# everything else held fixed, and re-runs clipping and shrinkage-DR at each.
repro-floor-sensitivity:
	$(PY) -m allocation_ope_bench.experiments.floor_sensitivity \
		experiment=full_run \
		results_dir=$(RESULTS_DIR)/floor_sensitivity \
		"datasets=[{name: synthetic},{name: ihdp}]"

# Expanded-slate selection run (7 candidate policies; RQ4 hardening).
repro-selection:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=selection_run \
		n_jobs=4 \
		results_dir=$(RESULTS_DIR)/selection_run \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# Outcome-model misspecification sweep: one sub-run per mu-hat quality rung.
MISSPEC_DATASETS := "datasets=[{name: synthetic},{name: acic, setting: 3},{name: acic, setting: 5}]"
repro-misspec:
	for om in lightgbm stump ridge; do \
		$(PY) -m allocation_ope_bench.experiments.runner \
			experiment=misspec_run \
			experiment.outcome_model=$$om \
			n_jobs=4 \
			results_dir=$(RESULTS_DIR)/misspec_run/$$om \
			$(MISSPEC_DATASETS); \
	done

# WP6 — turn result parquets into figures + LaTeX tables + a digest.
analyze:
	$(PY) -m allocation_ope_bench.analysis.run --results-dir $(RESULTS_DIR)/full_run

analyze-acic:
	$(PY) -m allocation_ope_bench.analysis.run --results-dir $(RESULTS_DIR)/acic_run --hardening

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not network and not slow"

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/
	black --check src/ tests/

fmt:
	ruff check --fix src/ tests/
	black src/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

# RQ4 common-logger comparison on the 3-candidate slate (Table rq4-logger).
repro-commonlog-3cand:
	$(PY) -m allocation_ope_bench.experiments.common_logger_selection \
		experiment=full_run \
		max_n=50000 \
		results_dir=$(RESULTS_DIR)/commonlog_3cand \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# Which hyper-parameters do the two tuned hybrids select? Replays the main-run grid
# and records Switch-DR's tau and mIPS's alpha, so the near-duplication of DR and IPS
# is reported as a property of the selection rules (Section 5.1).
repro-tuner-selection:
	$(PY) -m allocation_ope_bench.experiments.tuner_selection \
		experiment=full_run \
		results_dir=$(RESULTS_DIR)/tuner_selection \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# RQ1 addendum: does weight clipping rescue IPS? (Section 5.1). Clipped IPS at
# M in {5,10,50} on the exact-value datasets; the answer is essentially no, because
# the propensity floor already bounds weights and the pathology is ESS collapse.
repro-clipped-ips:
	$(PY) -m allocation_ope_bench.experiments.nuisance_crossfit \
		experiment=selection_run \
		results_dir=$(RESULTS_DIR)/clipped_ips \
		"datasets=[{name: synthetic},{name: ihdp},{name: acic, setting: 3}]" \
		"experiment.candidate_policies=[t_learner,s_learner]"

# RQ2 (rebuilt): logger-target ALIGNMENT as an explicit axis, crossed with temperature.
# The self-aligned main sweep cannot sweep overlap -- a deterministic top-k target is best
# covered by a logger that mimics it -- so temperature only bites under misalignment.
# Three regimes: self_aligned / misaligned (the other learned candidate) / independent.
repro-logger-alignment:
	$(PY) -m allocation_ope_bench.experiments.logger_alignment \
		experiment=full_run \
		results_dir=$(RESULTS_DIR)/logger_alignment \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# Held-out validation for the rebuilt RQ2 screen: same alignment axis on the two
# known-effect hardening suites (ACIC-style DGPs on IHDP and Hillstrom covariates).
repro-logger-alignment-acic:
	$(PY) -m allocation_ope_bench.experiments.logger_alignment \
		experiment=acic_run \
		results_dir=$(RESULTS_DIR)/logger_alignment_acic \
		"datasets=[{name: acic, setting: 1},{name: acic, setting: 2},{name: acic, setting: 3},{name: acic, setting: 4},{name: acic, setting: 5},{name: acic, setting: 6}]"

repro-logger-alignment-acic-hillstrom:
	$(PY) -m allocation_ope_bench.experiments.logger_alignment \
		experiment=acic_run \
		results_dir=$(RESULTS_DIR)/logger_alignment_acic_hillstrom \
		"datasets=[{name: acic, setting: 1, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 2, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 3, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 4, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 5, covariates: hillstrom, n_rows: 10000},{name: acic, setting: 6, covariates: hillstrom, n_rows: 10000}]"

# Anonymous submission archive: the paper source WITHOUT paper/author.tex, which is the
# only file carrying identifying details (main.tex falls back to an anonymous block via
# \IfFileExists). Verifies the result is clean before writing the tarball.
submission-archive:
	@test -s .anon-patterns || { echo "REFUSING: .anon-patterns not configured"; exit 1; }
	@rm -rf build/anon && mkdir -p build/anon/paper
	@cp paper_compact_readable/main.tex paper_compact_readable/refs.bib paper_compact_readable/tmlr.sty paper_compact_readable/tmlr.bst paper_compact_readable/fancyhdr.sty build/anon/paper/
	@cp -RL paper_compact_readable/tables paper_compact_readable/figures build/anon/paper/
	@if grep -rlniE "$$(cat .anon-patterns)" build/anon/ >/dev/null 2>&1; then \
		echo "REFUSING: identifying strings found in the archive:"; \
		grep -rlniE "$$(cat .anon-patterns)" build/anon/; exit 1; fi
	@cd build && tar czf ../submission-anon.tar.gz anon && cd .. && rm -rf build/anon
	@echo "wrote submission-anon.tar.gz (verified free of identifying strings)"

# Build-quality gate. Covers every warning class in one place so the check cannot be
# narrowed by accident (a shortened ad-hoc grep once let a multiply-defined label ship).
check-paper:
	$(PY) scripts/check_paper.py paper
	$(PY) scripts/check_paper.py paper_compact --max-body-pages 12 --max-abstract-words 340

# The propensity-estimation sweep: replaces the exact pi_b with an out-of-fold estimate
# (LightGBM / logistic / marginal) and re-runs both the accuracy comparison and the
# fragility screen under it. This is the assumption the benchmark was most favorable on.
repro-propensity:
	$(PY) -m allocation_ope_bench.experiments.propensity_estimation \
		experiment=full_run \
		results_dir=$(RESULTS_DIR)/propensity_estimation \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# Anonymous CODE supplement backing the paper's reproducibility claim: source, configs,
# tests, lockfile, Makefile and the raw result parquets (7.8M), with the identifying
# files (CITATION.cff, README.md, author.tex) excluded and a scrub gate that refuses to
# write the tarball if any identifying string survives. check_paper.py is excluded only
# because it contains the scrub pattern itself.
submission-supplement:
	@test -s .anon-patterns || { echo "REFUSING: .anon-patterns not configured"; exit 1; }
	@rm -rf build/supp && mkdir -p build/supp
	@cp -R src conf tests scripts results build/supp/
	@cp Makefile pyproject.toml build/supp/
	@grep -v "^-e " requirements.lock > build/supp/requirements.lock
	@sed -E 's/Copyright \(c\) ([0-9]+).*/Copyright (c) \1 Anonymous Authors (identity withheld for double-blind review)/' LICENSE > build/supp/LICENSE
	@find build/supp \( -name "__pycache__" -o -name "*.egg-info" \) -prune -exec rm -rf {} + 2>/dev/null; true
	@printf '%s\n' "Anonymous supplement for TMLR submission." \
		"Install: pip install -e .   Smoke test: make smoke" \
		"Every reproduction target is listed in the paper's Appendix B; results/ holds" \
		"the raw parquets each figure and table regenerates from (make analyze)." > build/supp/README_ANON.md
	@python3 scripts/strip_packaging_targets.py build/supp/Makefile
	@if grep -rlIiE "$$(cat .anon-patterns)" build/supp/ ; then \
		echo "REFUSING: identifying strings found above"; exit 1; fi
	@cd build && tar czf ../supplement-anon.tar.gz supp && cd .. && rm -rf build/supp
	@echo "wrote supplement-anon.tar.gz (code+configs+tests+parquets, scrub-verified)"

# Tail control where the tail exists: clipping (M in {5,10,50}) and shrinkage-DR under all
# three logger-alignment regimes on the exact-value datasets. Closes the two gaps the
# paper previously stated as untested: clipping under the independent logger, and the
# cited-but-unrun shrinkage-DR baseline.
repro-tail-control:
	$(PY) -m allocation_ope_bench.experiments.tail_control \
		experiment=full_run \
		results_dir=$(RESULTS_DIR)/tail_control \
		"datasets=[{name: synthetic},{name: ihdp},{name: acic, setting: 3}]"

# Score-aligned vs action-aligned logging under extreme sharpening (tau down to 0.05).
# Verifies the CORRECTED overlap limit on benchmark data: sharpening a score-aligned
# logger eventually collapses ESS through the mean-to-cutoff band, while an
# action-aligned logger (centered at the top-k cutoff) sharpens toward full support.
repro-sharpening-limit:
	$(PY) -m allocation_ope_bench.experiments.sharpening_limit \
		experiment=full_run \
		results_dir=$(RESULTS_DIR)/sharpening_limit \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# The whole RQ1 benchmark re-run with OUT-OF-FOLD outcome nuisances, so the
# model-based estimators never see a unit's own outcome. Reviewers asked whether
# the model-based advantage is a default or a reduced robustness check.
repro-full-oof:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=full_run \
		n_jobs=4 \
		+nuisance=out_of_fold \
		results_dir=$(RESULTS_DIR)/full_run_oof \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# Refit-aware intervals: how much of DM's undercoverage is the frozen nuisance?
# Refits mu-hat inside every bootstrap resample on a reduced grid.
repro-refit-intervals:
	$(PY) -m allocation_ope_bench.experiments.refit_intervals \
		experiment=full_run \
		results_dir=$(RESULTS_DIR)/refit_intervals \
		"datasets=[{name: synthetic},{name: ihdp}]"

# Policy/nuisance INDEPENDENCE ablation for RQ3: policy from an in-sample LightGBM
# tau-hat, DR nuisance from a different model class, so cross-fitting the nuisance no
# longer cross-fits the object the policy was built from.
repro-optbias-decoupled:
	$(PY) -m allocation_ope_bench.experiments.optbias_decoupled \
		experiment=opt_bias \
		results_dir=$(RESULTS_DIR)/optbias_decoupled \
		"datasets=[{name: synthetic},{name: ihdp},{name: acic, setting: 1},{name: acic, setting: 3},{name: acic, setting: 5}]"

# Appendix L names the mean-centred logistic as a design defect: it leaves the logger
# aligned with the candidate's score RANKING rather than its action boundary, and makes
# tau incomparable across budgets. This re-runs the whole accuracy sweep with the logistic
# centred at each budget's top-k cutoff, so the logger is genuinely action-aligned.
repro-full-cutoff:
	$(PY) -m allocation_ope_bench.experiments.runner \
		experiment=full_run \
		n_jobs=4 \
		+logger_center=cutoff \
		results_dir=$(RESULTS_DIR)/full_run_cutoff \
		"datasets=[{name: synthetic},{name: hillstrom},{name: lenta},{name: ihdp},{name: jobs}]"

# Recompute headline numbers from results/ and assert they appear in both manuscripts.
# Two review rounds found stale values that survived a claimed recomputation pass; this
# puts the recomputation in the repo so drift fails the build instead of shipping.
check-numbers:
	$(PY) scripts/check_numbers.py

# Appendix I: score perturbation-DR against its own matched (smoothed-policy)
# reference V_pert,ref = (1/M) sum_m V_exact(z_m), replicating the estimator's exact
# perturbation draws on the exact-value datasets. Estimand-coherent accuracy check.
repro-perturbation-matched:
	$(PY) -m allocation_ope_bench.experiments.perturbation_matched

# Twins external validation: a NON-SIMULATED PAIRED reference. Both co-twins'
# outcomes are recorded, so the reference is read off data rather than a surface we
# fit -- but the twins are different individuals, so this is a matched-pair contrast,
# not two potential outcomes for one unit (see the paper's Twins appendix).
# Tests whether the exact-value conclusions are mechanism or artifact.
repro-twins:
	$(PY) -m allocation_ope_bench.experiments.runner experiment=full_run n_jobs=4 \
		results_dir=$(RESULTS_DIR)/twins_run "datasets=[{name: twins}]"
	$(PY) -m allocation_ope_bench.experiments.logger_alignment experiment=full_run \
		results_dir=$(RESULTS_DIR)/logger_alignment_twins "datasets=[{name: twins}]"
	$(PY) -m allocation_ope_bench.experiments.optimization_bias experiment=opt_bias n_jobs=4 \
		results_dir=$(RESULTS_DIR)/twins_optbias "datasets=[{name: twins}]"

# Numerical cross-check against Open Bandit Pipeline: our IPS/SNIPS/DM/DR vs
# OBP's reference implementations on identical inputs (same logged data, same
# fitted mu-hat). Closed-form estimators must agree to floating point.
# Requires: pip install obp
reference-check:
	$(PY) scripts/reference_check.py
