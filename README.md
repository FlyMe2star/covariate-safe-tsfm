# Covariate-Safe TSFM

Research code for **applicability-gated covariate-policy routing in zero-shot time-series foundation models**.

Working paper title:

> When Covariate Routing Is Trustworthy: Applicability-Gated Input Selection for Zero-Shot Time-Series Foundation Models

## Status

The original universal cross-backbone direction **stopped at the P0c diagnostic
gate**. It has not been rescued by lowering its threshold. A narrower
task–backbone-applicability hypothesis is now authorized for calibration-only P1a
screening. No paper-level result is claimed. The original question was deliberately
falsifiable:

> Across native covariate-aware TSFMs, do extra covariates harm a meaningful fraction of forecasts, and can strictly historical evidence predict when to include them?

P0 uses frozen, zero-shot inference with Chronos-2 and TimesFM 3 on four public FEV tasks. Evaluation outcomes are sealed until candidate covariate policies have been selected from earlier rolling origins.

## Pre-registered P0 gates

The direction advances only if all four gates pass:

1. Adding all covariates harms at least 20% of series-origins by more than 5% relative loss.
2. A predeclared finite oracle candidate set improves at least 5% over the stronger of target-only and all-covariate inference.
3. Historical covariate utility predicts the sign of future utility with AUROC at least 0.65.
4. Gates 1–3 pass independently for both Chronos-2 and TimesFM 3.

The exact protocol is in [`notes/design/p0-diagnostic-contract.md`](notes/design/p0-diagnostic-contract.md). Thresholds and task definitions are machine-readable in [`configs/diagnostic/covariate_utility_p0.yaml`](configs/diagnostic/covariate_utility_p0.yaml).

Frozen P0 configuration hash: `b2ace898d038`.

## Layout

```text
brief/                 research and evidence contracts
configs/diagnostic/    frozen experiment configuration
notes/design/          protocol, leakage rules, and decisions
notes/innovation/      novelty search and candidate triage
plan/                  empirical-paper route and outline contract
src/covsafe/           model-independent diagnostic code
tests/                 unit tests for metrics and gates
```

## Local checks

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
pytest
python scripts/hash_config.py configs/diagnostic/covariate_utility_p0.yaml
```

## First run

Open [`notebooks/00_p0_data_audit.ipynb`](notebooks/00_p0_data_audit.ipynb) in Colab and use a CPU runtime. The notebook installs the FEV source at the pinned upstream commit, audits all four task schemas and temporal partitions, and emits `outputs/p0/p0_data_audit.json`. Send that final JSON back before the GPU adapters are run.

After that audit passes, run the two matched T4 smoke notebooks:

1. [`notebooks/01_chronos2_smoke.ipynb`](notebooks/01_chronos2_smoke.ipynb)
2. [`notebooks/02_timesfm3_smoke.ipynb`](notebooks/02_timesfm3_smoke.ipynb)

Each runs the same calibration origin through an official FEV path and writes a durable manifest to Google Drive.

Both smoke prerequisites are now verified. The next falsification stage is the
calibration-only P0a harm screen. Run these notebooks in order, preferably on an A100
or H100 (T4 remains valid but slower):

1. [`notebooks/03_chronos2_p0a.ipynb`](notebooks/03_chronos2_p0a.ipynb)
2. [`notebooks/04_timesfm3_p0a.ipynb`](notebooks/04_timesfm3_p0a.ipynb)

P0a has frozen configuration hash `902c15e63f71`. It writes each task-variant unit
directly to Google Drive and safely resumes after disconnects. The exact aggregation
clarification is recorded in
[`notes/design/p0a-harm-screen-contract.md`](notes/design/p0a-harm-screen-contract.md).

P0a passed independently on both backbones. The next sequential screen is P0b, which
tests finite-candidate oracle headroom without opening sealed evaluation origins. Run
Chronos-2 first and run TimesFM 3 only if the Chronos report passes review:

1. [`notebooks/05_chronos2_p0b.ipynb`](notebooks/05_chronos2_p0b.ipynb)
2. [`notebooks/06_timesfm3_p0b.ipynb`](notebooks/06_timesfm3_p0b.ipynb)

Frozen P0b configuration hash: `bfd5bae251a5`. An A100 is recommended because this
stage evaluates many more policies than P0a. P0b integrity-checks P0a as its prerequisite,
then reruns all policies on the current GPU for matched numerical comparison and saves
each task-origin atomically.

P0b passed independently on both backbones. The next screen is CPU-only P0c, which
uses strictly earlier calibration-origin utilities to predict the sign of utility at
later calibration origins. It does not run either forecasting model and does not open
the sealed evaluation origins:

1. [`notebooks/07_p0c_predictability.ipynb`](notebooks/07_p0c_predictability.ipynb)

Frozen P0c configuration hash: `00293b538504`. Both backbones must independently
reach a four-task macro AUROC of at least 0.65 before the selector and sealed P0
evaluation protocol may be frozen.

P0c did not pass: Chronos-2 reached 0.5487 task-macro AUROC and TimesFM 3 reached
0.7421, so only one of two required backbones crossed 0.65. The selector was not
frozen and sealed evaluation origins remain unopened. The original protocol must
not continue by lowering the threshold or changing the task aggregation. Further
work requires an explicit new framing and frozen contract.

That reframe is now frozen as P1a. It asks whether individual `(task, backbone)`
groups can be certified using a point AUROC of at least 0.60 and a clustered-bootstrap
one-sided 95% lower bound strictly above 0.50. Continuation requires at least three
eligible groups and at least one task for each backbone. P1a uses only the existing P0c
score table, performs no model inference, and keeps sealed origins closed:

1. [`notebooks/08_p1a_applicability_audit.ipynb`](notebooks/08_p1a_applicability_audit.ipynb)

Frozen P1a configuration hash: `6fe503a9d399`. Use a CPU runtime. A P1a pass authorizes
only a subsequent router-freeze contract—not sealed evaluation by itself.

Model weights remain governed by their upstream licenses; in particular, TimesFM 3
weights are not covered by this repository's code license.

## Reproducibility rule

Every result must carry the Git commit, configuration hash, model checkpoint revision, dataset fingerprint, runtime metadata, and a status from `{smoke_only, screening_only, paper_eligible}`. Official evaluation artifacts are never used for method revision.

## License

Code in this repository is released under the MIT License. Dataset and model licenses remain with their respective owners.
