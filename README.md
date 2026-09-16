# Covariate-Safe TSFM

Research code for **reliability-gated covariate selection in zero-shot time-series foundation models**.

Working paper title:

> When Extra Variables Hurt: Reliability-Gated Covariate Selection for Zero-Shot Time-Series Foundation Models

## Status

The project is in **P0 diagnostic screening**. No paper-level result is claimed yet. The first question is deliberately falsifiable:

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

The TimesFM 3 adapter is the next implementation milestone. Model weights remain governed by their upstream licenses; in particular, TimesFM 3 weights are not covered by this repository's code license.

## Reproducibility rule

Every result must carry the Git commit, configuration hash, model checkpoint revision, dataset fingerprint, runtime metadata, and a status from `{smoke_only, screening_only, paper_eligible}`. Official evaluation artifacts are never used for method revision.

## License

Code in this repository is released under the MIT License. Dataset and model licenses remain with their respective owners.
