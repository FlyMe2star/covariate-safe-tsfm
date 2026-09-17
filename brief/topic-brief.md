# Topic brief

Target: CCF B-class empirical ML/data-mining conference; exact venue will be
selected after P1b paper-eligible evidence is available.

Assumed length: 8–10 pages of main text, excluding references and appendix.

## Problem

New time-series foundation models can consume multiple target variables and past/future covariates natively. In practice, users still lack an evidence-based answer to a prior question: **which available variables should be supplied for a particular target and forecast origin?** More information is not guaranteed to help because covariate relevance, lag, noise, and distribution shift vary by series and time.

## Research question

Can strictly historical forecast evidence certify the task–backbone settings in which
dynamic covariate-policy routing is trustworthy for a frozen zero-shot TSFM?

## Scope

- Forecasting only; no FER code or data is reused.
- Frozen zero-shot inference; no fine-tuning in the core paper claim.
- Numeric and categorical covariates already exposed by public FEV tasks.
- Native covariate-aware backbones: Chronos-2 and TimesFM 3.
- Rolling-origin evaluation with a temporal calibration/evaluation boundary.
- Primary metric: scaled quantile loss (SQL), matching FEV task definitions.

## Intended contribution

1. A diagnostic showing when and how often covariates hurt native zero-shot TSFMs.
2. A leakage-safe applicability certificate that enables dynamic policy routing only
   for task–backbone groups with statistically reliable calibration evidence.
3. A safe fallback to a calibration-selected constant policy when applicability is not
   certified, plus cross-backbone and cross-domain failure analysis.

## Non-claims

- No claim that covariates are generally harmful.
- No causal interpretation of selected variables.
- No claim of a new foundation model or a universally optimal feature selector.
- No model arbitration, whole-forecast abstention, or causal-variable selection claim.
- No paper claim from P0/P1a screening artifacts; sealed evidence requires a separately
  frozen router and evaluation contract.

## Constraints

- Three-to-four-month execution window.
- Colab-class GPUs first; H100/A100 only after P0 passes.
- Public data and reproducible model checkpoints.
- Results must be separable into smoke, screening, and paper-eligible artifacts.
