# Topic brief

## Problem

New time-series foundation models can consume multiple target variables and past/future covariates natively. In practice, users still lack an evidence-based answer to a prior question: **which available variables should be supplied for a particular target and forecast origin?** More information is not guaranteed to help because covariate relevance, lag, noise, and distribution shift vary by series and time.

## Research question

Can strictly historical forecast evidence identify when native covariate conditioning helps a frozen zero-shot TSFM, and abstain to target-only inference when it is unreliable?

## Scope

- Forecasting only; no FER code or data is reused.
- Frozen zero-shot inference; no fine-tuning in the core paper claim.
- Numeric and categorical covariates already exposed by public FEV tasks.
- Native covariate-aware backbones: Chronos-2 and TimesFM 3.
- Rolling-origin evaluation with a temporal calibration/evaluation boundary.
- Primary metric: scaled quantile loss (SQL), matching FEV task definitions.

## Intended contribution

1. A diagnostic showing when and how often covariates hurt native zero-shot TSFMs.
2. A leakage-safe, model-agnostic reliability gate that chooses among a small predeclared set of covariate policies using only earlier origins.
3. Cross-backbone and cross-domain evidence about transferability, abstention, and failure modes.

## Non-claims

- No claim that covariates are generally harmful.
- No causal interpretation of selected variables.
- No claim of a new foundation model or a universally optimal feature selector.
- No paper claim until all pre-registered P0 gates pass.

## Constraints

- Three-to-four-month execution window.
- Colab-class GPUs first; H100/A100 only after P0 passes.
- Public data and reproducible model checkpoints.
- Results must be separable into smoke, screening, and paper-eligible artifacts.
