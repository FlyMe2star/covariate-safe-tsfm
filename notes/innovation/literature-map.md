# Preliminary literature map

Search snapshot: 2026-09-16. URLs point to primary project pages or papers. This map is provisional and must be refreshed before manuscript drafting.

## Native covariate-aware foundation models

- **Chronos-2** — native univariate, multivariate, and covariate-informed forecasting. The official API distinguishes `past_covariates` from known `future_covariates`, which is central to the leakage contract.  
  Paper/project: <https://github.com/amazon-science/chronos-forecasting>
- **TimesFM 3** — native multivariate forecasting with past-only and past-and-future covariate channels. The source is Apache-2.0, while model weights have their own usage terms.  
  Project: <https://github.com/google-research/timesfm>

## Covariate adaptation and injection

- **ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables** (2025), arXiv:2503.12107. Adds modular learned blocks for past and future covariates.  
  <https://arxiv.org/abs/2503.12107>
- **CoRA: Covariate-Aware Adaptation of Time Series Foundation Models** (2025), arXiv:2510.12681. Uses a frozen backbone, a Granger Causality Embedding, and trainable conditioning.  
  <https://arxiv.org/abs/2510.12681>
- **CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter** (2026), arXiv:2603.21828. Learns low-rank time-varying and invariant correlation adapters.  
  <https://arxiv.org/abs/2603.21828>

These works motivate covariate utility but increase collision risk if this project drifts toward trainable adapters. The defensible gap is a pre-adaptation reliability decision for models that already accept covariates natively.

## Evaluation infrastructure

- **FEV / fev-bench** supplies reproducible rolling-origin tasks, per-task covariate declarations, metrics, and official model adapters.  
  <https://github.com/autogluon/fev>
- **GIFT-Eval** covers 23 datasets, seven domains, and ten frequencies for zero-shot forecasting, but FEV is preferred for P0 because its task contract explicitly separates known-future and past-only columns.  
  <https://huggingface.co/datasets/Salesforce/GiftEval>

## Open gap to test, not assume

The search found extensive work on *how to inject or adapt to covariates*, and model APIs warn that cross-variate information does not uniformly help. It did not yet establish a paper that jointly studies:

1. per-series-origin harm from native all-covariate zero-shot inference;
2. temporal predictability of that harm using only prior origins;
3. an abstaining selector shared across two native covariate-aware TSFMs.

This is a search-derived hypothesis, not a novelty fact. The claim remains blocked until a broader title/abstract/full-text search and citation-chaining pass are complete.
