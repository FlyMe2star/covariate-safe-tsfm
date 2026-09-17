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

## Reframe collision map (2026-09-17)

- **TFMAdapter: Instance-level Adaptation of Time Series Foundation Models for
  Covariates** (CIKM 2025) adapts a pretrained forecaster at the instance level. It is
  close on covariate use, but changes model behavior through adaptation; our proposed
  core leaves the native TSFM frozen and routes only predeclared input policies.
  <https://openreview.net/pdf?id=0VhlfRxLvG>
- **Tune-as-Inference: Amortized Configuration Learning for Time-Series Foundation
  Models** (ICLR 2026 Workshop) learns to rank a model's configurations, such as
  context length and patch size, from statistical meta-features. It creates direct
  pressure against a broad “configuration selector” claim. Our narrower object is a
  semantic covariate-policy decision with an explicit applicability certificate and a
  constant-policy fallback. <https://openreview.net/pdf?id=LycMKa0o0b>
- **Synapse: Adaptive Arbitration of Complementary Expertise in Time-Series Foundation
  Models** (TMLR 2026) arbitrates predictive distributions across multiple TSFMs. Our
  router stays within one fixed backbone and never combines model outputs.
  <https://arxiv.org/abs/2511.05460>
- **Selective Time Series Forecasting via Metalearning** (2026) abstains on difficult
  forecast instances. Our system always emits a forecast; abstention applies only to
  dynamic covariate-policy routing. <https://arxiv.org/abs/2606.23448>
- General selective-regression work formalizes prediction/rejection tradeoffs, but it
  reinforces the need to avoid presenting policy fallback as ordinary whole-prediction
  rejection. <https://proceedings.mlr.press/v162/shah22a.html>

The resulting novelty hypothesis is: **reliability-certified abstention from
covariate-policy routing in frozen native covariate-aware TSFMs, using rolling-origin
utility and a constant-policy fallback**. This remains a provisional positioning, not
a “first” claim.
