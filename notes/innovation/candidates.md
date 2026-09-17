# Candidate direction screen

Search snapshot: 2026-09-16. These are research candidates, not novelty claims.

| Candidate | Direct prior-art pressure | Three-month feasibility | Decision |
|---|---:|---:|---|
| Reliability-gated covariate selection for native zero-shot TSFMs | Medium: covariate injection/adapters exist, but the zero-training selection/abstention question is less directly occupied | High | **Selected for P0** |
| Missingness-aware context retrieval for TabPFN | High: recent context sampling and tabular feature-selection benchmarks are close | Medium | Backup only |
| Missing-modality routing for multimodal foundation models | High: mature routing and modality-dropout literature | Low | Reject |
| Cross-domain anomaly detection with frozen foundation features | High: crowded benchmark and adapter space | Medium | Reject |

## Post-P0c reframe (2026-09-17)

The original candidate is stopped as a universal cross-backbone selector because its
frozen P0c gate failed. The selected replacement is narrower:

| Candidate | Direct prior-art pressure | Feasibility | Decision |
|---|---:|---:|---|
| Reliability-certified abstention from covariate-policy routing within a fixed native TSFM | Medium-high: configuration selection, adapters, model arbitration, and selective forecasting are adjacent | High: reuses calibration artifacts; no training for P1a | **Advance to P1a only** |

The method may route dynamically only in certified `(task, backbone)` groups. In all
other groups it falls back to a calibration-selected constant input policy and still
produces a forecast. This is not a retrospective relaxation of the failed P0c macro
gate; it is a new heterogeneous-applicability hypothesis with a new frozen contract.

## Why the selected direction survives the first screen

- It asks a decision question that appears before adaptation: whether a variable should be supplied at all.
- Native covariate support in Chronos-2 and TimesFM 3 makes matched zero-shot comparisons possible.
- FEV exposes rolling windows and known-versus-past-only covariate semantics, enabling a leakage-safe protocol.
- The direction has an inexpensive falsification stage. If harm, headroom, or predictability is absent, the project stops within a week rather than after full training.

## Nearest collision risk

CoRA uses a trainable adapter and Granger-causality embedding to inject covariates into frozen TSFMs. ChronosX and TFMAdapter also learn covariate adaptations. Tune-as-Inference selects configurations of one TSFM, Synapse arbitrates among different TSFMs, and selective forecasting abstains from difficult predictions. The defensible scope is therefore exact and narrow: no learned injection backbone, no causal-variable claim, no cross-model arbitration, and no refusal to forecast. We certify whether semantic covariate-policy routing within a fixed native model is safe enough to enable; otherwise a constant input policy is used. A trained gate would require another novelty audit.
