# Candidate direction screen

Search snapshot: 2026-09-16. These are research candidates, not novelty claims.

| Candidate | Direct prior-art pressure | Three-month feasibility | Decision |
|---|---:|---:|---|
| Reliability-gated covariate selection for native zero-shot TSFMs | Medium: covariate injection/adapters exist, but the zero-training selection/abstention question is less directly occupied | High | **Selected for P0** |
| Missingness-aware context retrieval for TabPFN | High: recent context sampling and tabular feature-selection benchmarks are close | Medium | Backup only |
| Missing-modality routing for multimodal foundation models | High: mature routing and modality-dropout literature | Low | Reject |
| Cross-domain anomaly detection with frozen foundation features | High: crowded benchmark and adapter space | Medium | Reject |

## Why the selected direction survives the first screen

- It asks a decision question that appears before adaptation: whether a variable should be supplied at all.
- Native covariate support in Chronos-2 and TimesFM 3 makes matched zero-shot comparisons possible.
- FEV exposes rolling windows and known-versus-past-only covariate semantics, enabling a leakage-safe protocol.
- The direction has an inexpensive falsification stage. If harm, headroom, or predictability is absent, the project stops within a week rather than after full training.

## Nearest collision risk

CoRA uses a trainable adapter and Granger-causality embedding to inject covariates into frozen TSFMs. ChronosX also learns modular covariate-injection blocks. Our proposed scope must remain distinct: no learned injection backbone, no causal-variable claim, and evaluation centered on native-model input selection and abstention under zero-shot inference. A later trained gate would require a new novelty audit.
