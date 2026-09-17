# P0a decision

Status: passed screening gate on 2026-09-17  
Frozen configuration: `902c15e63f71`

Both frozen zero-shot backbones crossed the preregistered task-macro harm-rate
threshold of 0.20:

| Backbone | Task-macro harm | Unit-micro harm | Decision |
|---|---:|---:|---|
| Chronos-2 | 0.2501 | 0.0150 | pass |
| TimesFM 3 | 0.2062 | 0.0052 | pass |

The evidence supports only the narrow screening statement that at least 5% relative
SQL degradation occurs often enough across equally weighted tasks to justify testing
selective covariate policies. It does not support the stronger statement that extra
covariates usually hurt: both unit-micro rates are below 2%, and all-dynamic inference
has positive mean relative SQL gain in both backbones.

Per the frozen transition rule, the project advances to a calibration-only finite
candidate oracle screen (P0b). P0b must quantify whether the predeclared policy set
contains at least 5% task-macro oracle headroom for each backbone. Sealed evaluation
origins remain uninstantiated until that screen is frozen and passed.
