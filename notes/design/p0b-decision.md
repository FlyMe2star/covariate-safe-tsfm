# P0b decision

Status: passed screening gate on 2026-09-17

Frozen configuration: `bfd5bae251a5`

Both frozen zero-shot backbones crossed the preregistered task-macro mean
finite-oracle headroom threshold of 0.05:

| Backbone | Task-macro headroom | Unit-micro headroom | Decision |
|---|---:|---:|---|
| Chronos-2 | 0.0675 | 0.0480 | pass |
| TimesFM 3 | 0.0627 | 0.0447 | pass |

The result supports only the narrow screening statement that the finite,
predeclared candidate set contains sufficient calibration headroom to justify
testing whether historical covariate behavior predicts future utility. It does
not establish a deployable selector, statistical uncertainty, or a paper-level
improvement claim. Rohlik and Rossmann do not both cross 0.05 within each
backbone, so the outcome must be reported as a four-task equal-weight macro gate.

Chronos-2 ran on a Tesla T4 and TimesFM 3 on an NVIDIA A100-SXM4-40GB. Every
candidate and both constants were matched within each backbone and origin, so
the accuracy gate remains valid; cross-backbone runtime comparisons are invalid.

Per the frozen transition rule, the next step is to freeze a calibration-only
historical utility predictability screen. Sealed evaluation origins remain
uninstantiated until that screen is specified and passed.
