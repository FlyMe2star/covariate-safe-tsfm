# P0c decision

Status: failed cross-backbone screening gate on 2026-09-17

Frozen configuration: `00293b538504`

| Backbone | Task-macro AUROC | Unit-micro AUROC | Decision |
|---|---:|---:|---|
| Chronos-2 | 0.5487 | 0.7952 | fail |
| TimesFM 3 | 0.7421 | 0.8576 | pass |

Only one of two backbones crossed the preregistered task-macro AUROC threshold
of 0.65. The original contribution map requires independent passage on both
backbones, so the cross-backbone historical-utility reliability gate stops here.

Chronos-2 illustrates why the primary statistic was task-macro rather than
unit-micro: Rossmann contributes 75,525 of 76,072 scored rows and reaches 0.7944
AUROC, while EPF, Rohlik, and Solar reach only 0.4913, 0.4821, and 0.4269.
The resulting 0.7952 micro AUROC is therefore not evidence of cross-domain
predictability.

No sealed evaluation origin was instantiated, no forecasting inference was
performed in P0c, and no paper claim is upgraded. Under the frozen transition
rule, the threshold, task weighting, and half-life cannot be altered to rescue
the original hypothesis. Any continuation requires an explicit new framing,
new contribution map, and new frozen protocol before sealed outcomes are read.
