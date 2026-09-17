# P0c historical-utility predictability screen contract

Version: `p0c-v1`  
Status: frozen before analysis  
Parent P0 hash: `b2ace898d038`  
Prerequisite P0b hash: `bfd5bae251a5`  
Canonical P0c configuration hash: `00293b538504`

## Decision question

P0c asks whether strictly earlier calibration outcomes contain enough signal to
justify freezing a deployable selector and opening the already defined sealed
origins. It is an analysis-only screen over integrity-checked P0b artifacts: it
does not run either forecasting model and does not instantiate any sealed origin.

P0c is not the final H3 test from the parent P0 contract. Passing P0c authorizes
freezing that test; it does not upgrade a paper claim.

## Prequential score and label

The evaluation unit is `(backbone, task, item, target, origin, policy_id)`.
`target_only` is the comparator and is excluded as a candidate label. For every
other deduplicated policy evaluated at a calibration origin, relative utility is

`(target_only_SQL - candidate_SQL) / max(abs(target_only_SQL), 1e-12)`.

At current origin `j`, the score for the same `(task, item, target, policy_id)` is
the exponentially weighted mean of finite utilities from origins `i < j`, with
weight `0.5 ** ((j - i) / 3)`. At least one finite prior observation is required.
The binary label is one exactly when current utility is strictly positive.

The origin index is only an ordering and decay coordinate. Current-origin losses,
later calibration outcomes, sealed outcomes, task identity, policy identity, and
availability metadata are not inputs to the score. Policies deduplicated in P0b
are evaluated under the retained `policy_id`; aliases are not re-expanded because
that would duplicate identical predictions.

## Aggregation and gate

Compute a tie-aware binary AUROC by pooling eligible policy-units within each
task. A task is valid only with at least 20 scored rows and both label classes.
The primary statistic is the equal-weight mean of the four task AUROCs; pooled
unit-micro AUROC is mandatory secondary evidence.

Each backbone passes when its four-task macro AUROC is at least 0.65. P0c advances
only when both backbones pass. Missing or single-class tasks fail the backbone
rather than being silently removed from the macro average.

These are screening-only point estimates. No bootstrap interval, learned model,
policy threshold, or paper-level result is produced.

## Transition rule

- **Pass:** freeze the selector and full sealed P0 evaluation protocol before
  reading any sealed outcome.
- **Fail:** stop or explicitly reframe the direction without accessing sealed
  evaluation origins.

P0c outputs a row-level score CSV and an integrity-linked JSON decision report in
the private Google Drive manifest tree.
