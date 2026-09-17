# P1a applicability-confidence audit contract

Status: **frozen before confidence intervals and before sealed evaluation**

## Why this is a new contract

P0c rejected the original universal cross-backbone claim: Chronos-2 did not reach the
predeclared four-task macro AUROC of 0.65. P1a does not revise that result or relabel it
as a pass. It tests a narrower question suggested by the observed heterogeneity:

> Can calibration-only evidence certify the task–backbone groups in which dynamic
> covariate-policy routing is sufficiently reliable to be attempted?

The proposed system always produces a forecast. “Abstention” means abstaining from
dynamic input-policy routing and using a calibration-selected constant policy instead.

## Frozen unit and confidence procedure

- Group: one `(backbone, task)` pair; eight groups in total.
- Score and binary utility label: the unchanged P0c prequential rows.
- Resampling unit: `(item_id, target, origin_index)` within a group.
- All policy rows belonging to a sampled unit receive the same bootstrap multiplicity.
- Replicates: 2,000 with seed 42 and deterministic group-specific seed offsets.
- Lower confidence bound: fifth percentile of valid bootstrap AUROCs (one-sided 95%).
- At least 90% of bootstrap replicates must contain both label classes.

A group is applicable only when all conditions hold:

1. at least 20 scored rows and both label classes;
2. point AUROC is at least 0.60;
3. the one-sided 95% lower bound is strictly greater than 0.50.

P1 may continue only if at least three of eight groups are applicable and every
backbone has at least one applicable task. The “three groups” rule requires evidence
beyond a single isolated task per backbone while allowing the central model–task
heterogeneity hypothesis.

## Leakage and evidence boundaries

- P1a reads the already-created P0c calibration score table; it performs no forecasting
  model inference.
- It verifies the P0c report, score checksum, frozen scientific-code checksum, and the
  archived failed P0c decision before computing intervals.
- It must not instantiate, inspect, summarize, or tune against sealed evaluation origins.
- Its output is `screening_only` and cannot support a paper result by itself.
- If P1a passes, the full dynamic router, constant fallback, baselines, and evaluation
  estimands must be frozen in a separate contract before sealed inference.

Machine-readable settings live in
`configs/diagnostic/applicability_gate_p1a.yaml`.
