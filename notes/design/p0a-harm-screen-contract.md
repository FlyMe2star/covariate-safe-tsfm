# P0a all-dynamic harm screen contract

Version: `p0a-v1`  
Status: frozen before the multi-origin sweep  
Parent P0 hash: `b2ace898d038`
Canonical P0a configuration hash: `902c15e63f71`

## Why this stage exists

The two one-origin smokes established only that both native model paths execute and
that their input strategies can produce different forecasts. They are not scientific
evidence. P0a is the cheapest admissible falsification stage: it tests H1 across every
calibration origin before any candidate-subset, oracle, or predictability computation.

## Scope and leakage boundary

- Backbones: frozen zero-shot Chronos-2 and TimesFM 3.
- Tasks: the four FEV tasks and exact schemas in the parent P0 contract.
- Policies: exactly `target_only` and `all_dynamic`.
- Origins: only the earliest 60% calibration partition already recorded by the data audit.
- Evaluation unit: `(task, item, target, origin)`.
- Primary metric: per-item FEV SQL.
- Sealed evaluation origins are not instantiated.
- No candidate subset, oracle, historical-utility feature, or method-selection decision is
  computed in this stage.

## Pre-result aggregation clarification

The parent contract specified a 20% harm-rate gate but did not say how tasks with very
different series counts should be weighted. This document resolves that ambiguity before
the multi-origin sweep and after viewing only implementation smokes.

For each backbone, compute the harm rate separately within each task, then take the
equal-weight mean of the four task rates. This task-macro harm rate is the gate statistic.
Also report the pooled unit-micro harm rate as a secondary diagnostic. Direct pooling is
not the gate because the 1115-series Rossmann task would otherwise dominate the three
smaller tasks. A pair is excluded only when either target-only or all-dynamic SQL is
non-finite; all exclusions are counted by task.

A unit is harmful when

`(target_only_SQL - all_dynamic_SQL) / max(abs(target_only_SQL), 1e-12) <= -0.05`.

Each backbone passes H1 when its task-macro harm rate is at least `0.20`. P0a advances
only when both backbones pass. Point estimates are screening evidence; uncertainty
intervals are deferred to the later paper-eligible protocol.

## Execution and retention

Each `(backbone, task, variant)` is an atomic resume unit written directly to Google
Drive as CSV plus a JSON sidecar. A sidecar records the configuration hash, scientific
code hash, Git commit, checkpoint revision, dataset fingerprint, runtime, and CSV
SHA-256. An existing unit is reused only if its configuration, scientific-code, and CSV
hashes match. The Git commit remains recorded for provenance, while documentation-only
commits do not force expensive recomputation. Partial temporary files are never treated
as completed units.

TimesFM 3 weights remain limited to academic non-commercial use under their upstream
license. Inference-time comparisons are not part of P0a.
