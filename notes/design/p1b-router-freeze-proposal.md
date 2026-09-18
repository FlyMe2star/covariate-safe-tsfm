# P1b router-freeze contract

Status: **owner-approved on 2026-09-17; frozen before router-state construction
and sealed access**

Canonical P1b configuration hash: `151648d511c7`.

## Method

The method has two layers. The P1a map decides whether dynamic routing is applicable
for a `(backbone, task)` group. In an applicable group, each series chooses the
available evaluated policy with the largest positive exponentially weighted
calibration utility; `target_only` has score zero and wins when no candidate has a
positive finite score. Scores use all and only calibration outcomes, retain the P0c
half-life of three origins, and remain fixed throughout sealed evaluation.

In an inapplicable group, the method abstains from dynamic routing and uses one
calibration-best constant logical policy. This policy is selected separately for every
`(backbone, task)` from the complete P0b logical set after deterministic alias
re-expansion. It uses P0b's already frozen symmetric anchor rule: exclude a unit if
target-only or all-dynamic SQL is non-finite. It then minimizes mean SQL among policies
with a recorded row for every raw unit and finite SQL on every anchor-eligible unit.
The selected policy is frozen before evaluation.

## Why scores remain calibration-frozen

No sealed outcome updates a later sealed decision. This prevents overlapping forecast
windows, delayed labels, or offline access patterns from creating a hidden temporal
advantage. It also makes every routing decision reproducible before any sealed model
loss is computed.

## Main comparisons

1. **Full-set calibration-best constant**: tests whether routing adds value beyond a
   strong fixed input policy.
2. **Ungated historical-utility router**: the identical router applied to all groups;
   this isolates the P1a applicability certificate.
3. Target-only, all-dynamic, binary calibration-best constant, and last-origin winner
   provide interpretable anchors. The finite candidate oracle is diagnostic only.

The two central comparisons use a fixed sequence: first test the gated router against
the full-set constant; only if that passes test against the ungated router. Each step
requires a one-sided 95% paired cluster-bootstrap lower bound above zero. This fixed
order controls the family-wise error rate without weakening either individual test.

## Aggregation and safety

Compute paired relative SQL gain within each of eight task–backbone groups, then
equal-weight the eight group means. This prevents Rossmann's sample count from
dominating. Resample `(item, target, origin)` units jointly across all compared
methods for 5,000 replicates. Both backbones must also have nonnegative task-macro
point gain versus the full-set constant.

## Execution boundary

The approved YAML is renamed/frozen, hashed, and used by a
CPU-only notebook to construct and hash the router state from P0b/P1a artifacts. Only
after that state passes review may separate GPU notebooks instantiate the sealed
origins. No sealed result can change the applicability map, policy set, half-life,
fallback, comparisons, or aggregation.

Execution order is fixed as notebooks 09 (CPU state), 10 and 11 (sealed backbone
evaluation), then 12 (CPU fixed-sequence analysis). Each stage is reviewed before the
next stage is authorized.

Implementation clarification on 2026-09-18: the first CPU execution exposed the
known 540 non-finite Rossmann anchor pairs already reported independently by both
P0b backbones. Before router-state materialization and without accessing sealed
origins, the constant selector was aligned exactly with P0b's symmetric finite-pair
rule. No applicability decision, policy candidate, outcome, or success threshold was
changed.
