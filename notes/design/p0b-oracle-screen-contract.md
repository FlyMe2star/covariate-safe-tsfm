# P0b finite-candidate oracle screen contract

Version: `p0b-v1`  
Status: frozen before candidate inference  
Parent P0 hash: `b2ace898d038`  
Prerequisite P0a hash: `902c15e63f71`
Canonical P0b configuration hash: `bfd5bae251a5`

## Decision question

P0b asks whether a finite, deployability-agnostic candidate oracle contains enough
headroom to justify learning a selector. It does not train or evaluate a selector.
P0a artifacts are integrity-checked as the authorization prerequisite, but their losses
are not mixed into P0b. Target-only and all-dynamic policies are rerun beside all
additional policies on the same current GPU. This prevents cross-device floating-point
differences when P0a used T4 and P0b uses A100. Every policy uses the same model
revisions, FEV tasks, origins, quantiles, and per-item SQL definition.

## Frozen candidate set

At every calibration origin the ordered candidate generator creates:

1. target-only and all-dynamic constants;
2. one policy for each single known-future covariate;
3. one policy leaving out each known-future or past-only dynamic covariate;
4. historical-correlation top-1, top-2, and top-4 policies.

Equivalent selected-column sets are evaluated once, with the later policy retained as an
alias. Top-k is clipped to the number of available dynamic variables. Static covariates
remain disabled.

## Historical correlation without leakage

For every origin and dynamic covariate, compute absolute Pearson correlation separately
within each `(item, target)` history using observations strictly before the origin. The
ranking score is the median of valid within-series correlations. Pairs with fewer than
three finite observations or zero variance are excluded. Non-numeric categorical
sequences receive a deterministic sorted-level ordinal encoding within the sequence.
Finite scores rank first in descending order; ties break by ascending column name.

The ranking is global for an origin because FEV model adapters accept one covariate schema
per task. This aggregation choice is a screening limitation and must be disclosed.

## Oracle and aggregation

For each `(task, item, target, origin)`, the constant reference is the lower SQL of
target-only and all-dynamic inference. The finite oracle is the lowest finite SQL across
the unique candidate set, including both constants. Oracle headroom is

`(constant_reference_SQL - oracle_SQL) / max(abs(constant_reference_SQL), 1e-12)`.

Units are excluded only if either constant SQL is non-finite. A non-finite additional
candidate is ignored for that unit. Compute mean headroom within each task, then take the
equal-weight mean across four tasks as the primary statistic. Also report the pooled
unit-micro mean, per-task positive-headroom fraction, candidate coverage, and oracle
winner counts.

Each backbone passes when task-macro mean oracle headroom is at least 0.05. P0b advances
only when both backbones pass. These remain `screening_only` point estimates; no paper
claim or uncertainty claim is created.

## Retention and sealed data

The resume unit is `(backbone, task, origin)`. Each completed origin is atomically written
to Google Drive with the historical ranking, unique policies, aliases, configuration and
scientific-code hashes, Git commit, model revision, runtime, and CSV SHA-256. Each dataset
is downloaded and prepared once per process to reduce Hub requests. Sealed evaluation
origins remain uninstantiated throughout P0b.
