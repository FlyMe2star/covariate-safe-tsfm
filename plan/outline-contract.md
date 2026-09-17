# Outline contract

Status: P1a passed; P1b method and evaluation design awaiting owner approval

## Provisional story

1. **Problem** — native covariate support creates a hidden input-policy decision, but
   routing reliability is not uniform across models and tasks.
2. **Diagnosis** — quantify covariate harm, finite policy headroom, and the observed
   task–backbone heterogeneity without test leakage.
3. **Certificate** — use calibration-only clustered uncertainty to decide where dynamic
   routing is applicable.
4. **Method** — enable a frozen historical-utility router only in certified groups;
   otherwise use a calibration-selected constant policy and always forecast.
5. **Evidence** — compare gated routing with constant policies and the ungated router on
   untouched origins, emphasizing paired effects and worst-group downside.

## Required main evidence

- Figure 1: distribution of all-covariate utility versus target-only by task/backbone.
- Figure 2: group-level predictability estimates and calibration-only confidence bounds.
- Table 1: constant policies, ungated router, applicability-gated router, and oracle diagnostic.
- Figure 3: routing coverage versus paired loss/downside; coverage means dynamic routing,
  not forecast coverage.
- Table 2: per-task/backbone transfer and worst-group results, including every fallback.
- Appendix: task schemas, exact candidate policies, leakage audit, compute, licenses, and bootstrap details.

## Citation and figure quotas

- Introduction and problem framing: 6–8 verified citations.
- Related work: 16–22 verified citations across TSFMs, covariate adaptation,
  configuration selection, and selective prediction.
- Method/evaluation motivation: 6–10 verified citations.
- Target total: 30–40 verified references, one method diagram, at least two result
  figures, and at least two main result tables.

## Drafting rule

No abstract, claimed contribution, or result paragraph is written in factual tense until its evidence row is `verified`. Failed P0 gates trigger a stop/reframe decision instead of a weaker retrospective threshold.

## P0 outcome

P0a and P0b passed, but P0c failed because Chronos-2 reached only 0.5487
task-macro historical-utility sign AUROC versus the frozen 0.65 threshold.
TimesFM 3 passed at 0.7421, leaving only one of the two required backbones.
The original outline is therefore inactive, and sealed evaluation origins remain
unopened. The replacement story above is authorized only for the CPU-only P1a
confidence audit. A P1a pass still does not authorize sealed access: the router,
fallback, baselines, estimands, and multiplicity plan require a subsequent freeze.

## P1a outcome and proposed P1b evidence

P1a certifies four groups: Chronos-2/Rossmann and TimesFM 3/Rohlik, Rossmann,
and Solar. All other groups must use the frozen constant fallback. The proposed P1b
contract uses two fixed-sequence comparisons: first against the full-candidate
calibration-best constant, then—only after passage—against the ungated version of the
same historical-utility router. Group means receive equal weight; unit-micro results
are secondary. The design is documented in `notes/design/` and remains unexecuted
until owner approval.
