# Outline contract

Status: stopped after failed P0c cross-backbone gate

## Provisional story

1. **Problem** — native covariate support creates a hidden input-selection decision.
2. **Diagnosis** — quantify when all-covariate inference helps or harms across origins and backbones.
3. **Opportunity** — measure finite-oracle headroom and temporal predictability without test leakage.
4. **Method** — use historical utility with reliability-aware abstention.
5. **Evidence** — paired cross-domain, cross-backbone evaluation plus risk-coverage and failure analyses.

## Required main evidence

- Figure 1: distribution of all-covariate utility versus target-only by task/backbone.
- Figure 2: historical utility versus future utility and calibration curves.
- Table 1: constant policies, heuristic subsets, oracle diagnostic, and fixed reliability gate.
- Figure 3: selective risk versus coverage.
- Table 2: cross-backbone and cross-domain transfer, including worst-group results.
- Appendix: task schemas, exact candidate policies, leakage audit, compute, licenses, and bootstrap details.

## Drafting rule

No abstract, claimed contribution, or result paragraph is written in factual tense until its evidence row is `verified`. Failed P0 gates trigger a stop/reframe decision instead of a weaker retrospective threshold.

## P0 outcome

P0a and P0b passed, but P0c failed because Chronos-2 reached only 0.5487
task-macro historical-utility sign AUROC versus the frozen 0.65 threshold.
TimesFM 3 passed at 0.7421, leaving only one of the two required backbones.
The original outline is therefore inactive, and sealed evaluation origins remain
unopened. Any replacement story requires a new approved research contract.
