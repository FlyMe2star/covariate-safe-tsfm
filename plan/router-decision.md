# Router decision

Route: **empirical ML paper**  
Status: accepted for P0 only

The project makes testable claims about model behavior and a deployable selection rule. A review-paper route would not answer the central question. Manuscript drafting remains blocked until the P0 gates pass.

## Execution stages

1. **P0 falsification** — frozen zero-shot inference, finite oracle headroom, historical-utility predictability.
2. **P1 method freeze** — only if P0 passes; specify selector, abstention rule, uncertainty, and baselines.
3. **P2 paper-eligible evaluation** — untouched later origins or held-out tasks, paired uncertainty, ablations, efficiency.
4. **P3 manuscript** — backfill only verified artifacts into the evidence matrix.

## Compute route

- CPU: dataset schema audit and unit tests.
- T4/L4: Chronos-2 adapter and reduced smoke.
- A100/H100: TimesFM 3 and full paired candidate inference only after smoke manifests agree.

The project does not purchase premium compute merely to discover that the phenomenon is absent.
