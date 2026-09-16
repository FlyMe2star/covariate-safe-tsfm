# P0 diagnostic contract

Version: `p0-v1`  
Status: frozen before model inference

Canonical configuration hash: `b2ace898d038` (first 12 characters of SHA-256 over canonical JSON).

## Decision

P0 is a falsification experiment, not a leaderboard run. It asks whether the phenomenon and the information needed for a selector exist. Official final-test data are not instantiated.

## Backbones

- `amazon/chronos-2` through the official Chronos-2 pipeline.
- `google/timesfm-3.0-pytorch` through the official TimesFM 3 evaluator.

Both run frozen and zero-shot. Revisions, library versions, hardware, and upstream licenses must be recorded in every manifest.

## Tasks

The task definitions are copied from the official FEV mini benchmark at commit
`38007871dcf6dc6b04aed3a54d9cd86678d48d0b` and retain its SQL primary metric.

| Task | Horizon | Windows | Known-future covariates | Past-only covariates |
|---|---:|---:|---|---|
| `epf_np` | 24 | 20 | Grid load forecast; Wind power forecast | none |
| `rohlik_orders_1D` | 61 | 5 | holiday; shops closed; school-holiday flags | operational, weather, and user-activity variables |
| `rossmann_1D` | 48 | 10 | store-open, promotion, weekday and holiday variables | Customers |
| `solar_with_weather_1H` | 24 | 20 | seven weather/day-length variables | irradiance; cloud cover |

Static columns are excluded in P0 because model support is not matched across both backbones. The evaluation unit is `(backbone, task, item, target, origin)`.

## Temporal split

For each task, origins are sorted chronologically. The earliest 60% are calibration origins and the latest 40% are sealed evaluation origins; floor rounding applies to calibration with at least two origins in each side. The split is created before any forecast loss is inspected.

## Candidate policies

The finite candidate set prevents exponential subset search and retrospective cherry-picking:

1. `target_only`
2. `all_dynamic`
3. every `single_known_future:<name>` policy
4. every `leave_one_out:<name>` policy over dynamic covariates
5. `correlation_top_1`, `correlation_top_2`, and `correlation_top_4`

Correlation ranking is computed only on observations before each origin. If fewer variables exist, `top_k` is clipped and duplicate policies are removed. The oracle is the minimum SQL within this exact set and is reported only as diagnostic headroom, never as a deployable result.

## Historical-utility signal

For each evaluation unit and candidate policy, historical utility is the exponentially weighted mean relative SQL gain over calibration origins. The half-life is three origins. No model is trained in P0. Future utility sign is positive iff the candidate's SQL is lower than target-only SQL at the sealed origin.

## Leakage rules

- Known-future covariate values may be supplied across the forecast horizon only when FEV declares the column known-dynamic.
- Past-only covariates stop strictly before the forecast origin.
- Scaling, missing-value handling, correlation ranking, and utility summaries use pre-origin observations only.
- Calibration origins precede every evaluation origin.
- Evaluation outcomes cannot alter policies, thresholds, tie-breaking, or preprocessing.
- Failed runs are retained in manifests; reruns may fix implementation errors but may not alter the scientific contract.

## Metrics

- Primary: FEV scaled quantile loss (SQL).
- Secondary: WQL and MASE where available.
- Utility: `(baseline_loss - candidate_loss) / max(abs(baseline_loss), 1e-12)`.
- Harm event: all-covariate utility versus target-only is at most `-0.05`.
- Oracle headroom: relative improvement of the finite oracle over `min(target_only, all_dynamic)`.
- Predictability: AUROC for historical utility predicting future positive utility.

All aggregate estimates receive paired item-origin bootstrap 95% confidence intervals in the paper-eligible stage.

## Advance/stop gates

Each backbone must independently satisfy:

- harm rate >= 0.20;
- mean oracle headroom >= 0.05;
- utility-sign AUROC >= 0.65.

P0 passes only when both backbones pass all three gates. Otherwise the core method direction stops or is explicitly reframed before any official final-test access.

## Artifact statuses

- `smoke_only`: validates loading, shapes, masks, and deterministic inference.
- `screening_only`: P0 calibration/evaluation diagnostic; not a paper result.
- `paper_eligible`: created only after a fixed method, independent final protocol, and reproducibility audit.
