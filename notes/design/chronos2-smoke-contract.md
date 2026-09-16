# Chronos-2 input-strategy smoke contract

Status: planned  
Purpose: implementation validation only

## Scope

The smoke run uses `epf_np`, calibration origin 0 (cutoff `-480`), and the frozen
`amazon/chronos-2` checkpoint. It evaluates exactly two input strategies:

- `target_only`: target history with no dynamic covariates;
- `all_dynamic`: target history plus both FEV-declared known-future covariates.

The same checkpoint instance, forecast horizon, quantile levels, and FEV scoring code are used for both variants.

## Required checks

1. CUDA is available and the GPU identity is recorded.
2. The dataset fingerprint equals `9830dda233defc6c`.
3. The selected origin belongs to the calibration partition.
4. The selected cutoff equals `-480`.
5. Both variants return a valid FEV prediction object.
6. SQL, WQL, MASE, WAPE, and inference time are finite.
7. No sealed evaluation origin is instantiated.

## Evidence boundary

Any numerical difference between the two variants is a pipeline diagnostic, not evidence for or against the paper hypothesis. No P0 gate is computed from this run. The output manifest must remain `smoke_only`.
