# TimesFM 3 input-strategy smoke contract

Status: planned  
Purpose: implementation validation only

This mirrors the verified Chronos-2 smoke on `epf_np` calibration origin 0. It uses the official FEV TimesFM-3 wrapper at commit `38007871dcf6dc6b04aed3a54d9cd86678d48d0b` and TimesFM source commit `20191171b74f51bfead932b6b8d0c8f515e70f63`.

The `google/timesfm-3.0-pytorch` weights use the TimesFM Non-Commercial License v1.0. This project uses them only for academic, non-commercial research. The runtime resolves and records the exact Hugging Face checkpoint revision before download.

The run must satisfy the same seven checks as the Chronos-2 smoke: CUDA, dataset fingerprint, calibration membership, cutoff, prediction validity, finite metrics, and no sealed evaluation access. Any target-only/all-dynamic difference remains `smoke_only` and cannot enter a scientific gate.
