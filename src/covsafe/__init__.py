"""Model-independent diagnostics for covariate-safe TSFM forecasting."""

from covsafe.config import canonical_config_hash
from covsafe.diagnostics import (
    GateReport,
    GateThresholds,
    ModelEvidence,
    binary_auroc,
    evaluate_p0_gates,
    finite_oracle_headroom,
    harm_rate,
    relative_gain,
)

__all__ = [
    "GateReport",
    "GateThresholds",
    "ModelEvidence",
    "binary_auroc",
    "canonical_config_hash",
    "evaluate_p0_gates",
    "finite_oracle_headroom",
    "harm_rate",
    "relative_gain",
]
