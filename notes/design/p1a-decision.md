# P1a decision

P1a passes its frozen calibration-only continuation gate. Four of eight
`(backbone, task)` groups meet both point AUROC and clustered-confidence criteria:

- Chronos-2: `rossmann_1D`;
- TimesFM 3: `rohlik_orders_1D`, `rossmann_1D`, and
  `solar_with_weather_1H`.

The result satisfies the requirement of at least three eligible groups and at least
one eligible task per backbone. The original P0c universal claim remains failed; P1a
supports only a heterogeneous-applicability framing.

The TimesFM 3 solar group is borderline because its one-sided 95% lower bound is
`0.5022684945`. It remains eligible because the frozen rule is strictly greater than
`0.50`, but it must be shown with its interval and cannot be described as strong
evidence. Chronos-2 has no certified scope outside Rossmann.

P1a authorizes preparation of a P1b router-freeze contract. It does not authorize
sealed evaluation. The private full report must be integrity-checked against the
archived decision before P1b state is created.
