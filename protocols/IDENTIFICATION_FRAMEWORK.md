# Identification framework for outcome-measurement transport

Let `P` denote a frozen predicted probability, `L*` the retained longitudinal laboratory trajectory, `R(t)` the time-varying measurement indicator, `Y^R = g(L*)` the retained operational reference endpoint, and `Y^M = g(R(t)L*)` the endpoint reconstructed after the trajectory is coarsened. The framework separates three estimands that are often conflated: calibration against `Y^R`, calibration against `Y^M`, and calibration after updating a model to `Y^M`.

## Identification propositions

1. If measurement affects only whether an otherwise invariant label is observed, inverse-probability-of-ascertainment weighting can identify target-population performance under conditional exchangeability, positivity and a sufficiently accurate observation model. Augmentation offers the usual double-robustness property only for this missing-label setting.
2. If measurement deletion changes the label-generating trajectory so that `Y^M` may differ from `Y^R`, record-level weighting cannot reconstruct the missing measurements. Identification then requires a representative retained-trajectory reference sample or an explicitly modelled trajectory process with stronger assumptions.
3. Local recalibration against `Y^M` targets `Pr(Y^M=1|P)`. It need not improve, and may degrade, calibration against `Y^R`. Both evaluation targets must therefore be reported when retained reference outcomes exist.
4. A probability reference sample can identify retained-reference calibration if inclusion probabilities are known and positive over the risk distribution. Precision depends primarily on reference-event yield and risk-range coverage, not sample fraction alone.

These are conditional identification statements, not causal claims about why a particular hospital measured more or less frequently. Empirical donor schedules represent observed timing distributions rather than interventions on hospital policy.
