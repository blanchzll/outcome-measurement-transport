# Outcome-measurement transport audit procedure

## Purpose

This procedure tests whether a clinical model appears calibrated because it
matches the intended endpoint or because it matches the local process that
records and reconstructs that endpoint. It is an evaluation workflow, not a new
universal correction estimator.

## Required inputs

1. A fixed prediction score at a frozen clinical landmark.
2. A longitudinal retained-reference subset with measurement times and values.
3. An explicit endpoint-construction function.
4. The local or hypothetical measurement process to be audited.
5. Prespecified calibration metrics and identifying assumptions.

## Five-step procedure

1. **Name both targets.** Define the intended retained-reference endpoint and
   the endpoint reconstructed from locally available measurements. State which
   target each performance estimate addresses.
2. **Audit observability.** Report measurement counts and timing, endpoint
   evaluability, positivity, covariate balance, weight distributions and
   effective sample size.
3. **Transport measurement, not risk.** Hold patients, retained trajectories
   and predictions fixed. Delete measurements under prespecified mechanisms or
   apply empirical donor timing vectors, then reconstruct the endpoint.
4. **Separate selection from reconstruction.** On one evaluable denominator,
   compare intact retained labels with reconstructed labels. Fit identical
   cross-fitted updates to each label using the same folds. Then transport the
   intact-label update from the selected denominator to the full retained
   cohort. Use a pure invariant-label selection control to verify weighting.
5. **Compare targets and bound interpretation.** Report paired reconstructed-
   target and retained-reference calibration. Use IPAW/AIPW only for their
   stated invariant-label estimand and use retained-reference sampling or
   assumption-indexed bounds when measurements change the label. Do not call
   local numerical agreement recovery of clinical risk without target alignment.

## Required outputs

- cohort and target ledger;
- observability, positivity and weight diagnostics;
- selection/reconstruction common-denominator decomposition;
- paired O/E, calibration intercept and slope for both targets;
- invariant-label positive control;
- sensitivity to deletion mechanism, strength, retention and score discrimination;
- a plain-language interpretation boundary.

## Relation to existing approaches

| Approach | Label assumption | What it can recover | What this procedure adds |
|---|---|---|---|
| Informative-presence analysis | Encounter or test occurrence is informative | Association or prediction conditional on an observation process | Makes the calibration target and endpoint-construction step explicit. |
| Missing-label IPW | A binary label exists before selection | Full-sample performance under exchangeability and positivity | Positive control and failure check when deletion changes the label itself. |
| Outcome-misclassification validation sample | Reference labels exist in a sample | Bias-adjusted performance under validation-sample assumptions | Designs the retained-reference sample around calibration events and risk-range coverage. |
| Local recalibration | Local observed label represents the intended target | Agreement with the local label distribution | Paired evaluation reveals whether updating follows the retained target or recording process. |
| Outcome-measurement transport audit | Longitudinal measurements generate the label | Diagnostic separation rather than universal correction | Holds risk and patients fixed, perturbs measurement before reconstruction, and decomposes selection, reconstruction and updating. |

## Interpretation rule

If an update improves reconstructed-target calibration but remains discordant
with the retained-reference target, report agreement with the local recording
process. Do not infer that the underlying clinical risk has been recovered.
