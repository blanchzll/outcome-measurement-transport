# Release notes: v1.3.4

This review release supports the Journal of Translational Medicine Methodology submission.

## Calibration-specific decomposition

- Adds an identical-record, identical-fold control that fits calibration updates to either intact retained labels or reconstructed labels among the same evaluable records.
- Evaluates each update on selected retained labels and full retained-reference cohorts, separating fitted-label reconstruction from transport out of a selected denominator.
- Reports 300 fixed-cohort deletion replicates and 100 patient- or hospital-resampled replicates with Monte Carlo standard errors.

## Empirical schedule transport

- Harmonises donor eligibility across INSPIRE, MIMIC-IV, eICU and VitalDB to at least one valid post-landmark creatinine time in (0, 168] hours.
- Retains the prespecified adult, single-operation restriction for VitalDB while removing dense-reference membership from donor eligibility.
- Repeats all 12-hour and 24-hour target-donor comparisons with 200 replicates per cell and refreshes aggregate source data.

## Reproducibility

- Adds a five-step outcome-measurement transport audit procedure.
- Adds tested workflow dependency extras and an exact clean-test lock file.
- Expands continuous integration to the complete test suite; 16 tests pass in a newly resolved environment and in the authorised analysis environment.

No patient-level data, row-level predictions or fitted clinical models are included.
