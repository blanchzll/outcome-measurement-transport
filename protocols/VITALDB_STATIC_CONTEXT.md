# VitalDB Extension: Static Analysis Context

## Scientific role

- This extension evaluates whether outcome-observation processes can alter apparent calibration while the underlying patient risk and prediction score are held fixed.
- VitalDB is a perioperative, non-ICU replication dataset. It complements, but does not replace, the five-center expert-adjudicated clinical cohort.
- The prediction landmark is the recorded end of surgery (`opend`). Only information available by that landmark may enter a prediction model.
- The VitalDB reference outcome is an operational **creatinine-only** postoperative AKI endpoint. It must not be described as complete KDIGO adjudication because longitudinal urine output, RRT timing, and blinded nephrologist adjudication are unavailable.

## Primary operational endpoint

- Baseline creatinine: latest valid creatinine strictly before `opend` when a timestamped measurement exists.
- Postoperative window: greater than 0 through 168 hours after `opend`.
- Event: creatinine increase of at least 0.3 mg/dL within 48 hours, or at least 1.5 times baseline within 168 hours.
- A preoperative-creatinine-field fallback may be evaluated only as a labeled sensitivity analysis.

## Reproducibility and governance

- Raw VitalDB files remain outside the public repository.
- Public code must accept dataset locations through command-line options or environment variables.
- Every reported analytic table must record dataset version, inclusion counts, endpoint construction, exclusions, and checksum provenance.
- Primary model and simulation analyses exclude patients with multiple operations because absolute chronology has been removed and a random case identifier cannot identify a first operation. All-operation sensitivity analyses require patient-clustered uncertainty.
- Missing outcome measurements are not ordinary covariate missingness. Observation-model assumptions and partial-identification limits must be reported explicitly.
