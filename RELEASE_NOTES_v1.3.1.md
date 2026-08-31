# Release notes: v1.3.1

This release freezes the Nature Communications submission analysis after two independent review rounds.

## Scientific and statistical changes

- Retains the 3,710-patient/152-event cohort as the primary source analysis and labels 4,014 records/155 events as a screened-population sensitivity.
- Rebuilds source-model fixed-prediction intervals with patient resampling within centre, centre sizes preserved, outcome counts free to vary and no outcome stratification.
- Adds an aggregate stable-ID linkage audit: 295 omitted records have unresolved sex coding and nine omitted records are female-coded; the exclusion mechanism remains incompletely explained.
- Displays the complete four-database 12-hour empirical schedule-compatibility matrix as Figure 5; 24 hours remains a sensitivity analysis.
- Adds the complete VitalDB waveform extension. The extension passed data and timing gates but failed its prespecified model-strength gate, so it is supplementary only.
- Separates dense-reference selection from reconstructed-endpoint error on a common denominator at the prespecified 35% strong mixed-MNAR condition. Fixed-cohort deletion uncertainty and nested patient or hospital resampling are reported separately.
- Verifies all 6,394 official VitalDB objects (102,456,164,496 bytes) against the release manifest with no missing or mismatched files.

## Reproducibility changes

- Adds one-freeze assertions for primary cohort counts, bridge estimates, hierarchical calibration values, Figure 5 tolerance and workbook roles.
- Adds indexed source-data and supplementary-table workbooks and release-safe aggregate discrepancy tables.
- Updates the reviewer guide, workflow manifest, privacy checks and immutable release manifest.
- Moves the five-centre hierarchical calibration display to Supplementary Figure 16 and rebuilds all final panels to satisfy the prespecified final-size legibility audit.
- Restores the authoritative 3,710-patient, 152-event source-model table; the 4,014-record result remains a labelled sensitivity analysis.

## Evidentiary boundary

The release does not contain patient-level clinical data or row-level predictions. It does not establish clinical impact, independent external validation of the five-centre source model, causal hospital-policy effects or full-KDIGO validation of public operational endpoints.
