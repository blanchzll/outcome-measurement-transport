# Outcome measurement transport and calibration

Tagged review release: <https://github.com/blanchzll/outcome-measurement-transport/tree/v1.3.4>

This repository contains the release-safe analysis code and aggregate outputs for the manuscript:

> **Outcome-measurement transport reveals misleading calibration in clinical prediction models: a multicohort methodology study**

The study asks whether a fixed set of predictions can receive different calibration assessments when a health system changes which longitudinal measurements are observed and how the endpoint is reconstructed.

## What this repository contains

- An executable, synthetic-data `ascertainment_stress` package.
- Analysis scripts arranged in the order used for the study.
- Fixed simulation protocols and estimand definitions.
- Empirical cross-database measurement-schedule transport experiments.
- A fourth, non-ICU perioperative mechanism replication in VitalDB.
- A second, haemoglobin-decline operational endpoint replication.
- Probability-sampled reference-review designs and source follow-up bounds.
- A 4,014-record screened-population sensitivity and a non-outcome-stratified source-model uncertainty rebuild.
- A complete four-database 12-hour primary schedule-compatibility matrix with indexed source data.
- Synthetic unit and contract tests.
- Aggregate main tables, cross-database validation tables, figure source data, and machine-readable audits.
- A reviewer guide linking claims and display items to generating scripts.

## What this repository does not contain

No patient-level source data, row-level predictions, fitted clinical models, direct identifiers, or row-level public-database extracts are distributed. Access to INSPIRE, MIMIC-IV and eICU remains subject to their original data-use requirements; VitalDB clinical and laboratory tables are openly available from PhysioNet. The five-centre clinical data are not publicly distributable.

The public operational endpoint is a creatinine-based computational reference. It is not biological truth, full Kidney Disease Improving Global Outcomes adjudication, or a substitute for the five-centre expert endpoint.

## Repository map

```text
.
├── ascertainment_stress.py       # Executable generic stress-test utility
├── workflow/                     # Ordered full-study analysis scripts
│   ├── 00_provenance_and_estimands/
│   ├── 01_source_cohort_and_models/
│   ├── 02_public_reference_cohorts/
│   ├── 03_observability_selection_and_fairness/
│   ├── 04_measurement_deletion_simulation/
│   ├── 05_reference_sampling_and_correction/
│   ├── 06_transport_external_validation/
│   └── 07_figures_tables_and_submission/
├── protocols/                    # Frozen analysis and reporting definitions
├── results/
│   ├── key_tables/               # Aggregate result tables
│   ├── figure_source_data/       # Panel-level source data
│   └── audits/                   # Machine-readable QA and provenance audits
├── tests/                        # Synthetic tests; no clinical data
├── WORKFLOW_MANIFEST.csv         # Ordered script inventory
└── CODE_REVIEW_GUIDE.md          # Claim-to-code map for editors and reviewers
```

## Quick synthetic test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest -q
```

These tests verify endpoint reconstruction, retention behaviour, estimand separation, and simulation contracts without accessing clinical records. The exact clean-test environment is pinned in `requirements-test-lock.txt`. The full authorised-data workflow adds the packages listed in `requirements-full.txt`; the minimal package intentionally excludes database clients and plotting libraries unless the corresponding extra is installed.

## Full workflow configuration

The full analysis scripts require authorised local copies of the corresponding datasets. Copy `environment.example`, set only the paths available in your authorised environment, and run scripts in the sequence recorded in `WORKFLOW_MANIFEST.csv`.

```bash
export AKI_ANALYSIS_ROOT=/path/to/analysis_workspace
export AKI_SOURCE_ROOT=/path/to/restricted_source_data
export INSPIRE_ROOT=/path/to/inspire/1.4.2
export MIMIC_ROOT=/path/to/mimic
export EICU_ROOT=/path/to/eicu/2.0
export VITALDB_ROOT=/path/to/vitaldb/1.0.0
export MIMIC_DUCKDB=/path/to/mimiciv31.duckdb
```

The optional waveform extension requires the complete VitalDB 1.0.0 mirror
(6,394 official objects; approximately 95.4 GiB). Verify every object against
the release `SHA256SUMS.txt` before running
`10_vitaldb_waveform_feature_extraction.py`. The frozen protocol and acceptance
gates are in `protocols/`. Patient-level waveform features and predictions are
restricted outputs and must not be committed; only aggregate audits and tables
belong in a public release.

The submitted release was checked against all 6,394 official VitalDB objects
(102,456,164,496 bytes); no object was missing or hash-mismatched. The
machine-readable verification record is
`results/audits/VITALDB_FULL_MANIFEST_VERIFICATION.json`.

The primary 35% strong mixed-MNAR experiment separates the change from the full
operational cohort to the evaluable retained-reference cohort (selection) from
the change introduced when the endpoint is reconstructed on that same
denominator. A calibration-specific control fits identical two-fold updates to
either intact retained labels or reconstructed labels among the same evaluable
records, then evaluates both on selected and full retained labels. Fixed-cohort
deletion draws and nested patient or hospital resampling are reported separately
in `Table_primary_selection_reconstruction_decomposition.csv` and
`Table_primary_calibration_selection_reconstruction_control.csv`.

The empirical schedule experiment uses the same minimum donor rule in every
database: at least one valid post-landmark creatinine time in (0, 168] hours.
Donor eligibility does not require a reconstructable endpoint or dense-reference
membership. VitalDB retains its prespecified adult, single-operation restriction.

Scripts retain their frozen numerical settings, seeds, cohort rules, model specifications, and output contracts. Internal server paths have been replaced by environment-based configuration in the public copy.

## Main evidentiary boundaries

1. The 3,710-patient source analysis is a motivating clinical case and does not have an untouched independent clinical external validation; 4,014 is the screened denominator, not the primary modelling cohort.
2. INSPIRE-to-MIMIC-IV/eICU is locked cross-database external validation of a public-data model against the harmonised creatinine endpoint, but performance was weak.
3. Applying the INSPIRE gastrointestinal model to the five-centre cohort is an endpoint-transport clinical bridge, not strict same-endpoint validation.
4. Dense-reference analyses are conditional on selected, highly monitored populations.
5. Retrospective decision analysis does not establish clinical impact.
6. Empirically sampled measurement schedules reproduce observed timing distributions; they are not estimates of causal hospital testing policies.
7. The haemoglobin-decline endpoint is an operational laboratory endpoint, not adjudicated bleeding.
8. VitalDB analyses use a creatinine-only operational reference and observed measurement schedules; they are not full KDIGO adjudication or causal tests of hospital policy.
9. The VitalDB waveform extension failed its prespecified model-strength gate and is retained only as a supplementary measurement-stress sensitivity.
10. The 4,014-record source sensitivity uses separately fitted leave-one-centre-out models; it evaluates numerical robustness but does not identify the effect of the unexplained 304-record exclusion.
11. The decomposition and nested intervals are empirical stress-test summaries. They do not identify a biological target under MNAR or resample source-model development.

## Licence and citation

Code is released under the MIT License. The licence does not cover clinical data, restricted database extracts, fitted clinical models, or institution-specific mappings. Cite tagged release `v1.3.4`; an archive DOI can be added after author-controlled deposit.
