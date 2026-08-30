# Outcome measurement transport and calibration

Tagged release: <https://github.com/blanchzll/outcome-measurement-transport/tree/v1.3.0>

This repository contains the release-safe analysis code and aggregate outputs for the manuscript:

> **Transported outcome-measurement schedules can alter calibration of clinical prediction models**

The study asks whether a fixed set of predictions can receive different calibration assessments when a health system changes which longitudinal measurements are observed and how the endpoint is reconstructed.

## What this repository contains

- An executable, synthetic-data `ascertainment_stress` package.
- Analysis scripts arranged in the order used for the study.
- Fixed simulation protocols and estimand definitions.
- Empirical cross-database measurement-schedule transport experiments.
- A fourth, non-ICU perioperative mechanism replication in VitalDB.
- A second, haemoglobin-decline operational endpoint replication.
- Probability-sampled reference-review designs and source follow-up bounds.
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
python -m pip install -e . pytest
pytest -q
```

These tests verify endpoint reconstruction, retention behaviour, estimand separation, and simulation contracts without accessing clinical records.

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

Scripts retain their frozen numerical settings, seeds, cohort rules, model specifications, and output contracts. Internal server paths have been replaced by environment-based configuration in the public copy.

## Main evidentiary boundaries

1. The 3,710-patient source analysis is a motivating clinical case and does not have an untouched independent clinical external validation; 4,014 is the screened denominator, not the primary modelling cohort.
2. INSPIRE-to-MIMIC-IV/eICU is true same-model public-database external validation, but performance was weak.
3. Applying the INSPIRE gastrointestinal model to the five-centre cohort is an endpoint-transport clinical bridge, not strict same-endpoint validation.
4. Dense-reference analyses are conditional on selected, highly monitored populations.
5. Retrospective decision analysis does not establish clinical impact.
6. Empirically sampled measurement schedules reproduce observed timing distributions; they are not estimates of causal hospital testing policies.
7. The haemoglobin-decline endpoint is an operational laboratory endpoint, not adjudicated bleeding.
8. VitalDB analyses use a creatinine-only operational reference and observed measurement schedules; they are not full KDIGO adjudication or causal tests of hospital policy.

## Licence and citation

Code is released under the MIT License. The licence does not cover clinical data, restricted database extracts, fitted clinical models, or institution-specific mappings. Cite tagged release `v1.3.0` and the associated manuscript; an archive DOI will be added after author-controlled archival deposit.
