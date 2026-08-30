# Code review guide

This guide lets editors and reviewers trace each manuscript claim to the relevant workflow stage and aggregate output without accessing patient-level data.

## Dataset and model roles

| Canonical role | Dataset | Model or endpoint role | What it can establish | What it cannot establish |
|---|---|---|---|---|
| Source clinical case | Five-centre gastric/colorectal cohort | End-of-surgery source models; expert binary KDIGO endpoint; scheduled creatinine availability as an observation-opportunity proxy | Internal-external transport, calibration heterogeneity and data-audit limits | Cause of centre differences, independent external validation or clinical impact |
| Longitudinal measurement testbed | INSPIRE | Surgery-anchored retained creatinine reference and deletion experiments | Measurement-induced endpoint and calibration changes | Full KDIGO or population-wide performance |
| ICU replication | MIMIC-IV and eICU | ICU-admission operational references and database-native models | Replication of the measurement mechanism | Validation of the source clinical model |
| Perioperative replication | VitalDB | Non-ICU perioperative creatinine reference, fixed held-out predictions and observed measurement schedules | Replication beyond the ICU databases | Full KDIGO adjudication or a causal hospital-policy effect |
| Locked public external validation | INSPIRE to MIMIC-IV/eICU | Same serialized model, ICU landmark, common variables, creatinine endpoint | True public same-model external validation | Clinical readiness; source-model validation |
| Endpoint-transport clinical bridge | INSPIRE gastrointestinal model to MIMIC-IV and five-centre cohort | Same predictors and model; reference endpoint changes in the source cohort | Risk transport under an endpoint-reference change | Strict same-endpoint external validation |
| Empirical schedule transport | INSPIRE, MIMIC-IV and eICU | Each target cohort is remeasured using measurement-time distributions sampled from another database | Robustness beyond parametric deletion rules | Causal effects of an actual hospital testing policy |
| Second-endpoint replication | MIMIC-IV and eICU | Operational 0-168 h haemoglobin-decline endpoint | Whether the measurement mechanism generalises beyond creatinine | Adjudicated bleeding or transfusion causality |

## Claim-to-code map

| Manuscript claim | Primary code | Key aggregate outputs |
|---|---|---|
| Dense-reference populations are selected | `workflow/00_provenance_and_estimands/78_dense_reference_selection_audit.py`; `workflow/02_public_reference_cohorts/50_inspire_observability_reference.py` | `results/figure_source_data/Figure1`; dense-reference audit JSON |
| Longitudinal deletion changes endpoint reconstruction and apparent calibration | `workflow/04_measurement_deletion_simulation/52_measurement_deletion_simulation.py`; `64_parallel_measurement_simulation.py` | Figure 2 source data; main Table 2; simulation audits |
| Weighting works for pure label selection under its identifying conditions | `workflow/03_observability_selection_and_fairness/72_pure_label_selection_control.py` | Positive-control audit JSON and supplementary tables |
| Recalibration can fit the reconstructed target but miss the retained reference | `workflow/04_measurement_deletion_simulation/52_measurement_deletion_simulation.py`; `67_resummarize_simulations.py` | Figure 3 source data; main Table 3 |
| Reference-event count controls updating precision | `workflow/05_reference_sampling_and_correction/77_reference_event_design.py` | Figure 3 source data; main Table 4 |
| The failure mode persists at designed discrimination levels | `workflow/04_measurement_deletion_simulation/83_discrimination_strength_stress_test.py` | Figure 4 source data and discrimination audit |
| Empirical cross-database schedules can make reconstructed calibration appear nearly exact while retained-reference calibration differs | `workflow/04_measurement_deletion_simulation/94_empirical_schedule_transport.py` | `Table_empirical_schedule_transport.csv`; Supplementary Figure 9 source data; empirical-schedule audit |
| The same estimand split is reproduced with an operational haemoglobin endpoint | `workflow/04_measurement_deletion_simulation/97_secondary_hemoglobin_endpoint.py` | `Table_hemoglobin_endpoint_replication.csv`; Supplementary Figure 11 source data; haemoglobin audit |
| Source model transport is internal-external, not independent external validation | `workflow/01_source_cohort_and_models/54_regenerate_loco_predictions.py`; `79_source_fixed_geography_validation.py`; `88_source_temporal_validation.py` | Main Table 1; temporal and geography audits |
| Source scheduled fields audit observation opportunity but do not identify its cause | `workflow/01_source_cohort_and_models/96_source_observation_bounds.py` | `SOURCE_OBSERVATION_AND_PREOP_AKI_SENSITIVITY_AUDIT.json`; Supplementary Figure 5 source data |
| Short postoperative stays impose partially identified source-cohort bounds | `workflow/01_source_cohort_and_models/96_source_observation_bounds.py` | `Table_source_postdischarge_sensitivity_bounds.csv`; Supplementary Figure 12 source data |
| INSPIRE-locked models transport poorly to MIMIC-IV and eICU | `workflow/06_transport_external_validation/84_build_inspire_surgical_icu_reference.py`; `85_inspire_locked_public_icu_transport.py` | `Table_inspire_locked_external_validation.csv`; Supplementary Figure 7 source data |
| VitalDB reproduces measurement-induced apparent versus retained calibration divergence in a non-ICU perioperative cohort | `workflow/02_public_reference_cohorts/02_vitaldb_creatinine_reference.py`; `workflow/04_measurement_deletion_simulation/03_vitaldb_measurement_transport.py`; `workflow/04_measurement_deletion_simulation/05_vitaldb_empirical_schedule_extension.py` | `Table_vitaldb_simulation_summary.csv`; `Table_vitaldb_empirical_schedule_extension.csv`; Figure 5 and Supplementary Figures 13-14 source data |
| The public gastrointestinal model is a clinical endpoint bridge | `workflow/06_transport_external_validation/86_inspire_gi_model_to_mimic_and_source.py` | `Table_public_model_to_source_clinical_bridge.csv`; Supplementary Figure 8 source data |
| Risk-enriched reference review increases event yield but not uniformly precision | `workflow/05_reference_sampling_and_correction/95_optimized_reference_sampling.py` | `Table_optimized_reference_sampling.csv`; Supplementary Figure 10 source data; sampling-design audit |

## Recommended review order

1. Read `protocols/STATISTICAL_ANALYSIS_PLAN.md` and `protocols/TERMINOLOGY_LEDGER.md`.
2. Inspect `WORKFLOW_MANIFEST.csv` and stages 00-02 for cohort and endpoint construction.
3. Inspect stages 03-05 for selection, parametric and empirical schedule deletion, correction, second-endpoint replication, and reference sampling.
4. Inspect stage 06 for transport and external validation.
5. Match numerical claims to `results/key_tables`, then inspect panel-level values under `results/figure_source_data`.
6. Review machine-readable completion and boundary checks under `results/audits`.

## Reproduction levels

- **Level 1, unrestricted:** Run the synthetic package and tests. No clinical data are required.
- **Level 2, public clinical data:** Rebuild INSPIRE, MIMIC-IV and eICU operational cohorts after obtaining authorised access, and rebuild VitalDB from its open PhysioNet release.
- **Level 3, governed source data:** Reproduce the five-centre analyses inside the approved institutional environment. The repository documents the workflow but does not distribute the data.

All reported manuscript numbers can be inspected in aggregate outputs. Patient-level verification requires the relevant data-governance permissions.
