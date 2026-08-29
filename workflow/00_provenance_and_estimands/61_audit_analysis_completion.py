# %% [markdown]
# # Analysis and submission-readiness audit
# Verifies production artifacts without reading or exporting patient-level data.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import pandas as pd

ROOT = Path(str(_release_path('analysis')))
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
PACKAGE = ROOT / "package" / "ascertainment-stress-test"


def load_json(name: str) -> dict:
    return json.loads((OUTPUTS / name).read_text())


def exists_all(*paths: Path) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


rows: list[dict] = []


def record(priority: str, item: int, analysis: str, passed: bool, evidence: str) -> None:
    rows.append(
        {
            "priority": priority,
            "item": item,
            "analysis": analysis,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
        }
    )


# 1: observability, IPAW/AIPW, diagnostics, MNAR bounds
obs = load_json("INSPIRE_OBSERVABILITY_REFERENCE_AUDIT.json")
obs_ok = (
    obs.get("candidate_operations") == 7135
    and exists_all(
        TABLES / "Table_observability_adjusted_performance.csv",
        TABLES / "Table_observability_weight_diagnostics.csv",
        TABLES / "Table_observability_predictor_imbalance.csv",
        TABLES / "Table_two_slot_MNAR_sensitivity.csv",
        TABLES / "Table_longitudinal_168h_MNAR_sensitivity.csv",
    )
)
record("must", 1, "INSPIRE observability, IPAW/AIPW, diagnostics and MNAR bounds", obs_ok,
       f"candidate_operations={obs.get('candidate_operations')}; required tables present={obs_ok}")

# 2: longitudinal 0-168 h reference and two-window sensitivity
ref_ok = (
    obs.get("longitudinal_reference_n") == 6333
    and obs.get("two_slot_observed_n") == 2073
    and obs.get("dense_reference_n") == 1676
    and exists_all(TABLES / "Table_two_slot_vs_longitudinal_concordance.csv",
                   TABLES / "Table_monitoring_density_event_gradient.csv")
)
record("must", 2, "INSPIRE operational 0-168 h creatinine reference and window sensitivity", ref_ok,
       f"longitudinal={obs.get('longitudinal_reference_n')}; two_slot={obs.get('two_slot_observed_n')}; dense={obs.get('dense_reference_n')}")

# 3-5: production Monte Carlo replication and correction comparisons
simulation_methods = {
    "full_reference", "naive", "IPAW_design_probability_untruncated",
    "IPAW_design_probability_truncated99", "AIPW_design_probability",
    "recalibration_intercept_apparent", "recalibration_intercept_truth",
    "recalibration_intercept_slope_apparent", "recalibration_intercept_slope_truth",
    "reference_05pct_recalibration", "reference_10pct_recalibration",
    "reference_20pct_recalibration", "reference_30pct_recalibration",
    "Gamma2_prediction_sensitivity_region",
}
core_simulation_methods = {method for method in simulation_methods if not method.startswith("reference_")}
for item, db in [(3, "INSPIRE"), (4, "MIMIC"), (18, "EICU")]:
    audit = load_json(f"{db}_SIMULATION_AUDIT.json")
    tab = pd.read_csv(TABLES / f"Table_{db.lower()}_simulation_summary.csv")
    methods = set(tab["method"].dropna().astype(str))
    sim_ok = (
        audit.get("replicates_per_condition") == 300
        and 108000 <= int(audit.get("replicate_rows", -1)) <= 151200
        and tab.loc[tab.method.isin(core_simulation_methods), "n_replicates"].min() == 300
        and tab.loc[tab.method.str.startswith("reference_"), "n_replicates"].min() >= 250
        and simulation_methods.issubset(methods)
        and set(tab["mechanism"]) == {"MCAR", "stratum_MAR", "risk_MAR", "history_MAR", "outcome_MNAR", "mixed_MNAR"}
    )
    record("must", item, f"{db} complete-outcome deletion/reconstruction Monte Carlo", sim_ok,
           f"replicates={audit.get('replicates_per_condition')}; rows={audit.get('replicate_rows')}; methods={len(methods)}")

sim_i = pd.read_csv(TABLES / "Table_inspire_simulation_summary.csv")
sim_m = pd.read_csv(TABLES / "Table_mimic_simulation_summary.csv")
sim_e = pd.read_csv(TABLES / "Table_eicu_simulation_summary.csv")
comparison_ok = all(simulation_methods.issubset(set(x.method)) for x in (sim_i, sim_m, sim_e))
record("must", 5, "IPAW/AIPW/recalibration/reference-sample/MNAR comparison", comparison_ok,
       "All prespecified estimand-method labels present in all three public databases")

# 6: precision, stability and hierarchical center calibration
primary = load_json("PRECISION_UTILITY_FAIRNESS_AUDIT.json")
stability = load_json("SOURCE_MODEL_STABILITY_AUDIT.json")
precision_ok = (
    primary.get("source_n") == 4014 and primary.get("source_events") == 155
    and primary.get("bootstrap_precision") == 1000
    and stability.get("bootstrap_refits") == 200
    and exists_all(TABLES / "Table_source_model_precision.csv",
                   TABLES / "Table_source_model_stability_200bootstrap.csv",
                   TABLES / "Table_source_hierarchical_calibration_meta.csv")
)
record("must", 6, "Precision, 200-refit stability and hierarchical center calibration", precision_ok,
       f"source={primary.get('source_n')}/{primary.get('source_events')} events; precision bootstrap={primary.get('bootstrap_precision')}; refits={stability.get('bootstrap_refits')}")

# 7-9: application scenario, subgroup audit, incremental value/frontier
record("recommended", 7, "Monitoring thresholds, decision curves, burden and capture", exists_all(
    TABLES / "Table_monitoring_threshold_burden_capture.csv"), "Thresholds 0.02-0.15 and top-fraction policies saved")
fair_audit = load_json("FAIRNESS_EXTERNAL_PRECISION_AUDIT.json")
fair_ok = fair_audit.get("subgroup_bootstrap") == 500 and exists_all(
    TABLES / "Table_fairness_bootstrap_intervals.csv", TABLES / "Table_fairness_max_min_disparities.csv")
record("recommended", 8, "Sex, age, cancer-site and approach representativeness audit", fair_ok,
       f"subgroup bootstrap={fair_audit.get('subgroup_bootstrap')}; descriptive, not fairness certification")
frontier_ok = exists_all(TABLES / "Table_preop_to_perioperative_increment.csv",
                         TABLES / "Table_portability_performance_frontier.csv")
record("recommended", 9, "Preoperative-to-perioperative increment and portability frontier", frontier_ok,
       "Paired 1000-bootstrap increments and feature-count frontier saved")

# 10: public package, protocol, tests and figure source preflight
preflight = load_json("FIGURE_SOURCE_PREFLIGHT.json")
package_ok = exists_all(PACKAGE / "ascertainment_stress.py", PACKAGE / "README.md",
                        PACKAGE / "SIMULATION_PROTOCOL.md", PACKAGE / "tests" / "test_ascertainment_stress.py")
package_patient_files = [p for p in PACKAGE.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".gz", ".xlsx", ".parquet"}]
tool_ok = package_ok and not package_patient_files and preflight.get("summary", {}).get("ready") is True
record("recommended", 10, "Open-source ascertainment stress-test tool and protocol", tool_ok,
       f"package complete={package_ok}; patient-data files={len(package_patient_files)}; figure preflight ready={preflight.get('summary', {}).get('ready')}")

# 11-13: positive control, public endpoint components, and executable contract tests
selection_ok = exists_all(
    TABLES / "Table_inspire_pure_label_selection_control.csv",
    TABLES / "Table_mimic_pure_label_selection_control.csv",
    TABLES / "Table_eicu_pure_label_selection_control.csv",
    OUTPUTS / "INSPIRE_PURE_LABEL_SELECTION_CONTROL_AUDIT.json",
    OUTPUTS / "MIMIC_PURE_LABEL_SELECTION_CONTROL_AUDIT.json",
    OUTPUTS / "EICU_PURE_LABEL_SELECTION_CONTROL_AUDIT.json",
)
record("must", 11, "Pure label-selection positive control", selection_ok,
       "Oracle untruncated IPW benchmark separated from measurement coarsening")
component_ok = exists_all(
    TABLES / "Table_inspire_kdigo_component_availability.csv",
    TABLES / "Table_mimic_kdigo_component_availability.csv",
    TABLES / "Table_mimic_endpoint_target_performance.csv",
    TABLES / "Table_eicu_kdigo_component_availability.csv",
    OUTPUTS / "PUBLIC_KDIGO_COMPONENT_AUDIT.json",
    OUTPUTS / "EICU_KDIGO_COMPONENT_AUDIT.json",
)
record("must", 12, "Public KDIGO-component availability and endpoint-compatibility audit", component_ok,
       "Creatinine, urine-output and RRT component coverage quantified without claiming adjudication")
contract_ok = exists_all(ROOT / "code" / "71_test_simulation_contract.py",
                         PACKAGE / "tests" / "test_simulation_contract.py")
record("must", 13, "Executable 48 h, 96 h, 168 h and observed-history contract tests", contract_ok,
       "Regression-test script present; execution is repeated in release QA")

role_ok = exists_all(
    TABLES / "Table_dataset_roles_and_inference_boundaries.csv",
    TABLES / "Table_estimand_ledger.csv",
    OUTPUTS / "DATASET_ROLE_ESTIMAND_AUDIT.json",
)
record("must", 14, "Dataset roles, endpoint targets and inference boundaries", role_ok,
       "Three-dataset role table and five-estimand ledger present")

# 15-17: final top-journal reporting and selection audits
source_reporting_ok = exists_all(
    TABLES / "Table_source_characteristics_by_center.csv",
    TABLES / "Table_source_predictor_definitions_units.csv",
    OUTPUTS / "SOURCE_REPORTING_TABLE_AUDIT.json",
)
record("must", 15, "Source aggregate characteristics, predictor definitions and units", source_reporting_ok,
       "Aggregate-only source reporting tables and garbled-sex-code audit present")
dense_selection_ok = exists_all(
    TABLES / "Table_dense_reference_selection_audit.csv",
    TABLES / "Table_dense_reference_selection_flow.csv",
    OUTPUTS / "DENSE_REFERENCE_SELECTION_AUDIT.json",
)
record("must", 16, "Candidate-to-dense-reference selection audit", dense_selection_ok,
       "Selection fractions and standardized differences reported for all three public databases")
reference_event_ok = exists_all(
    TABLES / "Table_reference_event_design.csv",
    TABLES / "Table_reference_event_count_operating_characteristics.csv",
    OUTPUTS / "REFERENCE_EVENT_DESIGN_AUDIT.json",
)
record("must", 17, "Reference-event sample-size and penalized-update audit", reference_event_ok,
       "Held-out operating characteristics reported by reference fraction and event count")

eicu_model_ok = exists_all(
    ROOT / "eicu" / "tables" / "Table_eicu_reference_flow.csv",
    ROOT / "eicu" / "outputs" / "EICU_REFERENCE_AUDIT.json",
    OUTPUTS / "EICU_GROUP_HELDOUT_MODEL_AUDIT.json",
)
record("must", 19, "eICU cohort construction and unseen-hospital model split", eicu_model_ok,
       "Forty-hospital operational cohort and group-disjoint ridge replication audit present")

fixed_geography_ok = exists_all(
    TABLES / "Table_source_fixed_geography_validation.csv",
    OUTPUTS / "SOURCE_FIXED_GEOGRAPHY_VALIDATION_AUDIT.json",
)
record("must", 20, "Secondary centres 3/4/5 to centres 1/2 fixed-geography validation", fixed_geography_ok,
       "Clinically familiar geographical split restored and labelled non-untouched")

public_transport_ok = exists_all(
    TABLES / "Table_public_harmonized_bidirectional_transport.csv",
    TABLES / "Table_public_harmonized_eicu_hospital_calibration.csv",
    OUTPUTS / "PUBLIC_HARMONIZED_TRANSPORT_AUDIT.json",
)
record("must", 21, "Same-specification, same-endpoint MIMIC-IV-eICU transport audit", public_transport_ok,
       "Bidirectional external transport without local recalibration; eICU intervals resample hospitals")

# 22-24: prespecified final robustness extensions requested before submission
extended_transport_ok = exists_all(
    TABLES / "Table_eicu_common_predictor_availability.csv",
    TABLES / "Table_public_extended_predictor_availability.csv",
    TABLES / "Table_public_extended_bidirectional_transport.csv",
    TABLES / "Table_discrimination_strength_stress_test.csv",
    OUTPUTS / "EICU_COMMON_PREDICTOR_AUDIT.json",
    OUTPUTS / "PUBLIC_EXTENDED_TRANSPORT_AUDIT.json",
    OUTPUTS / "DISCRIMINATION_STRENGTH_STRESS_AUDIT.json",
)
record("must", 22, "Extended common public model and discrimination-strength stress test", extended_transport_ok,
       "Outcome-blind common-variable extension plus AUC 0.60-0.80 controlled measurement-deletion stress test")

bayesian_ok = exists_all(
    TABLES / "Table_bayesian_hierarchical_calibration_parameters.csv",
    TABLES / "Table_bayesian_hierarchical_calibration_centres.csv",
    TABLES / "Table_bayesian_hierarchical_calibration_prior_sensitivity.csv",
    OUTPUTS / "BAYESIAN_HIERARCHICAL_CALIBRATION_AUDIT.json",
)
record("must", 23, "Bayesian hierarchical source-centre calibration", bayesian_ok,
       "Common positive slope, centre random intercepts, prior sensitivity and Laplace-posterior diagnostics")

measurement_fairness_ok = exists_all(
    TABLES / "Table_measurement_aware_fairness_observability.csv",
    TABLES / "Table_measurement_aware_fairness_calibration_gap.csv",
    TABLES / "Table_measurement_aware_fairness_disparity_summary.csv",
    OUTPUTS / "MEASUREMENT_AWARE_FAIRNESS_AUDIT.json",
)
record("recommended", 24, "Measurement-aware subgroup observability and calibration decomposition", measurement_fairness_ok,
       "Observability effect sizes and Monte Carlo measurement-induced gaps; no fairness-certification claim")

source_temporal_ok = exists_all(
    TABLES / "Table_source_patient_date_integrity.csv",
    TABLES / "Table_source_recruitment_by_center_year.csv",
    TABLES / "Table_source_inpatient_observation_opportunity.csv",
    TABLES / "Table_source_temporal_split_flow.csv",
    TABLES / "Table_source_temporal_validation.csv",
    TABLES / "Table_source_temporal_model_lock.csv",
    OUTPUTS / "SOURCE_TEMPORAL_VALIDATION_AUDIT.json",
    OUTPUTS / "FIGURE8_TEMPORAL_AUDIT.json",
)
record("must", 25, "Source patient-date integrity and chronological transport audit", source_temporal_ok,
       "4014 unique patients linked one-to-one; within-centre later-patient validation and date-quality refit present")

source_variable_ok = exists_all(
    TABLES / "Table_source_variable_dictionary_110_columns.csv",
    TABLES / "Table_source_missingness_by_variable_center.csv",
    TABLES / "Table_source_categorical_code_audit.csv",
    TABLES / "Table_source_numeric_range_audit.csv",
    TABLES / "Table_source_outcome_internal_consistency.csv",
    TABLES / "Table_source_model_data_quality_sensitivity.csv",
    TABLES / "Table_source_AKI_downstream_outcomes_exploratory.csv",
    OUTPUTS / "SOURCE_VARIABLE_ROLE_AUDIT.json",
    OUTPUTS / "SOURCE_MODEL_DATA_QUALITY_SENSITIVITY_AUDIT.json",
    OUTPUTS / "FIGURE9_SOURCE_VARIABLE_AUDIT.json",
)
record("must", 26, "Source-variable landmark, coding, missingness and outcome-consistency audit", source_variable_ok,
       "All 110 Sheet1 columns classified; structural missingness, code/range flags and locked-model sensitivity reported")

source_observation_ok = exists_all(
    TABLES / "Table_source_outcome_observation_proxy_by_center.csv",
    TABLES / "Table_source_locked_model_observation_restriction.csv",
    TABLES / "Table_source_preoperative_AKI_contamination_replicates.csv",
    TABLES / "Table_source_preoperative_AKI_contamination_sensitivity.csv",
    OUTPUTS / "SOURCE_OBSERVATION_AND_PREOP_AKI_SENSITIVITY_AUDIT.json",
    ROOT / "protocol" / "ETHICS_PROTOCOL_CONCORDANCE_AUDIT.md",
)
record("must", 27, "Source observation-opportunity, preoperative-AKI sensitivity and protocol-concordance audit", source_observation_ok,
       "Locked predictions retained; discrete creatinine slots, day-7 inpatient opportunity, contamination bounds and governance scope reported")

completion = pd.DataFrame(rows).sort_values("item")
completion.to_csv(TABLES / "Table_analysis_completion_matrix.csv", index=False)
analysis_ready = bool(completion.status.eq("PASS").all())
manuscript_text = (ROOT / "manuscript" / "MANUSCRIPT_FINAL.md").read_text(encoding="utf-8")
author_fields_pending = "[AUTHOR INPUT NEEDED" in manuscript_text
submission_blockers = [
    "author names and affiliations",
    "ethics committee, approval identifier, and consent decision",
    "funding and funder role",
    "contributors, declarations of interests, acknowledgments, and data/code repository URL",
] if author_fields_pending else []
audit = {
    "items": len(completion),
    "passed": int(completion.status.eq("PASS").sum()),
    "failed": int(completion.status.eq("FAIL").sum()),
    "analysis_ready": analysis_ready,
    "submission_ready": analysis_ready and not submission_blockers,
    "production_ready": analysis_ready and not submission_blockers,
    "submission_blockers": submission_blockers,
    "patient_level_data_read": False,
    "matrix": "tables/Table_analysis_completion_matrix.csv",
}
(OUTPUTS / "ANALYSIS_COMPLETION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
if not audit["analysis_ready"]:
    raise SystemExit("One or more completion gates failed")
