# %% [markdown]
# # Extract publication-level key results
# Produces only aggregate tables and a machine-readable fact base for manuscript writing.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import pandas as pd

ROOT = Path(str(_release_path('analysis')))
TABLES, OUTPUTS = ROOT / "tables", ROOT / "outputs"

precision = pd.read_csv(TABLES / "Table_source_model_precision.csv")
centers = pd.read_csv(TABLES / "Table_source_center_performance_complete.csv")
increment = pd.read_csv(TABLES / "Table_preop_to_perioperative_increment.csv")
stability = pd.read_csv(TABLES / "Table_source_model_stability_200bootstrap.csv")
utility = pd.read_csv(TABLES / "Table_monitoring_threshold_burden_capture.csv")
fairness = pd.read_csv(TABLES / "Table_fairness_bootstrap_intervals.csv")
public = pd.read_csv(TABLES / "Table_public_reference_precision.csv")
fixed_geography = pd.read_csv(TABLES / "Table_source_fixed_geography_validation.csv")
harmonized_transport = pd.read_csv(TABLES / "Table_public_harmonized_bidirectional_transport.csv")
extended_transport = pd.read_csv(TABLES / "Table_public_extended_bidirectional_transport.csv")
discrimination_stress = pd.read_csv(TABLES / "Table_discrimination_strength_stress_test.csv")
bayesian_parameters = pd.read_csv(TABLES / "Table_bayesian_hierarchical_calibration_parameters.csv")
bayesian_centres = pd.read_csv(TABLES / "Table_bayesian_hierarchical_calibration_centres.csv")
measurement_observability = pd.read_csv(TABLES / "Table_measurement_aware_fairness_observability.csv")
measurement_fairness = pd.read_csv(TABLES / "Table_measurement_aware_fairness_calibration_gap.csv")
sim = pd.concat(
    [pd.read_csv(TABLES / "Table_inspire_simulation_summary.csv"),
     pd.read_csv(TABLES / "Table_mimic_simulation_summary.csv"),
     pd.read_csv(TABLES / "Table_eicu_simulation_summary.csv")], ignore_index=True
)

core_methods = {
    "full_reference", "naive", "IPAW_design_probability_untruncated",
    "IPAW_design_probability_truncated99", "AIPW_design_probability",
    "recalibration_intercept_apparent", "recalibration_intercept_truth",
    "recalibration_intercept_slope_apparent", "recalibration_intercept_slope_truth",
    "Gamma2_prediction_sensitivity_region",
}
if sim.loc[sim.method.isin(core_methods), "n_replicates"].min() < 300:
    raise SystemExit("Production key-results extraction requires 300 core replicates per condition")

model_rows = precision[
    ((precision.feature_set == "PI") & precision.model.isin(["ridge", "restricted_rf", "gradient_boosting"]))
    | ((precision.feature_set == "H") & (precision.model == "restricted_rf"))
].copy()
model_cols = [
    "database", "feature_set", "model", "n", "events", "roc_auc", "roc_auc_ci_lower",
    "roc_auc_ci_upper", "brier", "brier_ci_lower", "brier_ci_upper", "oe_ratio",
    "oe_ratio_ci_lower", "oe_ratio_ci_upper", "calibration_slope",
    "calibration_slope_ci_lower", "calibration_slope_ci_upper",
]
model_rows[model_cols].to_csv(TABLES / "Table_key_source_model_results.csv", index=False)

sim_methods = [
    "naive", "IPAW_design_probability_untruncated", "IPAW_design_probability_truncated99",
    "AIPW_design_probability",
    "recalibration_intercept_truth", "recalibration_intercept_slope_truth",
    "reference_05pct_recalibration", "reference_10pct_recalibration",
    "reference_20pct_recalibration", "reference_30pct_recalibration",
    "Gamma2_prediction_sensitivity_region",
]
key_sim = sim[
    (sim.mechanism == "mixed_MNAR") & (sim.strength == "strong")
    & sim.method.isin(sim_methods)
    & sim.metric.isin(["event_rate", "oe", "reconstructed_sensitivity", "MNAR_event_rate_coverage", "event_rate_interval_coverage"])
].copy()
key_sim.to_csv(TABLES / "Table_key_mixed_mnar_results.csv", index=False)

recal = sim[
    (sim.mechanism == "mixed_MNAR") & (sim.strength == "strong")
    & sim.method.isin(["recalibration_intercept_slope_apparent", "recalibration_intercept_slope_truth"])
    & sim.metric.isin(["oe", "calibration_intercept", "calibration_slope"])
].copy()
recal.to_csv(TABLES / "Table_key_apparent_vs_reference_recalibration.csv", index=False)

global_rmse = (
    sim[(sim.strength == "strong") & sim.method.isin(sim_methods)
        & sim.metric.isin(["event_rate", "oe"])]
    .groupby(["database", "method", "metric"], as_index=False)
    .agg(mean_rmse=("rmse", "mean"), worst_rmse=("rmse", "max"))
)
global_rmse.to_csv(TABLES / "Table_key_strategy_robustness.csv", index=False)

reference_design = sim[
    sim.method.str.startswith("reference_")
    & sim.metric.isin(["oe", "calibration_intercept", "calibration_slope"])
].copy()
reference_design.to_csv(TABLES / "Table_key_reference_sample_design.csv", index=False)

selection_control = pd.concat(
    [pd.read_csv(TABLES / "Table_inspire_pure_label_selection_control.csv"),
     pd.read_csv(TABLES / "Table_mimic_pure_label_selection_control.csv"),
     pd.read_csv(TABLES / "Table_eicu_pure_label_selection_control.csv")],
    ignore_index=True,
)
selection_control.to_csv(TABLES / "Table_key_pure_label_selection_control.csv", index=False)

kdigo_components = {
    "inspire": pd.read_csv(TABLES / "Table_inspire_kdigo_component_availability.csv").to_dict("records"),
    "mimic": pd.read_csv(TABLES / "Table_mimic_kdigo_component_availability.csv").to_dict("records"),
    "mimic_endpoint_target": pd.read_csv(TABLES / "Table_mimic_endpoint_target_performance.csv").to_dict("records"),
    "eicu": pd.read_csv(TABLES / "Table_eicu_kdigo_component_availability.csv").to_dict("records"),
}

facts = {
    "source_primary": {
        "n": 4014,
        "events": 155,
        "event_rate": 155 / 4014,
        "primary_role": "observable source cohort",
        "historical_3710_role": "sensitivity analysis",
    },
    "source_pi_restricted_rf": model_rows[(model_rows.feature_set == "PI") & (model_rows.model == "restricted_rf")].iloc[0].to_dict(),
    "source_h_restricted_rf": model_rows[(model_rows.feature_set == "H") & (model_rows.model == "restricted_rf")].iloc[0].to_dict(),
    "source_center_event_counts": centers.groupby("center").first()[["n", "events"]].reset_index().to_dict("records"),
    "incremental_auc": increment[increment.metric == "auc_difference"].to_dict("records"),
    "stability_medians": stability[stability.metric.isin(["risk_spearman_vs_full_fit", "top20_jaccard_vs_full_fit"])].to_dict("records"),
    "top20_monitoring": utility[(utility.database == "source_4014") & (utility.policy == "top_fraction") & (utility.policy_value == 0.2) & (utility.additional_tests_per_selected == 1)].to_dict("records"),
    "public_reference_precision": public.to_dict("records"),
    "source_fixed_geography_validation": fixed_geography.to_dict("records"),
    "public_harmonized_bidirectional_transport": harmonized_transport.to_dict("records"),
    "public_extended_bidirectional_transport": extended_transport.to_dict("records"),
    "discrimination_strength_stress_test": discrimination_stress.to_dict("records"),
    "bayesian_hierarchical_calibration_parameters": bayesian_parameters.to_dict("records"),
    "bayesian_hierarchical_calibration_centres": bayesian_centres.to_dict("records"),
    "measurement_aware_observability": measurement_observability.to_dict("records"),
    "measurement_aware_calibration_gap": measurement_fairness.to_dict("records"),
    "subgroup_auc": fairness[(fairness.metric == "auc") & (fairness.inference_status == "bootstrap_500")].to_dict("records"),
    "simulation_mixed_mnar": key_sim.to_dict("records"),
    "apparent_vs_reference_recalibration": recal.to_dict("records"),
    "strategy_robustness": global_rmse.to_dict("records"),
    "reference_sample_design": reference_design.to_dict("records"),
    "pure_label_selection_control": selection_control.to_dict("records"),
    "public_kdigo_component_audit": kdigo_components,
    "claim_boundaries": [
        "The source PostopAKI outcome used site-level dual-nephrologist KDIGO 2012 adjudication with masked review and third-nephrologist resolution, but case-level component forms and agreement statistics are unavailable in the analytic extract.",
        "Public endpoints are operational creatinine endpoints, not full KDIGO or biological ground truth.",
        "INSPIRE urine-output density is insufficient for duration-based KDIGO reconstruction; MIMIC and eICU multicomponent endpoints are algorithmic and not clinician-adjudicated.",
        "Database-native MIMIC and eICU models are methodological replications; the harmonized minimal public model is transported under a common ICU landmark and endpoint but does not validate the five-centre source model.",
        "Monitoring policies are decision-analytic scenarios, not an observed impact study.",
        "Subgroup analyses are representativeness/performance audits, not fairness certification.",
        "Synthetic AUC-controlled scores are simulation design objects, not deployable prediction models.",
        "Bayesian centre calibration uses a joint-mode Laplace approximation and is conditional on locked predictions.",
    ],
}
(OUTPUTS / "PUBLICATION_FACT_BASE.json").write_text(json.dumps(facts, indent=2, default=float) + "\n")
print(json.dumps({"fact_base": "outputs/PUBLICATION_FACT_BASE.json", "aggregate_table_families": 13}, indent=2))
