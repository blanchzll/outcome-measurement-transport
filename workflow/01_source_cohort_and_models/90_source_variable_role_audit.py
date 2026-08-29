# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Source-variable dictionary, landmark, coding, and outcome-consistency audit
#
# The authoritative workbook is used only for its header. Direct identifiers and exact
# dates are never exported. Patient-level values come from the deidentified 4014-patient
# cohort and only aggregate audit tables leave the secure workspace.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(str(_release_path('source')))
ROOT = Path(str(_release_path('analysis')))
WORKBOOK = BASE / "source_data" / "20260823" / "inter3.xlsx"
COHORT = BASE / "secure_source" / "inter3_deidentified_4014.csv"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
SEED = 20260828
N_BOOTSTRAP = 2000

sys.path.insert(0, str(ROOT / "code"))
from source_temporal import read_authoritative_sheet1_header  # noqa: E402


# %%
PREOPERATIVE = {
    "Gender", "Age", "Height", "Weight", "BMI", "WeightLoss",
    "PreviousAbdominalSurgery", "OtherMalignancy", "Comorbidities", "Diabetes",
    "Hypertension", "CerebrovascularDisease", "HeartDisease", "Smoking", "Alcohol",
    "NeoadjuvantChemo", "ASAGrade", "PreopWBC", "PreopHb", "PreopAlb", "PreopALT",
    "PreopBUN", "PreopCr", "PreopGlucose",
}
INTRAOPERATIVE = {
    "SurgicalApproach", "GastricResectionSite", "CombinedOrganResection",
    "DistalGastrectomy", "ProximalGastrectomy", "TotalGastrectomy",
    "ColorectalResectionSite", "Stoma", "OperationTime", "IntraopBloodLoss",
    "IntraopTransfusion", "IntraopFluid", "IntraopHES", "IntraopDextran",
    "IntraopGelatin", "IntraopPlasma", "IntraopColloid", "IntraopDiuretics",
    "IntraopVasoactive",
}
AMBIGUOUS_TIMING = {
    "ICUAdmission", "VentilatorUse", "PCAUse", "EpiduralAnalgesia",
    "VasoactiveDrugs", "AminoglycosideUse", "NonSelectiveCOXInhibitor",
    "COX2Inhibitor", "NasogastricTube", "DrainTube", "T_Stage", "N_Stage",
    "M_Stage", "TNM_Stage",
}
POSTLANDMARK = {
    "HospitalDays", "PostopHospitalDays", "PreopHospitalDays", "ICUDays", "RRT",
    "Reoperation30d", "Readmission30d", "Mortality90d", "NonOpTransfusion",
    "NonOpHES", "NonOpDextran", "NonOpGelatin", "NonOpAlbumin", "NonOpPlasma",
    "NonOpColloid", "HospitalDiuretics", "PerioperativeVasoactive", "TubeRemovalDay",
    "DrainRemovalDay", "FirstLiquidDietDay", "SurgicalComplications",
    "InfectionComplications", "Fistula", "MotilityDisorder", "Bleeding", "CD3",
    "LymphNodesExamined", "PositiveLymphNodes", "PostopPOD1_WBC", "PreopPOD1_Hb",
    "PostopPOD1_Alb", "PostopPOD1_ALT", "PostopPOD1_BUN", "PostopPOD1_Cr",
    "PostopPOD2_POD3_WBC", "PostopPOD2_POD3_Hb", "PostopPOD2_POD3_Alb",
    "PostopPOD2_POD3_ALT", "PostopPOD2_POD3_BUN", "PostopPOD2_POD3_Cr",
    "PostopMaxGlucose",
}

P_FEATURES = {
    "Age", "PreopCr", "PreopHb", "PreopAlb", "Gender", "Diabetes",
    "Hypertension", "CerebrovascularDisease", "HeartDisease", "Gastrocolorectal",
    "NeoadjuvantChemo",
}
PI_ADDITIONS = {"IntraopTransfusion", "SurgicalApproach", "CombinedOrganResection"}

UNITS = {
    "Age": "years", "HospitalDays": "days", "PostopHospitalDays": "days",
    "PreopHospitalDays": "days", "ICUDays": "days", "Height": "cm", "Weight": "kg",
    "BMI": "kg/m2", "WeightLoss": "kg", "OperationTime": "min",
    "IntraopBloodLoss": "mL", "IntraopTransfusion": "mL", "IntraopFluid": "mL",
    "NonOpTransfusion": "mL", "IntraopHES": "mL", "IntraopDextran": "mL",
    "IntraopGelatin": "mL", "IntraopPlasma": "mL", "IntraopColloid": "mL",
    "NonOpHES": "mL", "NonOpDextran": "mL", "NonOpGelatin": "mL",
    "NonOpAlbumin": "g", "NonOpPlasma": "mL", "NonOpColloid": "mL",
    "PreopWBC": "10^9/L (expected)", "PreopHb": "g/L (expected)",
    "PreopAlb": "g/L (expected)", "PreopALT": "U/L (expected)",
    "PreopBUN": "mmol/L (expected)", "PreopCr": "umol/L (expected)",
    "PreopGlucose": "mmol/L", "PostopPOD1_Cr": "umol/L (expected)",
    "PostopPOD2_POD3_Cr": "umol/L (expected)",
}

EXPECTED_CODES = {
    "Center": {1, 2, 3, 4, 5}, "GastricColorectal": {0, 1, 2},
    "Gastrocolorectal": {1, 2}, "PostopAKI": {0, 1}, "AKIStage": {0, 1, 2, 3},
    "Gender": {0, 1, 2}, "ICUAdmission": {0, 1}, "VentilatorUse": {0, 1},
    "RRT": {0, 1}, "Reoperation30d": {0, 1}, "Readmission30d": {0, 1},
    "Mortality90d": {0, 1}, "PreviousAbdominalSurgery": {0, 1},
    "OtherMalignancy": {0, 1}, "Diabetes": {0, 1}, "Hypertension": {0, 1},
    "CerebrovascularDisease": {0, 1}, "HeartDisease": {0, 1}, "Smoking": {0, 1},
    "Alcohol": {0, 1}, "NeoadjuvantChemo": {0, 1}, "ASAGrade": {1, 2, 3, 4},
    "SurgicalApproach": {1, 2, 3, 4}, "CombinedOrganResection": {0, 1},
    "Stoma": {0, 1}, "PCAUse": {0, 1}, "EpiduralAnalgesia": {0, 1},
    "IntraopDiuretics": {0, 1}, "HospitalDiuretics": {0, 1},
    "IntraopVasoactive": {0, 1}, "PerioperativeVasoactive": {0, 1},
    "AminoglycosideUse": {0, 1}, "NonSelectiveCOXInhibitor": {0, 1},
    "COX2Inhibitor": {0, 1}, "NasogastricTube": {0, 1}, "DrainTube": {0, 1},
    "SurgicalComplications": {0, 1}, "InfectionComplications": {0, 1},
    "Fistula": {0, 1}, "MotilityDisorder": {0, 1}, "Bleeding": {0, 1}, "CD3": {0, 1},
}

PLAUSIBLE_RANGES = {
    "Age": (16, 105), "Height": (120, 210), "Weight": (25, 180), "BMI": (10, 60),
    "OperationTime": (15, 1200), "IntraopBloodLoss": (0, 20000),
    "IntraopTransfusion": (0, 20000), "IntraopFluid": (0, 30000),
    "PreopWBC": (1, 100), "PreopHb": (30, 220), "PreopAlb": (10, 70),
    "PreopALT": (0, 5000), "PreopBUN": (0.5, 50), "PreopCr": (20, 1500),
    "PreopGlucose": (1, 50),
}

MISSING_TOKENS = {"", "_", "/", "na", "n/a", "nan", "none", "null", "濂"}


def effective_missing(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.casefold()
    return series.isna() | text.isin(MISSING_TOKENS)


def role_for(variable: str) -> tuple[str, str]:
    if variable == "Center":
        return "cluster_identifier", "audit_and_validation_split_only"
    if variable in {"GastricColorectal", "Gastrocolorectal"}:
        return "case_mix_descriptor", "eligible_if_coding_is_resolved"
    if variable == "MajorID":
        return "stable_identifier", "audit_only_never_predictor"
    if variable == "PostopAKI":
        return "primary_outcome", "outcome_never_predictor"
    if variable == "AKIStage":
        return "secondary_outcome", "outcome_never_predictor"
    if variable in PREOPERATIVE:
        return "preoperative_candidate", "eligible_at_end_of_surgery"
    if variable in INTRAOPERATIVE:
        return "intraoperative_candidate", "eligible_at_end_of_surgery"
    if variable in AMBIGUOUS_TIMING:
        return "timing_ambiguous", "exclude_until_timestamp_confirmed"
    if variable in POSTLANDMARK:
        return "postlandmark_or_outcome_adjacent", "exclude_label_leakage"
    raise ValueError(f"Unclassified source variable: {variable}")


def current_model_role(variable: str) -> str:
    if variable in P_FEATURES:
        return "preoperative_primary"
    if variable in PI_ADDITIONS:
        return "perioperative_increment"
    return "not_in_primary_models"


def center_stratified_bootstrap(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    rng = np.random.default_rng(SEED + sum(map(ord, outcome)))
    draws = []
    groups = [group.reset_index(drop=True) for _, group in frame.groupby("Center")]
    for _ in range(N_BOOTSTRAP):
        sampled = pd.concat(
            [group.iloc[rng.integers(0, len(group), len(group))] for group in groups],
            ignore_index=True,
        )
        observed = sampled.dropna(subset=[outcome])
        rates = observed.groupby("PostopAKI")[outcome].mean()
        if not {0, 1}.issubset(rates.index):
            continue
        rate0, rate1 = float(rates[0]), float(rates[1])
        draws.append((rate1 - rate0, rate1 / rate0 if rate0 > 0 else np.nan))
    values = np.asarray(draws, dtype=float)
    return {
        "risk_difference_ci_lower": float(np.nanquantile(values[:, 0], 0.025)),
        "risk_difference_ci_upper": float(np.nanquantile(values[:, 0], 0.975)),
        "risk_ratio_ci_lower": float(np.nanquantile(values[:, 1], 0.025)),
        "risk_ratio_ci_upper": float(np.nanquantile(values[:, 1], 0.975)),
    }


# %%
raw_headers = read_authoritative_sheet1_header(WORKBOOK)
cohort = pd.read_csv(COHORT, low_memory=False)
if len(raw_headers) != 110:
    raise ValueError(f"Expected 110 Sheet1 headers, found {len(raw_headers)}.")
if cohort.shape != (4014, 104):
    raise ValueError(f"Expected a 4014 by 104 deidentified cohort, found {cohort.shape}.")
if cohort["MajorID"].nunique(dropna=False) != 4014:
    raise ValueError("MajorID must be complete and unique.")

deidentified = list(cohort.columns)
position_map: dict[int, str | None] = {}
for raw_position in range(1, 111):
    if raw_position <= 4:
        position_map[raw_position] = deidentified[raw_position - 1]
    elif 5 <= raw_position <= 10:
        position_map[raw_position] = None
    else:
        position_map[raw_position] = deidentified[raw_position - 7]
if position_map[110] != deidentified[-1]:
    raise AssertionError("Raw-to-deidentified positional mapping failed.")

special_raw = {
    5: ("source_row_number", "direct_identifier", "restricted_never_exported"),
    6: ("medical_record_number", "direct_identifier", "restricted_never_exported"),
    7: ("patient_name", "direct_identifier", "restricted_never_exported"),
    8: ("AdmissionDate", "date_audit_only", "audit_only_never_predictor"),
    9: ("SurgeryDate", "date_audit_only", "temporal_split_only_never_predictor"),
    10: ("DischargeDate", "date_audit_only", "observation_opportunity_only"),
}

dictionary_rows = []
for raw_position, raw_label in enumerate(raw_headers, 1):
    variable = position_map[raw_position]
    if variable is None:
        canonical, role, policy = special_raw[raw_position]
        dictionary_rows.append({
            "source_column": raw_position, "source_label_zh": raw_label,
            "canonical_variable": canonical, "analysis_role": role,
            "landmark_policy": policy, "current_model_role": "not_in_primary_models",
            "unit": "date" if "Date" in canonical else "identifier",
            "n_nonmissing": np.nan, "n_missing_effective": np.nan,
            "missing_rate_effective": np.nan, "n_unique_nonmissing": np.nan,
            "deidentified_export": False,
        })
        continue
    role, policy = role_for(variable)
    missing = effective_missing(cohort[variable])
    dictionary_rows.append({
        "source_column": raw_position, "source_label_zh": raw_label,
        "canonical_variable": variable, "analysis_role": role,
        "landmark_policy": policy, "current_model_role": current_model_role(variable),
        "unit": UNITS.get(variable, "categorical_or_unspecified"),
        "n_nonmissing": int((~missing).sum()), "n_missing_effective": int(missing.sum()),
        "missing_rate_effective": float(missing.mean()),
        "n_unique_nonmissing": int(cohort.loc[~missing, variable].nunique()),
        "deidentified_export": True,
    })

dictionary = pd.DataFrame(dictionary_rows)
dictionary.to_csv(TABLES / "Table_source_variable_dictionary_110_columns.csv", index=False)

role_summary = (
    dictionary.groupby(["analysis_role", "landmark_policy"], dropna=False)
    .agg(n_source_columns=("source_column", "size"), n_in_primary_models=("current_model_role", lambda x: int((x != "not_in_primary_models").sum())))
    .reset_index()
)
role_summary.to_csv(TABLES / "Table_source_variable_role_summary.csv", index=False)

missing_rows = []
for variable in deidentified:
    role, policy = role_for(variable)
    for center, group in cohort.groupby("Center"):
        missing = effective_missing(group[variable])
        missing_rows.append({
            "variable": variable, "analysis_role": role, "landmark_policy": policy,
            "center": int(center), "n": int(len(group)),
            "n_missing_effective": int(missing.sum()), "missing_rate_effective": float(missing.mean()),
        })
pd.DataFrame(missing_rows).to_csv(TABLES / "Table_source_missingness_by_variable_center.csv", index=False)

code_rows = []
for variable, allowed in EXPECTED_CODES.items():
    missing = effective_missing(cohort[variable])
    numeric = pd.to_numeric(cohort[variable].where(~missing), errors="coerce")
    invalid = (~missing) & (~numeric.isin(sorted(allowed)))
    values = sorted(cohort.loc[invalid, variable].astype(str).unique().tolist())
    code_rows.append({
        "variable": variable, "allowed_codes": "|".join(map(str, sorted(allowed))),
        "n_effectively_missing": int(missing.sum()), "n_invalid_nonmissing": int(invalid.sum()),
        "invalid_values_aggregate": "|".join(values[:20]),
        "status": "PASS" if not invalid.any() else "DATA_QUALITY_FLAG",
    })
pd.DataFrame(code_rows).to_csv(TABLES / "Table_source_categorical_code_audit.csv", index=False)

range_rows = []
for variable, (lower, upper) in PLAUSIBLE_RANGES.items():
    values = pd.to_numeric(cohort[variable], errors="coerce")
    below, above = values.lt(lower), values.gt(upper)
    range_rows.append({
        "variable": variable, "expected_unit": UNITS.get(variable, "unspecified"),
        "plausible_lower": lower, "plausible_upper": upper,
        "n_numeric": int(values.notna().sum()), "n_below": int(below.sum()),
        "n_above": int(above.sum()), "minimum": float(values.min()),
        "p01": float(values.quantile(0.01)), "median": float(values.median()),
        "p99": float(values.quantile(0.99)), "maximum": float(values.max()),
        "status": "PASS" if not (below | above).any() else "SOURCE_VERIFICATION_NEEDED",
    })
pd.DataFrame(range_rows).to_csv(TABLES / "Table_source_numeric_range_audit.csv", index=False)

redundancy = pd.DataFrame([
    {"group": "cancer_site", "variables": "GastricColorectal|Gastrocolorectal", "finding": "The first field is a partially observed gastric-colon-rectum code; the second is complete gastric-versus-colorectal.", "primary_action": "Use Gastrocolorectal; retain the three-level field only as a missingness-described sensitivity."},
    {"group": "body_size", "variables": "Height|Weight|BMI", "finding": "BMI is derived and structurally missing in centre 1 and much of centre 4.", "primary_action": "Do not enter all three together; BMI is excluded from the primary model."},
    {"group": "cardiovascular_history", "variables": "CerebrovascularDisease|HeartDisease", "finding": "The primary model uses their prespecified union.", "primary_action": "Retain the union to limit parameters."},
    {"group": "intraoperative_colloid", "variables": "IntraopHES|IntraopDextran|IntraopGelatin|IntraopPlasma|IntraopColloid", "finding": "Component and total fields may overlap.", "primary_action": "Do not enter component and total amounts together without a source-derived reconciliation rule."},
    {"group": "nonoperative_colloid", "variables": "NonOpHES|NonOpDextran|NonOpGelatin|NonOpAlbumin|NonOpPlasma|NonOpColloid", "finding": "These occur after the prediction landmark and may overlap.", "primary_action": "Exclude from all end-of-surgery prediction models."},
    {"group": "haemoglobin_label", "variables": "PreopPOD1_Hb", "finding": "The Chinese header says preoperative POD1 Hb, an internally contradictory label placed among postoperative labs.", "primary_action": "Treat as postoperative and exclude until the source system confirms timing."},
])
redundancy.to_csv(TABLES / "Table_source_redundancy_and_timing_decisions.csv", index=False)

# %%
stage = pd.to_numeric(cohort["AKIStage"], errors="coerce")
aki = pd.to_numeric(cohort["PostopAKI"], errors="coerce")
rrt = pd.to_numeric(cohort["RRT"], errors="coerce")
outcome_rows = []
for center, group in cohort.groupby("Center"):
    group_stage = pd.to_numeric(group["AKIStage"], errors="coerce")
    group_aki = pd.to_numeric(group["PostopAKI"], errors="coerce")
    group_rrt = pd.to_numeric(group["RRT"], errors="coerce")
    mismatch = ((group_aki == 0) & (group_stage > 0)) | ((group_aki == 1) & (group_stage == 0))
    outcome_rows.append({
        "center": int(center), "n": int(len(group)), "postop_aki_events": int(group_aki.sum()),
        "stage_0": int((group_stage == 0).sum()), "stage_1": int((group_stage == 1).sum()),
        "stage_2": int((group_stage == 2).sum()), "stage_3": int((group_stage == 3).sum()),
        "aki_stage_binary_mismatch": int(mismatch.sum()), "rrt_code_1": int((group_rrt == 1).sum()),
        "rrt_invalid_code_2_to_5": int(group_rrt.isin([2, 3, 4, 5]).sum()),
        "rrt_code_1_among_non_aki": int(((group_rrt == 1) & (group_aki == 0)).sum()),
    })
pd.DataFrame(outcome_rows).to_csv(TABLES / "Table_source_outcome_internal_consistency.csv", index=False)

downstream_rows = []
for outcome in ("Reoperation30d", "Readmission30d", "Mortality90d"):
    analysis = cohort[["Center", "PostopAKI", outcome]].copy()
    analysis[outcome] = pd.to_numeric(analysis[outcome], errors="coerce")
    observed = analysis.dropna(subset=[outcome])
    summary = observed.groupby("PostopAKI")[outcome].agg(["size", "sum", "mean"])
    rate0, rate1 = float(summary.loc[0, "mean"]), float(summary.loc[1, "mean"])
    intervals = center_stratified_bootstrap(analysis, outcome)
    downstream_rows.append({
        "outcome": outcome, "non_aki_n_observed": int(summary.loc[0, "size"]),
        "non_aki_events": int(summary.loc[0, "sum"]), "non_aki_rate": rate0,
        "aki_n_observed": int(summary.loc[1, "size"]), "aki_events": int(summary.loc[1, "sum"]),
        "aki_rate": rate1, "risk_difference": rate1 - rate0,
        "risk_ratio": rate1 / rate0 if rate0 > 0 else np.nan, **intervals,
        "inference_boundary": "exploratory descriptive association; not causal or incremental utility evidence",
    })
pd.DataFrame(downstream_rows).to_csv(TABLES / "Table_source_AKI_downstream_outcomes_exploratory.csv", index=False)

stage_rows = []
for stage_value, group in cohort.assign(AKIStage_numeric=stage).groupby("AKIStage_numeric"):
    row = {"aki_stage": int(stage_value), "n": int(len(group))}
    for outcome in ("Reoperation30d", "Readmission30d", "Mortality90d"):
        values = pd.to_numeric(group[outcome], errors="coerce")
        row[f"{outcome}_n_observed"] = int(values.notna().sum())
        row[f"{outcome}_events"] = int(values.sum(skipna=True))
        row[f"{outcome}_rate"] = float(values.mean())
    stage_rows.append(row)
pd.DataFrame(stage_rows).to_csv(TABLES / "Table_source_AKI_stage_downstream_outcomes.csv", index=False)

# %%
missing_by_center = pd.DataFrame(missing_rows)
key_structural = {}
for variable in ("BMI", "ASAGrade", "OperationTime", "IntraopVasoactive", "IntraopFluid"):
    rows = missing_by_center.loc[missing_by_center.variable.eq(variable)]
    key_structural[variable] = {str(int(row.center)): float(row.missing_rate_effective) for row in rows.itertuples()}

stage_mismatch = ((aki == 0) & (stage > 0)) | ((aki == 1) & (stage == 0))
audit = {
    "source_rows": int(len(cohort)), "source_columns_raw": len(raw_headers),
    "source_columns_deidentified": int(cohort.shape[1]), "major_id_unique": int(cohort.MajorID.nunique()),
    "direct_identifier_columns_not_exported": [5, 6, 7],
    "date_columns_audit_only": [8, 9, 10],
    "primary_model_predictor_count": int(dictionary.current_model_role.ne("not_in_primary_models").sum()),
    "postlandmark_or_ambiguous_columns_excluded": int(dictionary.analysis_role.isin(["postlandmark_or_outcome_adjacent", "timing_ambiguous"]).sum()),
    "aki_stage_binary_mismatches": int(stage_mismatch.sum()),
    "rrt_code_1_among_non_aki": int(((rrt == 1) & (aki == 0)).sum()),
    "rrt_invalid_codes_2_to_5": int(rrt.isin([2, 3, 4, 5]).sum()),
    "gender_effectively_unresolved": int(effective_missing(cohort.Gender).sum()),
    "gastric_colon_rectum_code_missing": int(effective_missing(cohort.GastricColorectal).sum()),
    "structural_missingness_by_center": key_structural,
    "prediction_landmark": "end of surgery",
    "primary_policy": "retain the prespecified low-dimensional P and PI feature sets; do not add postoperative or structurally absent fields",
    "patient_level_data_exported": False,
}
(OUTPUTS / "SOURCE_VARIABLE_ROLE_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(audit, indent=2, ensure_ascii=False))
