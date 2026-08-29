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
# # Source-cohort patient/date integrity and temporal validation
#
# The primary validation is a within-centre chronological 70/30 split. A fixed
# 1 Jan 2022 split is secondary because centre participation changed over time.
# Dates, discharge information, and postoperative length of stay are never used
# as predictors at the end-of-surgery landmark.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(str(_release_path('source')))
ROOT = Path(str(_release_path('analysis')))
SOURCE_WORKBOOK = BASE / "source_data" / "20260823" / "inter3.xlsx"
SOURCE_COHORT = BASE / "secure_source" / "inter3_deidentified_4014.csv"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
SECURE = ROOT / "secure_work"
N_BOOTSTRAP = 1000
SEED = 20260828

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "code"))

from analysis import CENTER, TARGET, load_cohort  # noqa: E402
from loco_analysis import (  # noqa: E402
    BASE_MODELS,
    FEATURE_SET_SPECS,
    bootstrap_metric_ci,
    build_loco_search,
    engineer_loco_features,
    probability_metrics,
)
from source_temporal import (  # noqa: E402
    fixed_calendar_split,
    read_authoritative_sheet1_dates,
    wilson_interval,
    within_centre_chronological_split,
)


def iso_date(value) -> str | None:
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


def fit_temporal_models(frame: pd.DataFrame, split: pd.Series, split_name: str):
    development = split.eq("development").to_numpy()
    validation = split.eq("validation").to_numpy()
    y_train = frame.loc[development, TARGET].astype(int).to_numpy()
    y_test = frame.loc[validation, TARGET].astype(int).to_numpy()
    groups_train = frame.loc[development, CENTER].astype(int).to_numpy()
    groups_test = frame.loc[validation, CENTER].astype(int).to_numpy()
    predictions: dict[tuple[str, str], np.ndarray] = {}
    fit_rows: list[dict] = []

    for feature_set in ("P", "PI"):
        spec = FEATURE_SET_SPECS[feature_set]
        base_probabilities = []
        for model in BASE_MODELS:
            search = build_loco_search(
                spec,
                model,
                n_inner_centers=len(np.unique(groups_train)),
                fast=False,
            )
            search.fit(
                frame.loc[development, list(spec.features)],
                y_train,
                groups=groups_train,
            )
            probability = search.predict_proba(
                frame.loc[validation, list(spec.features)]
            )[:, 1]
            predictions[(feature_set, model)] = probability
            base_probabilities.append(probability)
            fit_rows.append(
                {
                    "split_definition": split_name,
                    "feature_set": feature_set,
                    "model": model,
                    "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                    "inner_cv_best_neg_brier": float(search.best_score_),
                }
            )
        predictions[(feature_set, "soft_voting")] = np.mean(base_probabilities, axis=0)

    performance_rows: list[dict] = []
    centre_rows: list[dict] = []
    for index, ((feature_set, model), probability) in enumerate(predictions.items()):
        metrics = probability_metrics(y_test, probability)
        intervals = bootstrap_metric_ci(
            y_test,
            probability,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED + index + (100 if split_name == "calendar_2022" else 0),
            groups=groups_test,
        )
        row = {
            "split_definition": split_name,
            "feature_set": feature_set,
            "model": model,
            **metrics,
            "bootstrap_unit": "patient_within_validation_centre",
            "n_bootstrap": N_BOOTSTRAP,
        }
        for metric, (lower, upper) in intervals.items():
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        performance_rows.append(row)
        for centre in sorted(np.unique(groups_test)):
            mask = groups_test == centre
            centre_metrics = probability_metrics(y_test[mask], probability[mask])
            centre_rows.append(
                {
                    "split_definition": split_name,
                    "feature_set": feature_set,
                    "model": model,
                    "center": int(centre),
                    **centre_metrics,
                    "inference_boundary": (
                        "descriptive; centre-specific temporal events are sparse"
                    ),
                }
            )

    secure = frame.loc[
        validation, ["MajorID", CENTER, TARGET, "SurgeryYear"]
    ].copy()
    secure.insert(0, "split_definition", split_name)
    for (feature_set, model), probability in predictions.items():
        secure[f"pred_{feature_set}_{model}"] = probability
    return (
        pd.DataFrame(performance_rows),
        pd.DataFrame(centre_rows),
        pd.DataFrame(fit_rows),
        secure,
    )


# %%
dates = read_authoritative_sheet1_dates(SOURCE_WORKBOOK)
if len(dates) != 4014 or dates["MajorID"].nunique() != 4014:
    raise ValueError("Sheet1 must contain 4014 complete and unique MajorID values.")
if int(dates[TARGET].sum()) != 155:
    raise ValueError("Sheet1 outcome count does not match the locked source cohort.")
if dates["SurgeryDate"].isna().any():
    raise ValueError(
        "Every Sheet1 SurgeryDate must be parseable after documented format harmonisation."
    )

source = engineer_loco_features(load_cohort(SOURCE_COHORT)).reset_index(drop=True)
linked = source.merge(
    dates,
    on="MajorID",
    how="outer",
    validate="one_to_one",
    suffixes=("", "_sheet1"),
    indicator=True,
)
if set(linked["_merge"]) != {"both"}:
    raise ValueError("The authoritative Sheet1 and deidentified cohort are not one-to-one.")
if not linked[CENTER].eq(linked[f"{CENTER}_sheet1"]).all():
    raise ValueError("Centre codes differ after MajorID linkage.")
if not linked[TARGET].eq(linked[f"{TARGET}_sheet1"]).all():
    raise ValueError("Outcomes differ after MajorID linkage.")
linked = linked.drop(
    columns=[f"{CENTER}_sheet1", f"{TARGET}_sheet1", "_merge"]
)
linked["SurgeryYear"] = linked["SurgeryDate"].dt.year.astype(int)

# %%
date_integrity_rows = []
for item, value, status, note in (
    (
        "analytic_patients",
        len(linked),
        "PASS",
        "one operation per unique MajorID; investigator-confirmed unique patients",
    ),
    ("unique_MajorID", linked["MajorID"].nunique(), "PASS", "complete and unique"),
    ("Sheet1_to_analysis_ID_matches", len(linked), "PASS", "one-to-one linkage"),
    ("Sheet1_to_analysis_center_mismatches", 0, "PASS", "checked after linkage"),
    ("Sheet1_to_analysis_outcome_mismatches", 0, "PASS", "checked after linkage"),
    (
        "missing_AdmissionDate",
        int(linked["AdmissionDate"].isna().sum()),
        "PASS",
        "complete in Sheet1",
    ),
    (
        "missing_SurgeryDate",
        int(linked["SurgeryDate"].isna().sum()),
        "PASS",
        "mixed date formats harmonised deterministically",
    ),
    (
        "missing_DischargeDate",
        int(linked["DischargeDate"].isna().sum()),
        "LIMIT",
        "not suitable for full-cohort discharge-time analysis",
    ),
    (
        "admission_after_surgery",
        int((linked["AdmissionDate"] > linked["SurgeryDate"]).sum()),
        "DATA_QUALITY_FLAG",
        "retained in the primary cohort and excluded in a temporal sensitivity analysis",
    ),
    (
        "surgery_after_discharge",
        int((linked["SurgeryDate"] > linked["DischargeDate"]).sum()),
        "DATA_QUALITY_FLAG",
        "retained in the primary cohort and excluded in a temporal sensitivity analysis",
    ),
):
    date_integrity_rows.append(
        {"audit_item": item, "value": value, "status": status, "note": note}
    )
date_integrity = pd.DataFrame(date_integrity_rows)

centre_year = (
    linked.groupby([CENTER, "SurgeryYear"])[TARGET]
    .agg(n="size", events="sum", event_rate="mean")
    .reset_index()
)
intervals = [
    wilson_interval(int(row.events), int(row.n)) for row in centre_year.itertuples()
]
centre_year["event_rate_ci_lower"] = [x[0] for x in intervals]
centre_year["event_rate_ci_upper"] = [x[1] for x in intervals]

postop_days = pd.to_numeric(linked["PostopHospitalDays"], errors="coerce")
linked["inpatient_observation_group"] = np.select(
    [postop_days.ge(7), postop_days.lt(7)],
    [">=7 postoperative inpatient days", "<7 postoperative inpatient days"],
    default="missing",
)
observation_rows = []
for centre, group in linked.groupby(CENTER, sort=True):
    days = pd.to_numeric(group["PostopHospitalDays"], errors="coerce")
    for observation_group, subgroup in group.groupby(
        "inpatient_observation_group", sort=False
    ):
        n = len(subgroup)
        events = int(subgroup[TARGET].sum())
        lower, upper = wilson_interval(events, n)
        observation_rows.append(
            {
                "center": int(centre),
                "observation_group": observation_group,
                "n": int(n),
                "events": events,
                "event_rate": events / n if n else np.nan,
                "event_rate_ci_lower": lower,
                "event_rate_ci_upper": upper,
                "postop_days_median": float(days.median()) if days.notna().any() else np.nan,
                "postop_days_q1": float(days.quantile(0.25)) if days.notna().any() else np.nan,
                "postop_days_q3": float(days.quantile(0.75)) if days.notna().any() else np.nan,
                "interpretation": (
                    "inpatient opportunity proxy; not proof of complete 0-168 h ascertainment"
                ),
            }
        )
observation = pd.DataFrame(observation_rows)

available_exact = linked["SurgeryDate"].notna() & linked["DischargeDate"].notna()
linked["chronology_valid"] = (
    available_exact
    & linked["AdmissionDate"].le(linked["SurgeryDate"])
    & linked["SurgeryDate"].le(linked["DischargeDate"])
)
chronology_by_center = (
    linked.groupby(CENTER)
    .agg(
        n=("MajorID", "size"),
        events=(TARGET, "sum"),
        admission_after_surgery=(
            "AdmissionDate",
            lambda series: int(
                (
                    series
                    > linked.loc[series.index, "SurgeryDate"]
                ).sum()
            ),
        ),
        surgery_after_discharge=(
            "SurgeryDate",
            lambda series: int(
                (
                    series
                    > linked.loc[series.index, "DischargeDate"]
                ).sum()
            ),
        ),
        chronology_valid=("chronology_valid", "sum"),
    )
    .reset_index()
)
exact_postop_days = (
    linked.loc[linked["chronology_valid"], "DischargeDate"]
    - linked.loc[linked["chronology_valid"], "SurgeryDate"]
).dt.total_seconds() / 86400
recorded_postop_days = pd.to_numeric(
    linked.loc[linked["chronology_valid"], "PostopHospitalDays"], errors="coerce"
)
agreement_difference = exact_postop_days - recorded_postop_days

# %%
within_split, cutoffs = within_centre_chronological_split(
    linked, training_fraction=0.70
)
calendar_split = fixed_calendar_split(linked, cutoff="2022-01-01")
valid_chronology = linked.loc[linked["chronology_valid"]].copy().reset_index(drop=True)
valid_split, valid_cutoffs = within_centre_chronological_split(
    valid_chronology, training_fraction=0.70
)

split_flow_rows = []
for split_name, split in (
    ("within_centre_70_30", within_split),
    ("calendar_2022", calendar_split),
):
    for (centre, label), group in linked.assign(temporal_split=split).groupby(
        [CENTER, "temporal_split"]
    ):
        split_flow_rows.append(
            {
                "split_definition": split_name,
                "center": int(centre),
                "temporal_split": str(label),
                "n": int(len(group)),
                "events": int(group[TARGET].sum()),
                "start_date": iso_date(group["SurgeryDate"].min()),
                "end_date": iso_date(group["SurgeryDate"].max()),
            }
        )
split_flow = pd.DataFrame(split_flow_rows)
for (centre, label), group in valid_chronology.assign(
    temporal_split=valid_split
).groupby([CENTER, "temporal_split"]):
    split_flow_rows.append(
        {
            "split_definition": "within_centre_70_30_valid_chronology",
            "center": int(centre),
            "temporal_split": str(label),
            "n": int(len(group)),
            "events": int(group[TARGET].sum()),
            "start_date": iso_date(group["SurgeryDate"].min()),
            "end_date": iso_date(group["SurgeryDate"].max()),
        }
    )
split_flow = pd.DataFrame(split_flow_rows)

performance_frames = []
centre_frames = []
fit_frames = []
secure_frames = []
for split_name, split in (
    ("within_centre_70_30", within_split),
    ("calendar_2022", calendar_split),
):
    performance, centre_performance, fits, secure_predictions = fit_temporal_models(
        linked, split, split_name
    )
    performance_frames.append(performance)
    centre_frames.append(centre_performance)
    fit_frames.append(fits)
    secure_frames.append(secure_predictions)

performance_valid, centre_valid, fits_valid, secure_valid = fit_temporal_models(
    valid_chronology,
    valid_split,
    "within_centre_70_30_valid_chronology",
)
performance_frames.append(performance_valid)
centre_frames.append(centre_valid)
fit_frames.append(fits_valid)
secure_frames.append(secure_valid)

performance = pd.concat(performance_frames, ignore_index=True)
centre_performance = pd.concat(centre_frames, ignore_index=True)
fits = pd.concat(fit_frames, ignore_index=True)
secure_predictions = pd.concat(secure_frames, ignore_index=True)

# %%
TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
SECURE.mkdir(parents=True, exist_ok=True)
date_integrity.to_csv(TABLES / "Table_source_patient_date_integrity.csv", index=False)
centre_year.to_csv(TABLES / "Table_source_recruitment_by_center_year.csv", index=False)
observation.to_csv(
    TABLES / "Table_source_inpatient_observation_opportunity.csv", index=False
)
chronology_by_center.to_csv(
    TABLES / "Table_source_date_chronology_by_center.csv", index=False
)
split_flow.to_csv(TABLES / "Table_source_temporal_split_flow.csv", index=False)
all_cutoffs = pd.concat([cutoffs, valid_cutoffs], ignore_index=True)
all_cutoffs.loc[
    all_cutoffs.index >= len(cutoffs), "split_definition"
] = "within_centre_70_30_valid_chronology"
all_cutoffs.assign(
    cutoff_date=all_cutoffs["cutoff_date"].dt.strftime("%Y-%m-%d")
).to_csv(TABLES / "Table_source_within_center_temporal_cutoffs.csv", index=False)
performance.to_csv(TABLES / "Table_source_temporal_validation.csv", index=False)
centre_performance.to_csv(
    TABLES / "Table_source_temporal_validation_by_center.csv", index=False
)
fits.to_csv(TABLES / "Table_source_temporal_model_lock.csv", index=False)
secure_predictions.to_csv(
    SECURE / "SOURCE_TEMPORAL_VALIDATION_PREDICTIONS_SECURE.csv.gz",
    index=False,
    compression="gzip",
)

audit = {
    "analysis": (
        "source patient/date integrity, inpatient observation opportunity, "
        "and temporal validation"
    ),
    "source_workbook": str(SOURCE_WORKBOOK),
    "source_workbook_sha256": hashlib.sha256(SOURCE_WORKBOOK.read_bytes()).hexdigest(),
    "sheet": "Sheet1",
    "n_unique_patients": int(len(linked)),
    "n_unique_majorid": int(linked["MajorID"].nunique()),
    "events": int(linked[TARGET].sum()),
    "surgery_date_start": iso_date(linked["SurgeryDate"].min()),
    "surgery_date_end": iso_date(linked["SurgeryDate"].max()),
    "reported_2015_start_supported_by_sheet1": bool(
        linked["SurgeryDate"].min().year == 2015
    ),
    "date_missingness": {
        variable: int(linked[variable].isna().sum())
        for variable in ("AdmissionDate", "SurgeryDate", "DischargeDate")
    },
    "date_order_violations": {
        "admission_after_surgery": int(
            (linked["AdmissionDate"] > linked["SurgeryDate"]).sum()
        ),
        "surgery_after_discharge": int(
            (linked["SurgeryDate"] > linked["DischargeDate"]).sum()
        ),
    },
    "postoperative_inpatient_days": {
        "missing": int(postop_days.isna().sum()),
        "at_least_7": int(postop_days.ge(7).sum()),
        "less_than_7": int(postop_days.lt(7).sum()),
        "median": float(postop_days.median()),
        "q1": float(postop_days.quantile(0.25)),
        "q3": float(postop_days.quantile(0.75)),
        "exact_date_pairs": int(available_exact.sum()),
        "chronology_valid_date_pairs": int(linked["chronology_valid"].sum()),
        "median_exact_minus_recorded_days": float(agreement_difference.median()),
        "within_one_day": int(agreement_difference.abs().le(1).sum()),
    },
    "primary_temporal_split": (
        "earliest 70% within each centre by SurgeryDate; same-day records kept together"
    ),
    "secondary_temporal_split": (
        "calendar cutoff 2022-01-01; centre composition changes across the cutoff"
    ),
    "date_quality_sensitivity": (
        "within-centre 70/30 models refitted after excluding invalid admission-surgery-discharge chronology"
    ),
    "models": [
        "ridge",
        "restricted_rf",
        "gradient_boosting",
        "soft_voting_secondary",
    ],
    "feature_sets": ["P", "PI"],
    "prediction_landmark": "end of surgery",
    "date_variables_used_as_predictors": False,
    "postoperative_length_of_stay_used_as_predictor": False,
    "bootstrap": (
        "1000 patient resamples within validation centre, conditional on locked "
        "temporal predictions"
    ),
    "inference_boundaries": [
        "Temporal validation reuses participating centres and is not independent external validation.",
        "AdmissionDate is complete and DischargeDate is missing once, but 23 records have invalid date order.",
        "PostopHospitalDays is an inpatient observation-opportunity proxy, not proof of complete 0-168 h ascertainment.",
        "The fixed 2022 split mixes temporal change with changing centre composition.",
    ],
}
(OUTPUTS / "SOURCE_TEMPORAL_VALIDATION_AUDIT.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

display_columns = [
    "split_definition",
    "feature_set",
    "model",
    "n",
    "events",
    "roc_auc",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "oe_ratio",
    "oe_ratio_ci_lower",
    "oe_ratio_ci_upper",
    "calibration_slope",
]
print(json.dumps(audit, indent=2, ensure_ascii=False))
print(performance[display_columns].to_string(index=False))
