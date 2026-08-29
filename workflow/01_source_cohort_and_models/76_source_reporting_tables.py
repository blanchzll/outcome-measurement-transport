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
# # Auditable source-cohort reporting tables
# Aggregate-only outputs: centre characteristics and frozen predictor definitions.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(str(_release_path('source')))
ROOT = BASE / "ascertainment_framework_20260826"
sys.path.insert(0, str(BASE))

from analysis import CENTER, TARGET, load_cohort  # noqa: E402
from loco_analysis import FEATURE_SET_SPECS, engineer_loco_features  # noqa: E402

DATA = BASE / "secure_source" / "inter3_deidentified_4014.csv"
TABLES, OUTPUTS = ROOT / "tables", ROOT / "outputs"


def median_iqr(series: pd.Series) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna()
    return f"{x.median():.1f} ({x.quantile(.25):.1f}-{x.quantile(.75):.1f})"


def n_percent(mask: pd.Series) -> str:
    boolean = mask.astype("boolean")
    observed = boolean.notna()
    numerator = int(boolean.sum(skipna=True))
    denominator = int(observed.sum())
    return f"{numerator} ({100 * numerator / denominator:.1f}%)" if denominator else "NA"


raw = load_cohort(DATA)
data = engineer_loco_features(raw)
columns: dict[str, pd.Series] = {"Overall": pd.Series(True, index=data.index)}
for centre in sorted(data[CENTER].unique()):
    columns[f"Centre {int(centre)}"] = data[CENTER].eq(centre)

rows: list[dict[str, object]] = []
for label, mask in columns.items():
    d = data.loc[mask]
    values = {
        "Analytic records, n": str(len(d)),
        "Adjudicated postoperative AKI, n (%)": n_percent(d[TARGET].eq(1)),
        "Age, years, median (IQR)": median_iqr(d["Age"]),
        "Male, n (%) among known": n_percent(d["Gender"].eq("Male").where(d["Gender"].notna())),
        "Female, n (%) among known": n_percent(d["Gender"].eq("Female").where(d["Gender"].notna())),
        "Sex unknown or nonconforming, n (%)": f"{d['Gender'].isna().sum()} ({100*d['Gender'].isna().mean():.1f}%)",
        "Gastric cancer, n (%)": n_percent(d["Gastrocolorectal"].eq("1").where(d["Gastrocolorectal"].notna())),
        "Colorectal cancer, n (%)": n_percent(d["Gastrocolorectal"].eq("2").where(d["Gastrocolorectal"].notna())),
        "Preoperative creatinine, umol/L, median (IQR)": median_iqr(d["PreopCr"]),
        "Preoperative haemoglobin, g/L, median (IQR)": median_iqr(d["PreopHb"]),
        "Preoperative albumin, g/L, median (IQR)": median_iqr(d["PreopAlb"]),
        "Intraoperative transfusion, mL, median (IQR)": median_iqr(d["IntraopTransfusion"]),
    }
    for variable, value in values.items():
        rows.append({"characteristic": variable, "stratum": label, "value": value})

pd.DataFrame(rows).to_csv(TABLES / "Table_source_characteristics_by_center.csv", index=False)

definition = {
    "Age": ("Age at surgery", "years", "continuous"),
    "Gender": ("Recorded sex", "1 or Male=male; 0, 2, or Female=female; garbled/nonconforming token=missing", "categorical"),
    "Diabetes": ("Preoperative diabetes", "0=no; 1=yes", "categorical"),
    "Hypertension": ("Preoperative hypertension", "0=no; 1=yes", "categorical"),
    "CardiovascularDisease": ("Heart disease or cerebrovascular disease", "derived binary indicator", "categorical"),
    "Gastrocolorectal": ("Cancer operation site", "1=gastric; 2=colorectal", "categorical"),
    "NeoadjuvantChemo": ("Neoadjuvant chemotherapy", "0=no; 1=yes", "categorical"),
    "PreopCr": ("Most recent available preoperative serum creatinine", "umol/L", "continuous"),
    "LogPreopCr": ("Natural logarithm of preoperative creatinine", "log(umol/L), row-wise transform", "continuous"),
    "PreopHb": ("Preoperative haemoglobin", "g/L", "continuous"),
    "PreopAlb": ("Preoperative albumin", "g/L", "continuous"),
    "IntraopTransfusion": ("Total intraoperative transfusion volume", "mL", "continuous"),
    "SurgicalApproach": ("Surgical approach", "1=open; 2=laparoscopic; 3=converted; 4=robotic", "categorical"),
    "CombinedOrganResection": ("Combined organ resection", "0=no; 1=yes", "categorical"),
}

predictors = sorted({v for spec in FEATURE_SET_SPECS.values() if spec.name in {"P", "PI", "H"} for v in spec.features})
definition_rows = []
for variable in predictors:
    description, unit_or_codes, kind = definition[variable]
    source_variable = "PreopCr" if variable == "LogPreopCr" else variable
    missing_n = int(data[source_variable].isna().sum()) if source_variable in data else 0
    roles = [name for name in ("P", "PI", "H") if variable in FEATURE_SET_SPECS[name].features]
    definition_rows.append({
        "predictor": variable,
        "definition": description,
        "unit_or_codes": unit_or_codes,
        "type": kind,
        "feature_sets": "|".join(roles),
        "available_by_prediction_landmark": "yes; end of surgery",
        "source_missing_n": missing_n,
        "missing_data_handling": "training-fold-only imputation; continuous median plus indicator, categorical most frequent",
    })

pd.DataFrame(definition_rows).to_csv(TABLES / "Table_source_predictor_definitions_units.csv", index=False)

audit = {
    "source_rows": int(len(data)),
    "source_events": int(data[TARGET].sum()),
    "centers": int(data[CENTER].nunique()),
    "unique_analytic_record_ids": int(raw["MajorID"].nunique()),
    "repeat_patient_status": "not verifiable because a separate stable patient identifier was not retained",
    "study_dates_status": "not available in the deidentified analytic extract",
    "sex_known_male": int(data["Gender"].eq("Male").sum()),
    "sex_known_female": int(data["Gender"].eq("Female").sum()),
    "sex_unknown_or_nonconforming": int(data["Gender"].isna().sum()),
    "garbled_token_policy": "the token 濂 is treated as missing, never as female",
    "outputs_are_aggregate_only": True,
}
(OUTPUTS / "SOURCE_REPORTING_TABLE_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(audit, indent=2, ensure_ascii=False))
