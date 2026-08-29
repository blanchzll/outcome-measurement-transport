# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pre-landmark eICU predictors shared with MIMIC-IV
#
# All laboratory predictors use the latest valid value from 30 days before ICU
# admission to immediately before the landmark. Predictor extraction is outcome
# blind and mirrors the MIMIC-IV baseline window.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(str(_release_path('analysis')))
DATA = Path(str(_release_path('eicu')))
SECURE = ROOT / "eicu" / "secure"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
for directory in (SECURE, TABLES, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)

LABS = {
    "albumin": ("baseline_albumin", "g/dL", 1.0, 6.0),
    "bun": ("baseline_bun", "mg/dL", 2.0, 200.0),
    "glucose": ("baseline_glucose", "mg/dL", 20.0, 1000.0),
    "sodium": ("baseline_sodium", "mmol/L", 100.0, 180.0),
    "potassium": ("baseline_potassium", "mmol/L", 2.0, 8.0),
    "hgb": ("baseline_hemoglobin", "g/dL", 3.0, 25.0),
    "wbc x 1000": ("baseline_wbc", "K/uL", 0.1, 100.0),
    "platelets x 1000": ("baseline_platelet", "K/uL", 5.0, 2000.0),
}


def sha256(path: Path, chunk: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# %%
reference_path = SECURE / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz"
reference = pd.read_csv(reference_path, low_memory=False)
assert reference["reference_id"].is_unique

con = duckdb.connect(str(SECURE / "eicu_common_predictors.duckdb"))
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='12GB'")
con.register("eligible_ids", reference[["patientunitstayid", "reference_id"]])

name_list = ",".join(f"'{name}'" for name in LABS)
case_expr = "\n".join(
    f"WHEN LOWER(TRIM(labname))='{raw}' AND TRY_CAST(labresult AS DOUBLE) BETWEEN {lo} AND {hi} "
    f"THEN TRY_CAST(labresult AS DOUBLE)"
    for raw, (_, _, lo, hi) in LABS.items()
)
con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE shared_labs AS
    SELECT l.patientunitstayid,
           TRY_CAST(l.labresultoffset AS INTEGER) AS offset_min,
           LOWER(TRIM(l.labname)) AS lab_name,
           CASE {case_expr} ELSE NULL END AS valid_value
    FROM read_csv_auto('{DATA / 'lab.csv.gz'}', header=true, sample_size=500000) l
    JOIN eligible_ids e USING (patientunitstayid)
    WHERE LOWER(TRIM(l.labname)) IN ({name_list})
      AND TRY_CAST(l.labresultoffset AS BIGINT) BETWEEN -43200 AND -1
    """
)

selects = []
for raw, (column, _, _, _) in LABS.items():
    selects.append(
        f"ARG_MAX(valid_value, offset_min) FILTER (WHERE lab_name='{raw}' AND valid_value IS NOT NULL) AS {column}"
    )
query = f"""
SELECT e.reference_id, e.patientunitstayid,
       {', '.join(selects)}
FROM eligible_ids e
LEFT JOIN shared_labs s USING (patientunitstayid)
GROUP BY 1,2
ORDER BY 2
"""
predictors = con.execute(query).fetchdf()
assert len(predictors) == len(reference)
assert predictors["reference_id"].is_unique

out_path = SECURE / "EICU_COMMON_PREDICTORS_SECURE.csv.gz"
predictors.to_csv(out_path, index=False, compression="gzip")

# %%
joined = reference[["reference_id", "R_dense"]].merge(predictors, on="reference_id", validate="one_to_one")
rows = []
for cohort_label, subset in (("candidate", joined), ("dense_reference", joined.loc[joined.R_dense.eq(1)])):
    for raw, (column, unit, lo, hi) in LABS.items():
        rows.append(
            {
                "cohort": cohort_label,
                "predictor": column,
                "eicu_source_labname": raw,
                "harmonized_unit": unit,
                "valid_range": f"{lo:g}-{hi:g}",
                "n": len(subset),
                "n_observed": int(subset[column].notna().sum()),
                "missing_fraction": float(subset[column].isna().mean()),
                "selection_rule": "latest valid value from -30 days to <0 h relative to ICU admission",
            }
        )
missingness = pd.DataFrame(rows)
missingness.to_csv(TABLES / "Table_eicu_common_predictor_availability.csv", index=False)

audit = {
    "analysis": "outcome-blind extraction of eICU predictors shared with MIMIC-IV",
    "landmark": "ICU admission",
    "lookback_window": "-30 days to immediately before ICU admission",
    "selection": "latest valid value within predictor-specific prespecified range",
    "predictors": {column: {"source_labname": raw, "unit": unit, "valid_range": [lo, hi]} for raw, (column, unit, lo, hi) in LABS.items()},
    "n_candidate": int(len(reference)),
    "n_dense": int(reference.R_dense.sum()),
    "patient_level_output": str(out_path.relative_to(ROOT)),
    "patient_level_output_delivered": False,
    "input_sha256": {
        "eicu_lab_csv_gz": sha256(DATA / "lab.csv.gz"),
        "eicu_reference_csv_gz": sha256(reference_path),
    },
}
(OUTPUTS / "EICU_COMMON_PREDICTOR_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(missingness.loc[missingness.cohort.eq("dense_reference")].to_string(index=False))

