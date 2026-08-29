# %% [markdown]
# # eICU 2.0 surgical-ICU ascertainment testbed
#
# This is a database-native computational replication. The landmark is ICU
# admission, not the end of surgery. The primary operational endpoint is a
# creatinine-only worsening endpoint harmonised to the MIMIC testbed.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# %%
DATA = Path(str(_release_path('eicu')))
PROJECT = Path(str(_release_path('analysis')))
ROOT = PROJECT / "eicu"
SECURE = ROOT / "secure"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
for directory in (SECURE, TABLES, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)

SURGICAL_UNITS = ("SICU", "CSICU", "CTICU", "CCU-CTICU")


def sha256(path: Path, chunk: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# %%
con = duckdb.connect(str(ROOT / "secure" / "eicu_build.duckdb"))
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='12GB'")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE patients AS
    SELECT *,
           CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age_num,
           ROW_NUMBER() OVER (
               PARTITION BY patienthealthsystemstayid
               ORDER BY unitvisitnumber, patientunitstayid
           ) AS first_icu_number
    FROM read_csv_auto('{DATA / 'patient.csv.gz'}', header=true, sample_size=200000)
    """
)

unit_list = ", ".join(f"'{unit}'" for unit in SURGICAL_UNITS)
con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE eligible AS
    SELECT p.*, h.numbedscategory, h.teachingstatus, h.region
    FROM patients p
    LEFT JOIN read_csv_auto('{DATA / 'hospital.csv.gz'}', header=true) h USING (hospitalid)
    WHERE p.first_icu_number = 1
      AND p.age_num >= 18
      AND p.unittype IN ({unit_list})
    """
)

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE creatinine AS
    SELECT patientunitstayid,
           TRY_CAST(labresultoffset AS INTEGER) AS offset_min,
           TRY_CAST(labresult AS DOUBLE) AS creatinine
    FROM read_csv_auto('{DATA / 'lab.csv.gz'}', header=true, sample_size=500000)
    WHERE LOWER(TRIM(labname)) = 'creatinine'
      AND LOWER(COALESCE(labmeasurenamesystem, 'mg/dl')) = 'mg/dl'
      AND TRY_CAST(labresult AS DOUBLE) BETWEEN 0.2 AND 20
      AND TRY_CAST(labresultoffset AS BIGINT) BETWEEN -43200 AND 10080
    """
)

con.execute(
    """
    CREATE OR REPLACE TEMP TABLE baseline AS
    SELECT e.patientunitstayid,
           ARG_MAX(c.creatinine, c.offset_min) AS baseline_creatinine,
           MAX(c.offset_min) AS baseline_offset_min
    FROM eligible e
    JOIN creatinine c USING (patientunitstayid)
    WHERE c.offset_min BETWEEN -43200 AND -1
    GROUP BY 1
    """
)

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE chronic_rrt AS
    WITH diagnosis_flag AS (
      SELECT patientunitstayid, 1 AS flag
      FROM read_csv_auto('{DATA / 'diagnosis.csv.gz'}', header=true, sample_size=500000)
      WHERE LOWER(COALESCE(diagnosisstring, '')) LIKE '%end stage renal%'
         OR LOWER(COALESCE(diagnosisstring, '')) LIKE '%chronic dialysis%'
         OR REGEXP_MATCHES(COALESCE(icd9code, ''), '(^|[, ]+)(585\\.6|N18\\.6)($|[, ])')
      GROUP BY 1
    ), history_flag AS (
      SELECT patientunitstayid, 1 AS flag
      FROM read_csv_auto('{DATA / 'pastHistory.csv.gz'}', header=true, sample_size=500000)
      WHERE LOWER(COALESCE(pasthistorypath, '')) LIKE '%dialysis%'
         OR LOWER(COALESCE(pasthistoryvaluetext, '')) LIKE '%dialysis%'
         OR LOWER(COALESCE(pasthistorypath, '')) LIKE '%end stage renal%'
      GROUP BY 1
    ), treatment_flag AS (
      SELECT patientunitstayid, 1 AS flag
      FROM read_csv_auto('{DATA / 'treatment.csv.gz'}', header=true, sample_size=500000)
      WHERE LOWER(COALESCE(treatmentstring, '')) LIKE '%for chronic renal failure%'
      GROUP BY 1
    )
    SELECT patientunitstayid, 1 AS chronic_rrt
    FROM (
      SELECT patientunitstayid FROM diagnosis_flag
      UNION
      SELECT patientunitstayid FROM history_flag
      UNION
      SELECT patientunitstayid FROM treatment_flag
    )
    """
)


# %%
cohort = con.execute(
    """
    SELECT e.*, b.baseline_creatinine, b.baseline_offset_min,
           COALESCE(r.chronic_rrt, 0) AS chronic_rrt
    FROM eligible e
    JOIN baseline b USING (patientunitstayid)
    LEFT JOIN chronic_rrt r USING (patientunitstayid)
    WHERE COALESCE(r.chronic_rrt, 0) = 0
    ORDER BY hospitaldischargeyear, hospitalid, patientunitstayid
    """
).fetchdf()
assert cohort.patientunitstayid.is_unique
cohort["reference_id"] = [f"EICU-{i:07d}" for i in range(1, len(cohort) + 1)]

serial = con.execute(
    """
    SELECT e.patientunitstayid, c.offset_min / 60.0 AS hour, c.creatinine
    FROM eligible e
    JOIN creatinine c USING (patientunitstayid)
    WHERE c.offset_min BETWEEN 0 AND 10080
    ORDER BY e.patientunitstayid, c.offset_min
    """
).fetchdf()
serial = serial.loc[serial.patientunitstayid.isin(cohort.patientunitstayid)].copy()
serial = serial.merge(
    cohort[["patientunitstayid", "reference_id", "baseline_creatinine"]],
    on="patientunitstayid",
    how="left",
    validate="many_to_one",
)

agg = serial.groupby("reference_id", observed=True).agg(
    n_creatinine_0_168h=("creatinine", "size"),
    first_hour=("hour", "min"),
    last_hour=("hour", "max"),
    max_creatinine_168h=("creatinine", "max"),
).reset_index()
max48 = serial.loc[serial.hour <= 48].groupby("reference_id").creatinine.max().rename("max_creatinine_48h")
n48 = serial.loc[serial.hour <= 48].groupby("reference_id").size().rename("n_creatinine_0_48h")
n96 = serial.loc[(serial.hour > 48) & (serial.hour <= 96)].groupby("reference_id").size().rename("n_creatinine_48_96h")
agg = agg.merge(max48, on="reference_id", how="left").merge(n48, on="reference_id", how="left").merge(n96, on="reference_id", how="left")
agg[["n_creatinine_0_48h", "n_creatinine_48_96h"]] = agg[["n_creatinine_0_48h", "n_creatinine_48_96h"]].fillna(0).astype(int)

reference = cohort.merge(agg, on="reference_id", how="left", validate="one_to_one")
reference["n_creatinine_0_168h"] = reference.n_creatinine_0_168h.fillna(0).astype(int)
reference["span_hours"] = reference.last_hour - reference.first_hour
reference["R_longitudinal"] = (reference.n_creatinine_0_168h >= 1).astype(int)
reference["R_two_slot"] = ((reference.n_creatinine_0_48h >= 1) & (reference.n_creatinine_48_96h >= 1)).astype(int)
reference["R_dense"] = (
    (reference.n_creatinine_0_168h >= 3)
    & (reference.R_two_slot == 1)
    & (reference.span_hours >= 72)
).astype(int)
event = (
    (reference.max_creatinine_48h >= reference.baseline_creatinine + 0.3)
    | (reference.max_creatinine_168h >= 1.5 * reference.baseline_creatinine)
)
reference["Y_longitudinal"] = np.where(reference.R_longitudinal.eq(1), event.astype(float), np.nan)
reference["Y_two_slot"] = np.where(reference.R_two_slot.eq(1), event.astype(float), np.nan)

reference.to_csv(SECURE / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", index=False, compression="gzip")
serial[["reference_id", "hour", "creatinine", "baseline_creatinine"]].to_csv(
    SECURE / "EICU_CREATININE_SERIAL_SECURE.csv.gz", index=False, compression="gzip"
)


# %%
flow = []
for stage, mask in {
    "adult first ICU stay in explicit surgical unit": np.ones(len(reference), dtype=bool),
    "valid pre-ICU baseline creatinine and no chronic-RRT flag": np.ones(len(reference), dtype=bool),
    "at least one 0-168h serum creatinine": reference.R_longitudinal.eq(1),
    "two-slot operational reference": reference.R_two_slot.eq(1),
    "dense longitudinal operational reference": reference.R_dense.eq(1),
}.items():
    outcome = reference.loc[mask, "Y_longitudinal"]
    flow.append(
        {
            "stage": stage,
            "n": int(np.asarray(mask).sum()),
            "events": int(outcome.fillna(0).sum()),
            "event_rate": float(outcome.mean()) if outcome.notna().any() else np.nan,
        }
    )
pd.DataFrame(flow).to_csv(TABLES / "Table_eicu_reference_flow.csv", index=False)

by_hospital = reference.groupby("hospitalid", dropna=False).agg(
    n=("reference_id", "size"),
    longitudinal_observed=("R_longitudinal", "mean"),
    two_slot_observed=("R_two_slot", "mean"),
    dense_observed=("R_dense", "mean"),
    events=("Y_longitudinal", "sum"),
    event_rate=("Y_longitudinal", "mean"),
    median_measurements=("n_creatinine_0_168h", "median"),
).reset_index()
by_hospital.to_csv(TABLES / "Table_eicu_observability_by_hospital.csv", index=False)

audit = {
    "database": "eICU Collaborative Research Database 2.0",
    "role": "database-native computational replication; not external validation of the surgery-end source model",
    "independent_unit": "first ICU unit stay within each deidentified hospital stay",
    "centres": int(reference.hospitalid.nunique()),
    "landmark": "ICU admission",
    "population": "adults in explicit surgical ICU types: SICU, CSICU, CTICU, or CCU-CTICU",
    "baseline": "last valid serum creatinine from -30 days to immediately before ICU admission",
    "operational_reference": "creatinine-only: >=0.3 mg/dL above baseline by 48h or >=1.5-fold above baseline by 168h",
    "exclusions": ["age <18 years", "not first ICU unit stay in the hospital stay", "no valid pre-ICU baseline creatinine", "flagged chronic dialysis or end-stage kidney disease"],
    "limitations": ["not chart-adjudicated", "ICU admission is not surgery end", "patient identity across separate hospital stays is unavailable", "creatinine-only primary target"],
    "n_candidate_after_baseline_and_chronic_rrt_exclusion": int(len(reference)),
    "n_longitudinal": int(reference.R_longitudinal.sum()),
    "events_longitudinal": int(reference.Y_longitudinal.sum(skipna=True)),
    "n_two_slot": int(reference.R_two_slot.sum()),
    "events_two_slot": int(reference.loc[reference.R_two_slot.eq(1), "Y_two_slot"].sum()),
    "n_dense": int(reference.R_dense.sum()),
    "events_dense": int(reference.loc[reference.R_dense.eq(1), "Y_longitudinal"].sum()),
    "source_integrity": {
        "patient_sha256": sha256(DATA / "patient.csv.gz"),
        "lab_sha256": sha256(DATA / "lab.csv.gz"),
        "checksum_manifest_sha256": sha256(DATA / "SHA256SUMS.txt"),
    },
    "patient_level_data_exported_outside_secure_directory": False,
}
(OUTPUTS / "EICU_REFERENCE_AUDIT.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2, ensure_ascii=False))

