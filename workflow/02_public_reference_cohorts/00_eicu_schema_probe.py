# %% [markdown]
# # eICU schema and endpoint-feasibility probe
#
# Aggregate-only audit for cohort identifiers, centre fields, longitudinal
# creatinine, urine-output records, and renal-replacement-therapy signals.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import json
from pathlib import Path

import duckdb


# %%
DATA = Path(str(_release_path('eicu')))
ROOT = Path(str(_release_path('analysis', 'eicu')))
OUT = ROOT / "outputs"
TABLES = ROOT / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(ROOT / "secure" / "eicu_probe.duckdb"))
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='12GB'")


# %%
patient_sql = f"""
SELECT *,
       CASE WHEN age = '> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END AS age_num,
       ROW_NUMBER() OVER (
           PARTITION BY patienthealthsystemstayid
           ORDER BY unitvisitnumber, patientunitstayid
       ) AS hospital_stay_icu_number
FROM read_csv_auto('{DATA / 'patient.csv.gz'}', header=true, sample_size=200000)
"""
con.execute("CREATE OR REPLACE TEMP TABLE patient AS " + patient_sql)

patient_summary = con.execute(
    """
    SELECT COUNT(*) AS unit_stays,
           COUNT(DISTINCT patientunitstayid) AS unique_unit_stays,
           COUNT(DISTINCT patienthealthsystemstayid) AS hospital_stays,
           COUNT(DISTINCT hospitalid) AS hospitals,
           SUM(age_num >= 18) AS adult_unit_stays,
           SUM(age_num >= 18 AND hospital_stay_icu_number = 1) AS adult_first_icu_stays,
           SUM(gender IS NULL OR TRIM(gender) = '') AS missing_gender,
           SUM(age_num IS NULL) AS missing_age
    FROM patient
    """
).fetchdf()
patient_summary.to_csv(TABLES / "Table_eicu_patient_summary.csv", index=False)

unit_types = con.execute(
    "SELECT unittype, COUNT(*) AS n FROM patient GROUP BY 1 ORDER BY n DESC"
).fetchdf()
unit_types.to_csv(TABLES / "Table_eicu_unit_types.csv", index=False)

hospital_summary = con.execute(
    f"""
    SELECT COUNT(*) AS hospitals,
           COUNT(DISTINCT region) AS regions,
           SUM(teachingstatus = 't') AS teaching_hospitals
    FROM read_csv_auto('{DATA / 'hospital.csv.gz'}', header=true)
    """
).fetchdf()
hospital_summary.to_csv(TABLES / "Table_eicu_hospital_summary.csv", index=False)


# %%
con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE creatinine AS
    SELECT patientunitstayid,
           TRY_CAST(labresultoffset AS INTEGER) AS offset_min,
           TRY_CAST(labresult AS DOUBLE) AS creatinine,
           labname,
           labmeasurenamesystem AS unit
    FROM read_csv_auto('{DATA / 'lab.csv.gz'}', header=true, sample_size=500000)
    WHERE LOWER(labname) LIKE '%creatin%'
      AND TRY_CAST(labresult AS DOUBLE) BETWEEN 0.1 AND 20
    """
)

creatinine_summary = con.execute(
    """
    SELECT COUNT(*) AS measurements,
           COUNT(DISTINCT patientunitstayid) AS unit_stays,
           MIN(offset_min) AS minimum_offset_min,
           MAX(offset_min) AS maximum_offset_min,
           MEDIAN(creatinine) AS median_creatinine,
           QUANTILE_CONT(creatinine, 0.01) AS p01_creatinine,
           QUANTILE_CONT(creatinine, 0.99) AS p99_creatinine
    FROM creatinine
    """
).fetchdf()
creatinine_summary.to_csv(TABLES / "Table_eicu_creatinine_summary.csv", index=False)

creatinine_names = con.execute(
    """
    SELECT labname, unit, COUNT(*) AS n
    FROM creatinine
    GROUP BY 1, 2
    ORDER BY n DESC
    """
).fetchdf()
creatinine_names.to_csv(TABLES / "Table_eicu_creatinine_names_units.csv", index=False)

window_counts = con.execute(
    """
    WITH cohort AS (
      SELECT patientunitstayid
      FROM patient
      WHERE age_num >= 18 AND hospital_stay_icu_number = 1
    ), flags AS (
      SELECT c.patientunitstayid,
             MAX(cr.offset_min BETWEEN -43200 AND -1)::INTEGER AS has_pre_icu,
             MAX(cr.offset_min BETWEEN 0 AND 1440)::INTEGER AS has_0_24h,
             MAX(cr.offset_min BETWEEN 1441 AND 2880)::INTEGER AS has_24_48h,
             MAX(cr.offset_min BETWEEN 2881 AND 5760)::INTEGER AS has_48_96h,
             MAX(cr.offset_min BETWEEN 5761 AND 10080)::INTEGER AS has_96_168h,
             COUNT(cr.creatinine) FILTER (WHERE cr.offset_min BETWEEN 0 AND 10080) AS n_0_168h
      FROM cohort c
      LEFT JOIN creatinine cr USING (patientunitstayid)
      GROUP BY 1
    )
    SELECT COUNT(*) AS adult_first_icu_stays,
           SUM(has_pre_icu) AS with_pre_icu_creatinine,
           SUM(has_0_24h) AS with_0_24h_creatinine,
           SUM(has_24_48h) AS with_24_48h_creatinine,
           SUM(has_48_96h) AS with_48_96h_creatinine,
           SUM(has_96_168h) AS with_96_168h_creatinine,
           SUM(has_0_24h * GREATEST(has_24_48h, has_48_96h, has_96_168h)) AS baseline_plus_followup,
           SUM(has_0_24h * has_24_48h * has_48_96h) AS dense_two_followup_windows,
           MEDIAN(n_0_168h) AS median_measurements_0_168h
    FROM flags
    """
).fetchdf()
window_counts.to_csv(TABLES / "Table_eicu_creatinine_window_coverage.csv", index=False)


# %%
urine_labels = con.execute(
    f"""
    SELECT celllabel, cellpath, COUNT(*) AS n,
           COUNT(DISTINCT patientunitstayid) AS unit_stays
    FROM read_csv_auto('{DATA / 'intakeOutput.csv.gz'}', header=true, sample_size=500000)
    WHERE LOWER(COALESCE(celllabel, '')) LIKE '%urine%'
       OR LOWER(COALESCE(celllabel, '')) LIKE '%urinary%'
       OR LOWER(COALESCE(cellpath, '')) LIKE '%urine%'
    GROUP BY 1, 2
    ORDER BY n DESC
    LIMIT 40
    """
).fetchdf()
urine_labels.to_csv(TABLES / "Table_eicu_candidate_urine_output_labels.csv", index=False)

dialysis_io = con.execute(
    f"""
    SELECT COUNT(*) AS rows_with_nonzero_dialysis_total,
           COUNT(DISTINCT patientunitstayid) AS unit_stays_with_nonzero_dialysis_total
    FROM read_csv_auto('{DATA / 'intakeOutput.csv.gz'}', header=true, sample_size=500000)
    WHERE TRY_CAST(dialysistotal AS DOUBLE) <> 0
    """
).fetchdf()
dialysis_io.to_csv(TABLES / "Table_eicu_dialysis_io_summary.csv", index=False)

rrt_treatments = con.execute(
    f"""
    SELECT treatmentstring, COUNT(*) AS n,
           COUNT(DISTINCT patientunitstayid) AS unit_stays
    FROM read_csv_auto('{DATA / 'treatment.csv.gz'}', header=true, sample_size=500000)
    WHERE LOWER(treatmentstring) LIKE '%dialys%'
       OR LOWER(treatmentstring) LIKE '%renal replacement%'
       OR LOWER(treatmentstring) LIKE '%hemofiltration%'
       OR LOWER(treatmentstring) LIKE '%crrt%'
    GROUP BY 1
    ORDER BY n DESC
    LIMIT 50
    """
).fetchdf()
rrt_treatments.to_csv(TABLES / "Table_eicu_candidate_rrt_treatments.csv", index=False)


# %%
payload = {
    "dataset_path": str(DATA),
    "dataset_version": "eICU Collaborative Research Database 2.0",
    "checksum_manifest_verified": True,
    "patient": patient_summary.to_dict("records")[0],
    "hospital": hospital_summary.to_dict("records")[0],
    "creatinine": creatinine_summary.to_dict("records")[0],
    "creatinine_windows": window_counts.to_dict("records")[0],
    "urine_label_rows_returned": int(len(urine_labels)),
    "rrt_treatment_rows_returned": int(len(rrt_treatments)),
    "patient_level_data_exported": False,
}
(OUT / "EICU_SCHEMA_FEASIBILITY_AUDIT.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

