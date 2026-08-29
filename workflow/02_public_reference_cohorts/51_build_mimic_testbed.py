# %% [markdown]
# # MIMIC-IV 3.1 surgical-ICU ascertainment testbed
# Methodological replication only. The landmark is first ICU admission, not surgery end.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(str(_release_path('analysis')))
DB = Path(str(_release_path('mimic_duckdb')))
SERVICES = Path(str(_release_path('mimic', 'hosp/services.csv.gz')))
CHANGELOG = Path(str(_release_path('mimic', 'CHANGELOG.txt')))
SECURE = ROOT / "secure_work"
TABLES = ROOT / "tables"
OUTPUTS = ROOT / "outputs"
for directory in (SECURE, TABLES, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)

SURGICAL_SERVICES = (
    "SURG", "NSURG", "CSURG", "VSURG", "TSURG", "ORTHO", "ENT",
    "GU", "GYN", "PSURG", "TRAUM",
)


def sha256(path: Path, chunk: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            h.update(block)
    return h.hexdigest()


# %%
assert DB.exists() and SERVICES.exists(), "MIMIC database or services file is unavailable"
con = duckdb.connect(str(DB), read_only=True)
service_list = ", ".join(f"'{x}'" for x in SURGICAL_SERVICES)

cohort_sql = f"""
WITH first_icu AS (
  SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY intime, stay_id) AS rn
  FROM mimiciv_icu.icustays
),
services AS (
  SELECT hadm_id, transfertime, curr_service
  FROM read_csv_auto('{SERVICES.as_posix()}', header=true)
),
eligible AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, i.first_careunit,
         a.admittime, a.dischtime, a.race, a.admission_type,
         ag.age, p.gender,
         (SELECT arg_max(s.curr_service, s.transfertime)
          FROM services s
          WHERE s.hadm_id=i.hadm_id AND s.transfertime <= i.intime) AS active_service
  FROM first_icu i
  JOIN mimiciv_hosp.admissions a USING (subject_id, hadm_id)
  JOIN mimiciv_derived.age ag USING (subject_id, hadm_id)
  JOIN mimiciv_hosp.patients p USING (subject_id)
  WHERE i.rn=1 AND ag.age >= 18
),
base_chem AS (
  SELECT e.stay_id,
         arg_max(c.creatinine, c.charttime) FILTER (WHERE c.creatinine IS NOT NULL) AS baseline_creatinine,
         arg_max(c.albumin, c.charttime) FILTER (WHERE c.albumin IS NOT NULL) AS baseline_albumin,
         arg_max(c.bun, c.charttime) FILTER (WHERE c.bun IS NOT NULL) AS baseline_bun,
         arg_max(c.glucose, c.charttime) FILTER (WHERE c.glucose IS NOT NULL) AS baseline_glucose,
         arg_max(c.sodium, c.charttime) FILTER (WHERE c.sodium IS NOT NULL) AS baseline_sodium,
         arg_max(c.potassium, c.charttime) FILTER (WHERE c.potassium IS NOT NULL) AS baseline_potassium
  FROM eligible e
  LEFT JOIN mimiciv_derived.chemistry c
    ON c.hadm_id=e.hadm_id AND c.charttime >= e.intime - INTERVAL 30 DAY AND c.charttime < e.intime
  GROUP BY e.stay_id
),
base_cbc AS (
  SELECT e.stay_id,
         arg_max(c.hemoglobin, c.charttime) FILTER (WHERE c.hemoglobin IS NOT NULL) AS baseline_hemoglobin,
         arg_max(c.wbc, c.charttime) FILTER (WHERE c.wbc IS NOT NULL) AS baseline_wbc,
         arg_max(c.platelet, c.charttime) FILTER (WHERE c.platelet IS NOT NULL) AS baseline_platelet
  FROM eligible e
  LEFT JOIN mimiciv_derived.complete_blood_count c
    ON c.hadm_id=e.hadm_id AND c.charttime >= e.intime - INTERVAL 30 DAY AND c.charttime < e.intime
  GROUP BY e.stay_id
),
dx AS (
  SELECT e.stay_id,
         max(CASE WHEN (d.icd_version=9 AND d.icd_code LIKE '250%') OR
                            (d.icd_version=10 AND regexp_matches(d.icd_code, '^E(08|09|10|11|12|13)'))
                  THEN 1 ELSE 0 END) AS diabetes
  FROM eligible e
  LEFT JOIN mimiciv_hosp.diagnoses_icd d USING (subject_id, hadm_id)
  GROUP BY e.stay_id
)
SELECT e.*, b.*, c.* EXCLUDE(stay_id), d.diabetes
FROM eligible e
JOIN base_chem b USING (stay_id)
JOIN base_cbc c USING (stay_id)
JOIN dx d USING (stay_id)
WHERE e.active_service IN ({service_list})
  AND b.baseline_creatinine BETWEEN 0.2 AND 20
ORDER BY e.intime, e.stay_id
"""

cohort = con.execute(cohort_sql).fetchdf()
assert cohort.stay_id.is_unique
cohort["reference_id"] = [f"MIMIC-{i:07d}" for i in range(1, len(cohort) + 1)]

serial_sql = f"""
WITH first_icu AS (
  SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY intime, stay_id) AS rn
  FROM mimiciv_icu.icustays
),
services AS (
  SELECT hadm_id, transfertime, curr_service
  FROM read_csv_auto('{SERVICES.as_posix()}', header=true)
),
eligible AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime,
         ag.age,
         (SELECT arg_max(s.curr_service, s.transfertime)
          FROM services s
          WHERE s.hadm_id=i.hadm_id AND s.transfertime <= i.intime) AS active_service
  FROM first_icu i JOIN mimiciv_derived.age ag USING(subject_id, hadm_id)
  WHERE i.rn=1 AND ag.age >= 18
)
SELECT e.stay_id, c.charttime,
       date_diff('minute', e.intime, c.charttime)/60.0 AS hour,
       c.creatinine
FROM eligible e
JOIN mimiciv_derived.chemistry c USING(subject_id, hadm_id)
WHERE e.active_service IN ({service_list})
  AND c.creatinine BETWEEN 0.2 AND 20
  AND c.charttime >= e.intime AND c.charttime <= e.intime + INTERVAL 168 HOUR
ORDER BY e.stay_id, c.charttime
"""
serial = con.execute(serial_sql).fetchdf()
serial = serial[serial.stay_id.isin(cohort.stay_id)].copy()
serial = serial.merge(cohort[["stay_id", "reference_id", "baseline_creatinine"]], on="stay_id", how="left", validate="many_to_one")

# %%
agg = serial.groupby("reference_id", observed=True).agg(
    n_postop_creatinine=("creatinine", "size"),
    first_hour=("hour", "min"),
    last_hour=("hour", "max"),
    max_creatinine_168h=("creatinine", "max"),
).reset_index()
max48 = serial.loc[serial.hour <= 48].groupby("reference_id").creatinine.max().rename("max_creatinine_48h")
n48 = serial.loc[serial.hour <= 48].groupby("reference_id").size().rename("n_creatinine_0_48h")
n96 = serial.loc[(serial.hour > 48) & (serial.hour <= 96)].groupby("reference_id").size().rename("n_creatinine_48_96h")
agg = agg.merge(max48, on="reference_id", how="left").merge(n48, on="reference_id", how="left").merge(n96, on="reference_id", how="left")
agg[["n_creatinine_0_48h", "n_creatinine_48_96h"]] = agg[["n_creatinine_0_48h", "n_creatinine_48_96h"]].fillna(0).astype(int)
ref = cohort.merge(agg, on="reference_id", how="left", validate="one_to_one")
ref["n_postop_creatinine"] = ref.n_postop_creatinine.fillna(0).astype(int)
ref["span_hours"] = ref.last_hour - ref.first_hour
ref["R_longitudinal"] = (ref.n_postop_creatinine >= 1).astype(int)
ref["R_two_slot"] = ((ref.n_creatinine_0_48h >= 1) & (ref.n_creatinine_48_96h >= 1)).astype(int)
ref["R_dense"] = ((ref.n_postop_creatinine >= 3) & (ref.R_two_slot == 1) & (ref.span_hours >= 72)).astype(int)
ref["Y_longitudinal"] = np.where(
    ref.R_longitudinal == 1,
    ((ref.max_creatinine_48h >= ref.baseline_creatinine + 0.3) |
     (ref.max_creatinine_168h >= 1.5 * ref.baseline_creatinine)).astype(float),
    np.nan,
)
ref["Y_two_slot"] = np.where(
    ref.R_two_slot == 1,
    ((ref.max_creatinine_48h >= ref.baseline_creatinine + 0.3) |
     (ref.max_creatinine_168h >= 1.5 * ref.baseline_creatinine)).astype(float),
    np.nan,
)
ref["calendar_year"] = pd.to_datetime(ref.intime).dt.year

keep_serial = serial[["reference_id", "hour", "creatinine", "baseline_creatinine"]]
ref.to_csv(SECURE / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", index=False, compression="gzip")
keep_serial.to_csv(SECURE / "MIMIC_CREATININE_SERIAL_SECURE.csv.gz", index=False, compression="gzip")

flow = []
for name, mask in {
    "adult first ICU stay, surgical service, baseline creatinine": np.ones(len(ref), dtype=bool),
    "at least one 0-168h creatinine": ref.R_longitudinal == 1,
    "two-slot operational reference": ref.R_two_slot == 1,
    "dense longitudinal reference": ref.R_dense == 1,
}.items():
    y = ref.loc[mask, "Y_longitudinal"]
    flow.append({"stage": name, "n": int(mask.sum()), "events": int(y.fillna(0).sum()), "event_rate": float(y.mean())})
pd.DataFrame(flow).to_csv(TABLES / "Table_mimic_reference_flow.csv", index=False)

by_service = ref.groupby("active_service", dropna=False).agg(
    n=("reference_id", "size"), longitudinal_observed=("R_longitudinal", "mean"),
    two_slot_observed=("R_two_slot", "mean"), dense_observed=("R_dense", "mean"),
    events=("Y_longitudinal", "sum"), event_rate=("Y_longitudinal", "mean"),
).reset_index()
by_service.to_csv(TABLES / "Table_mimic_observability_by_service.csv", index=False)

audit = {
    "database": "MIMIC-IV 3.1",
    "role": "methodological replication testbed; not clinical external validation of the surgery-end model",
    "landmark": "first ICU admission during an admission whose active service is surgical",
    "operational_reference": "creatinine-only: >=0.3 mg/dL by 48h or >=1.5-fold by 168h",
    "limitations": ["no urine-output criterion", "not chart-adjudicated", "ICU admission is not surgery end"],
    "n_candidate": int(len(ref)),
    "n_longitudinal": int(ref.R_longitudinal.sum()),
    "events_longitudinal": int(ref.Y_longitudinal.sum(skipna=True)),
    "n_two_slot": int(ref.R_two_slot.sum()),
    "events_two_slot": int(ref.loc[ref.R_two_slot.eq(1), "Y_two_slot"].sum()),
    "n_dense": int(ref.R_dense.sum()),
    "events_dense": int(ref.loc[ref.R_dense.eq(1), "Y_longitudinal"].sum()),
    "source_integrity": {
        "services_sha256": sha256(SERVICES),
        "changelog_sha256": sha256(CHANGELOG) if CHANGELOG.exists() else None,
        "duckdb_size_bytes": DB.stat().st_size,
    },
}
(OUTPUTS / "MIMIC_REFERENCE_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
