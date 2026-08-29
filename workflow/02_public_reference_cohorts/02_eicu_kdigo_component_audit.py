# %% [markdown]
# # eICU KDIGO component-availability audit
#
# Creatinine remains the harmonised primary public-data endpoint. This script
# adds a conservative algorithmic urine-output and RRT sensitivity endpoint and
# quantifies how endpoint composition changes database-native model evaluation.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


# %%
DATA = Path(str(_release_path('eicu')))
ROOT = Path(str(_release_path('analysis')))
EICU = ROOT / "eicu"
SECURE, TABLES, OUTPUTS = EICU / "secure", ROOT / "tables", ROOT / "outputs"
sys.path.insert(0, str(ROOT / "code"))

spec = importlib.util.spec_from_file_location("simulation_core", ROOT / "code" / "52_measurement_deletion_simulation.py")
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)

reference = pd.read_csv(SECURE / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)

con = duckdb.connect(str(SECURE / "eicu_component_audit.duckdb"))
con.execute("PRAGMA threads=8")
con.execute("PRAGMA memory_limit='12GB'")
con.register("reference_frame", reference[["patientunitstayid", "reference_id", "admissionweight"]])


# %%
urine_bins = con.execute(
    f"""
    WITH urine AS (
      SELECT r.reference_id,
             TRY_CAST(io.intakeoutputoffset AS INTEGER) AS offset_min,
             TRY_CAST(io.cellvaluenumeric AS DOUBLE) AS volume_ml,
             TRY_CAST(r.admissionweight AS DOUBLE) AS weight_kg
      FROM read_csv_auto('{DATA / 'intakeOutput.csv.gz'}', header=true, sample_size=500000) io
      JOIN reference_frame r USING (patientunitstayid)
      WHERE TRY_CAST(io.intakeoutputoffset AS BIGINT) BETWEEN 0 AND 10080
        AND TRY_CAST(io.cellvaluenumeric AS DOUBLE) BETWEEN 0 AND 5000
        AND LOWER(COALESCE(io.cellpath, '')) LIKE '%i&o%output (ml)%'
        AND (LOWER(COALESCE(io.celllabel, '')) LIKE '%urine%'
             OR LOWER(COALESCE(io.celllabel, '')) LIKE '%urinary catheter%')
        AND LOWER(COALESCE(io.celllabel, '')) NOT LIKE '%count%'
        AND LOWER(COALESCE(io.celllabel, '')) NOT LIKE '%occurrence%'
        AND LOWER(COALESCE(io.celllabel, '')) NOT LIKE '%mixed urine/stool%'
    )
    SELECT reference_id,
           FLOOR(offset_min / 360) AS six_hour_bin,
           COUNT(*) AS records_in_bin,
           SUM(volume_ml) AS urine_ml,
           MAX(weight_kg) AS weight_kg,
           SUM(volume_ml) / NULLIF(MAX(weight_kg) * 6.0, 0) AS ml_kg_hour
    FROM urine
    WHERE weight_kg BETWEEN 20 AND 350
    GROUP BY 1, 2
    """
).fetchdf()

urine_patient = urine_bins.groupby("reference_id", observed=True).agg(
    urine_bins_recorded=("six_hour_bin", "nunique"),
    interpretable_six_hour_bins=("records_in_bin", lambda x: int((x >= 3).sum())),
    minimum_ml_kg_hour=("ml_kg_hour", "min"),
).reset_index()
eligible_bins = urine_bins.loc[urine_bins.records_in_bin >= 3]
urine_event = eligible_bins.groupby("reference_id").ml_kg_hour.min().lt(0.5).rename("urine_output_aki")
urine_patient = urine_patient.merge(urine_event, on="reference_id", how="left")
urine_patient["urine_output_observable"] = urine_patient.interpretable_six_hour_bins.gt(0).astype(int)
urine_patient["urine_output_aki"] = urine_patient.urine_output_aki.fillna(False).astype(int)


# %%
rrt_treatment = con.execute(
    f"""
    SELECT DISTINCT r.reference_id
    FROM read_csv_auto('{DATA / 'treatment.csv.gz'}', header=true, sample_size=500000) t
    JOIN reference_frame r USING (patientunitstayid)
    WHERE TRY_CAST(t.treatmentoffset AS BIGINT) BETWEEN 0 AND 10080
      AND LOWER(COALESCE(t.treatmentstring, '')) NOT LIKE '%for chronic renal failure%'
      AND (
           LOWER(COALESCE(t.treatmentstring, '')) LIKE '%hemodialysis%'
        OR LOWER(COALESCE(t.treatmentstring, '')) LIKE '%renal replacement%'
        OR LOWER(COALESCE(t.treatmentstring, '')) LIKE '%hemofiltration%'
        OR LOWER(COALESCE(t.treatmentstring, '')) LIKE '%c v v h%'
        OR LOWER(COALESCE(t.treatmentstring, '')) LIKE '%crrt%'
      )
    """
).fetchdf()

rrt_io = con.execute(
    f"""
    SELECT DISTINCT r.reference_id
    FROM read_csv_auto('{DATA / 'intakeOutput.csv.gz'}', header=true, sample_size=500000) io
    JOIN reference_frame r USING (patientunitstayid)
    WHERE TRY_CAST(io.intakeoutputoffset AS BIGINT) BETWEEN 0 AND 10080
      AND ABS(TRY_CAST(io.dialysistotal AS DOUBLE)) > 0
    """
).fetchdf()
rrt_ids = set(rrt_treatment.reference_id) | set(rrt_io.reference_id)


# %%
component = reference[[
    "reference_id", "hospitalid", "R_longitudinal", "R_dense", "Y_longitudinal",
    "n_creatinine_0_168h", "admissionweight",
]].copy()
component = component.merge(urine_patient, on="reference_id", how="left")
component[["urine_bins_recorded", "interpretable_six_hour_bins", "urine_output_observable", "urine_output_aki"]] = (
    component[["urine_bins_recorded", "interpretable_six_hour_bins", "urine_output_observable", "urine_output_aki"]]
    .fillna(0)
)
component["rrt_0_168h"] = component.reference_id.isin(rrt_ids).astype(int)
component["creatinine_aki"] = component.Y_longitudinal.fillna(0).astype(int)
component["algorithmic_multicomponent_aki"] = (
    component[["creatinine_aki", "urine_output_aki", "rrt_0_168h"]].max(axis=1).astype(int)
)
component.to_csv(SECURE / "EICU_MULTICOMPONENT_KDIGO_SECURE.csv.gz", index=False, compression="gzip")

rows = []
for label, mask in {
    "all operational-reference candidates": np.ones(len(component), dtype=bool),
    "dense creatinine reference": component.R_dense.eq(1),
    "at least one interpretable urine-output bin": component.urine_output_observable.eq(1),
    "dense creatinine plus interpretable urine output": component.R_dense.eq(1) & component.urine_output_observable.eq(1),
}.items():
    subset = component.loc[mask]
    rows.append({
        "population": label,
        "n": len(subset),
        "creatinine_events": int(subset.creatinine_aki.sum()),
        "urine_output_events": int(subset.urine_output_aki.sum()),
        "rrt_events": int(subset.rrt_0_168h.sum()),
        "multicomponent_events": int(subset.algorithmic_multicomponent_aki.sum()),
        "urine_only_events": int(((subset.urine_output_aki == 1) & (subset.creatinine_aki == 0) & (subset.rrt_0_168h == 0)).sum()),
        "rrt_only_events": int(((subset.rrt_0_168h == 1) & (subset.creatinine_aki == 0) & (subset.urine_output_aki == 0)).sum()),
    })
component_table = pd.DataFrame(rows)
component_table.to_csv(TABLES / "Table_eicu_kdigo_component_availability.csv", index=False)

by_hospital = component.groupby("hospitalid", observed=True).agg(
    n=("reference_id", "size"),
    creatinine_observed=("R_longitudinal", "mean"),
    dense_creatinine=("R_dense", "mean"),
    urine_output_observed=("urine_output_observable", "mean"),
    creatinine_event_rate=("creatinine_aki", "mean"),
    urine_output_event_rate=("urine_output_aki", "mean"),
    multicomponent_event_rate=("algorithmic_multicomponent_aki", "mean"),
).reset_index()
by_hospital.to_csv(TABLES / "Table_eicu_component_observability_by_hospital.csv", index=False)


# %%
patient, _ = core.prepare_eicu()
test = patient.merge(
    component[["reference_id", "creatinine_aki", "algorithmic_multicomponent_aki", "urine_output_observable"]],
    on="reference_id",
    how="left",
    validate="one_to_one",
)
metrics = {
    "creatinine_only": core.weighted_metrics(test.creatinine_aki, test.risk),
    "available_component_union": core.weighted_metrics(test.algorithmic_multicomponent_aki, test.risk),
}

audit = {
    "database": "eICU Collaborative Research Database 2.0",
    "primary_public_endpoint": "creatinine-only operational reference",
    "sensitivity_endpoint": "available-component algorithmic union of creatinine, conservative fixed-bin urine output, and RRT",
    "urine_rule": "any fixed 6-hour bin with at least 3 recorded urine-volume entries and total <0.5 mL/kg/h",
    "urine_rule_status": "conservative algorithmic proxy; not full duration-certified KDIGO adjudication",
    "rrt_rule": "non-chronic dialysis/RRT treatment string or nonzero dialysis total within 0-168h",
    "reference_candidates": int(len(component)),
    "urine_output_observable": int(component.urine_output_observable.sum()),
    "rrt_cases": int(component.rrt_0_168h.sum()),
    "test_set_metrics": metrics,
    "interpretation": "component sensitivity analysis; not replacement of the harmonised creatinine target",
    "patient_level_data_exported_outside_secure_directory": False,
}
(OUTPUTS / "EICU_KDIGO_COMPONENT_AUDIT.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2, ensure_ascii=False))
