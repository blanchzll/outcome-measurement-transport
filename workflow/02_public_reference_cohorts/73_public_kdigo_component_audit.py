# %% [markdown]
# # Public-database KDIGO component and endpoint-compatibility audit
#
# This analysis quantifies whether creatinine, urine-output, and renal replacement
# therapy components can be operationalised in INSPIRE and MIMIC-IV. It never calls
# an algorithmic EHR endpoint clinician-adjudicated or a biological gold standard.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import importlib.util
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(str(_release_path('analysis')))
INSPIRE = Path(str(_release_path('inspire')))
MIMIC_DB = Path(str(_release_path('mimic_duckdb')))
MIMIC_PROCEDURES = Path(str(_release_path('mimic', 'icu/procedureevents.csv.gz')))
MIMIC_ITEMS = Path(str(_release_path('mimic', 'icu/d_items.csv.gz')))
BUILDER = Path(str(_release_path('source', 'public_validation_20260824/code/02_build_inspire_external_cohort.py')))
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
RRT_ITEM_IDS = (225441, 225802, 225803, 225805, 225809, 225955)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def inspire_component_audit() -> dict:
    builder = load_module("inspire_builder", BUILDER)
    operations = builder.read_operations(INSPIRE)
    diagnoses = builder.read_diagnoses(INSPIRE)
    cohort, _ = builder.select_index_operations(operations, diagnoses)
    cohort = cohort.sort_values(["subject_id", "hadm_id", "opend_time", "op_id"]).reset_index(drop=True)
    cohort["reference_id"] = np.arange(len(cohort), dtype=int)
    subjects = set(cohort.subject_id.astype("int64"))
    pieces = []
    for chunk in pd.read_csv(
        INSPIRE / "ward_vitals.csv.gz",
        usecols=["subject_id", "chart_time", "item_name", "value"],
        chunksize=1_000_000,
    ):
        selected = chunk.loc[chunk.subject_id.isin(subjects) & chunk.item_name.isin(["uo", "crrt"])].copy()
        if not selected.empty:
            pieces.append(selected)
    vitals = pd.concat(pieces, ignore_index=True).drop_duplicates()
    vitals["chart_time"] = pd.to_numeric(vitals.chart_time, errors="coerce")
    vitals["value"] = pd.to_numeric(vitals.value, errors="coerce")
    vitals = vitals.merge(cohort[["reference_id", "subject_id", "opend_time"]], on="subject_id", how="inner")
    vitals["hour"] = (vitals.chart_time - vitals.opend_time) / 60.0
    vitals = vitals.loc[vitals.hour.gt(0) & vitals.hour.le(168)].copy()

    urine = vitals.loc[vitals.item_name.eq("uo") & vitals.value.ge(0)].copy()
    urine_summary = urine.groupby("reference_id").agg(
        urine_records=("chart_time", "nunique"),
        urine_first_hour=("hour", "min"),
        urine_last_hour=("hour", "max"),
    )
    urine_summary["urine_span_hours"] = urine_summary.urine_last_hour - urine_summary.urine_first_hour
    urine_summary["minimum_6h_sequence_proxy"] = (
        urine_summary.urine_records.ge(6) & urine_summary.urine_span_hours.ge(6)
    )
    rrt_ids = set(vitals.loc[vitals.item_name.eq("crrt") & vitals.value.gt(0), "reference_id"])

    analysis = pd.read_csv(SECURE / "INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz", low_memory=False)
    component = analysis[["reference_id", "dense_reference", "full168_creatinine_aki"]].merge(
        urine_summary.reset_index(), on="reference_id", how="left", validate="one_to_one"
    )
    component["urine_records"] = component.urine_records.fillna(0).astype(int)
    component["rrt_0_168h"] = component.reference_id.isin(rrt_ids).astype(int)
    component["creatinine_or_rrt_aki"] = np.where(
        component.full168_creatinine_aki.notna(),
        component.full168_creatinine_aki.astype("Int64").fillna(0).astype(int) | component.rrt_0_168h,
        np.nan,
    )
    component.to_csv(SECURE / "INSPIRE_KDIGO_COMPONENT_AUDIT_SECURE.csv.gz", index=False, compression="gzip")

    rows = []
    for label, mask in {
        "candidate": np.ones(len(component), dtype=bool),
        "dense_creatinine_reference": component.dense_reference.eq(1),
    }.items():
        subset = component.loc[mask]
        rows.append(
            {
                "database": "INSPIRE",
                "population": label,
                "n": len(subset),
                "creatinine_events": int(subset.full168_creatinine_aki.fillna(0).sum()),
                "any_urine_output_n": int(subset.urine_records.gt(0).sum()),
                "minimum_6h_sequence_proxy_n": int(subset.minimum_6h_sequence_proxy.fillna(False).sum()),
                "rrt_n": int(subset.rrt_0_168h.sum()),
                "creatinine_or_rrt_events": int(pd.Series(subset.creatinine_or_rrt_aki).fillna(0).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "Table_inspire_kdigo_component_availability.csv", index=False)
    return {
        "candidate_n": len(component),
        "any_urine_output_n": int(component.urine_records.gt(0).sum()),
        "minimum_6h_sequence_proxy_n": int(component.minimum_6h_sequence_proxy.fillna(False).sum()),
        "rrt_n": int(component.rrt_0_168h.sum()),
        "full_urine_output_kdigo_estimable": False,
        "reason": "post-landmark urine-output records were too sparse for continuous KDIGO duration criteria",
    }


def mimic_component_audit() -> dict:
    reference = pd.read_csv(SECURE / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False, parse_dates=["intime"])
    connection = duckdb.connect(str(MIMIC_DB), read_only=True)
    connection.register("reference", reference[["stay_id", "reference_id", "intime"]])
    urine_sql = """
        SELECT r.reference_id,
               count(u.charttime) AS urine_rows,
               max(CASE WHEN u.uo_tm_6hr >= 6 AND u.uo_mlkghr_6hr < 0.5 THEN 1 ELSE 0 END) AS uo_stage1,
               max(CASE WHEN u.uo_tm_12hr >= 12 AND u.uo_mlkghr_12hr < 0.5 THEN 1 ELSE 0 END) AS uo_stage2,
               max(CASE WHEN (u.uo_tm_24hr >= 24 AND u.uo_mlkghr_24hr < 0.3)
                         OR (u.uo_tm_12hr >= 12 AND u.urineoutput_12hr = 0) THEN 1 ELSE 0 END) AS uo_stage3
        FROM reference r
        LEFT JOIN mimiciv_derived.urine_output_rate u USING (stay_id)
        WHERE u.charttime >= r.intime AND u.charttime <= r.intime + INTERVAL 168 HOUR
        GROUP BY r.reference_id
    """
    urine = connection.execute(urine_sql).fetchdf()
    item_ids = ",".join(str(value) for value in RRT_ITEM_IDS)
    rrt_sql = f"""
        SELECT DISTINCT r.reference_id
        FROM reference r
        JOIN read_csv_auto('{MIMIC_PROCEDURES.as_posix()}', header=true) p USING (stay_id)
        WHERE p.itemid IN ({item_ids})
          AND p.starttime >= r.intime
          AND p.starttime <= r.intime + INTERVAL 168 HOUR
          AND coalesce(p.statusdescription, '') != 'Rewritten'
    """
    rrt = set(connection.execute(rrt_sql).fetchdf().reference_id)
    connection.close()

    component = reference.merge(urine, on="reference_id", how="left", validate="one_to_one")
    for column in ("urine_rows", "uo_stage1", "uo_stage2", "uo_stage3"):
        component[column] = component[column].fillna(0).astype(int)
    component["rrt_0_168h"] = component.reference_id.isin(rrt).astype(int)
    creatinine_ratio = component.max_creatinine_168h / component.baseline_creatinine
    component["creatinine_stage"] = np.select(
        [
            component.rrt_0_168h.eq(1) | creatinine_ratio.ge(3)
            | (component.max_creatinine_168h.ge(4) & component.Y_longitudinal.fillna(0).eq(1)),
            creatinine_ratio.ge(2),
            component.Y_longitudinal.fillna(0).eq(1),
        ],
        [3, 2, 1],
        default=0,
    )
    component["urine_stage"] = np.select(
        [component.uo_stage3.eq(1), component.uo_stage2.eq(1), component.uo_stage1.eq(1)],
        [3, 2, 1],
        default=0,
    )
    component["multicomponent_stage"] = component[["creatinine_stage", "urine_stage"]].max(axis=1)
    component["multicomponent_aki"] = component.multicomponent_stage.gt(0).astype(int)
    component.to_csv(SECURE / "MIMIC_MULTICOMPONENT_KDIGO_SECURE.csv.gz", index=False, compression="gzip")

    rows = []
    for label, mask in {
        "candidate": np.ones(len(component), dtype=bool),
        "dense_creatinine_reference": component.R_dense.eq(1),
    }.items():
        subset = component.loc[mask]
        rows.append(
            {
                "database": "MIMIC-IV",
                "population": label,
                "n": len(subset),
                "creatinine_events": int(subset.Y_longitudinal.fillna(0).sum()),
                "urine_observed_n": int(subset.urine_rows.gt(0).sum()),
                "urine_stage_1plus_n": int(subset.urine_stage.gt(0).sum()),
                "rrt_n": int(subset.rrt_0_168h.sum()),
                "multicomponent_events": int(subset.multicomponent_aki.sum()),
                "urine_or_rrt_only_events": int((subset.multicomponent_aki.eq(1) & subset.Y_longitudinal.fillna(0).eq(0)).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "Table_mimic_kdigo_component_availability.csv", index=False)

    dense = component.loc[component.R_dense.eq(1) & component.Y_longitudinal.notna()].copy()
    concordance = pd.crosstab(
        dense.Y_longitudinal.astype(int), dense.multicomponent_aki.astype(int),
        rownames=["creatinine_only_aki"], colnames=["multicomponent_aki"],
    ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    concordance.stack(future_stack=True).rename("n").reset_index().to_csv(
        TABLES / "Table_mimic_creatinine_vs_multicomponent_concordance.csv", index=False
    )

    simulation = load_module("simulation", ROOT / "code" / "52_measurement_deletion_simulation.py")
    temporal_patient, _ = simulation.prepare_mimic()
    temporal = temporal_patient.merge(
        component[["reference_id", "multicomponent_aki", "creatinine_stage", "urine_stage", "rrt_0_168h"]],
        on="reference_id", how="left", validate="one_to_one",
    )
    metric_rows = []
    for endpoint in ("y_full", "multicomponent_aki"):
        metrics = simulation.weighted_metrics(temporal[endpoint], temporal.risk)
        metric_rows.extend({"endpoint": endpoint, "metric": key, "value": value} for key, value in metrics.items())
    pd.DataFrame(metric_rows).to_csv(TABLES / "Table_mimic_endpoint_target_performance.csv", index=False)
    return {
        "candidate_n": len(component),
        "dense_n": int(component.R_dense.sum()),
        "dense_creatinine_events": int(dense.Y_longitudinal.sum()),
        "dense_multicomponent_events": int(dense.multicomponent_aki.sum()),
        "dense_urine_or_rrt_only_events": int((dense.multicomponent_aki.eq(1) & dense.Y_longitudinal.eq(0)).sum()),
        "rrt_item_ids": list(RRT_ITEM_IDS),
        "endpoint_is_clinician_adjudicated": False,
        "endpoint_role": "algorithmic multicomponent KDIGO sensitivity endpoint",
    }


if __name__ == "__main__":
    inspire = inspire_component_audit()
    mimic = mimic_component_audit()
    audit = {
        "inspire": inspire,
        "mimic": mimic,
        "source_hashes": {
            "inspire_ward_vitals_sha256": sha256(INSPIRE / "ward_vitals.csv.gz"),
            "mimic_urine_database_size": MIMIC_DB.stat().st_size,
            "mimic_procedureevents_sha256": sha256(MIMIC_PROCEDURES),
            "mimic_d_items_sha256": sha256(MIMIC_ITEMS),
        },
        "claim_boundary": "public operational endpoints do not reproduce site-level nephrologist adjudication",
    }
    (OUTPUTS / "PUBLIC_KDIGO_COMPONENT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
