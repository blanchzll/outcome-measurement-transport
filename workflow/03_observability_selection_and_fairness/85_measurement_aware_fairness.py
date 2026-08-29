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
# # Measurement-aware subgroup audit
#
# This analysis separates disparity in dense endpoint observability from disparity
# in prediction performance. It is a representativeness and measurement audit,
# not a fairness certification or a causal analysis.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(str(_release_path('analysis')))
sys.path.insert(0, str(ROOT / "code"))
from ascertainment_stress import delete_and_reconstruct, weighted_metrics  # noqa: E402

SEED = 20260827
N_BOOTSTRAP = 500
N_SIMULATION = 100
RETENTION = 0.55
MECHANISM = "mixed_MNAR"
STRENGTH = "strong"
MIN_N, MIN_EVENTS, MIN_NONEVENTS = 100, 20, 20


def age_group(values):
    return pd.cut(pd.to_numeric(values, errors="coerce"), [-np.inf, 64, 74, np.inf], labels=["<65", "65-74", "75+"]).astype("string").fillna("Unknown")


def mimic_race(values):
    text = values.astype("string").str.upper().fillna("UNKNOWN")
    out = pd.Series("Other or unknown", index=text.index, dtype="string")
    out[text.str.contains("WHITE", na=False)] = "White"
    out[text.str.contains("BLACK", na=False)] = "Black"
    out[text.str.contains("ASIAN", na=False)] = "Asian"
    out[text.str.contains("HISPANIC|LATINO", regex=True, na=False)] = "Hispanic or Latino"
    return out


def load_observability():
    frames = []
    inspire = pd.read_csv(ROOT / "secure_work" / "INSPIRE_OBSERVABILITY_ANALYSIS_SECURE.csv.gz", low_memory=False)
    frames.append(pd.DataFrame({
        "database": "INSPIRE", "record_id": inspire.reference_id.astype("string"), "cluster": inspire.reference_id.astype("string"),
        "dense": pd.to_numeric(inspire.dense_reference, errors="coerce").fillna(0).astype(int),
        "full_observed": pd.to_numeric(inspire.full168_observed, errors="coerce").fillna(0).astype(int),
        "full_outcome": pd.to_numeric(inspire.full168_creatinine_aki, errors="coerce").where(inspire.full168_observed.eq(1)),
        "sex": inspire.Gender.astype("string").str.title().fillna("Unknown"), "age": age_group(inspire.Age),
        "clinical_group": inspire.cancer_site_label.astype("string").fillna("Unknown"),
        "clinical_group_name": "cancer_site",
    }))
    mimic = pd.read_csv(ROOT / "secure_work" / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
    frames.append(pd.DataFrame({
        "database": "MIMIC-IV", "record_id": mimic.reference_id.astype("string"), "cluster": mimic.subject_id.astype("string"),
        "dense": mimic.R_dense.astype(int), "full_observed": mimic.R_longitudinal.astype(int),
        "full_outcome": pd.to_numeric(mimic.Y_longitudinal, errors="coerce"),
        "sex": mimic.gender.astype("string").str.upper().map({"M": "Male", "F": "Female"}).fillna("Unknown"),
        "age": age_group(mimic.age), "clinical_group": mimic_race(mimic.race), "clinical_group_name": "race_or_ethnicity",
    }))
    eicu = pd.read_csv(ROOT / "eicu" / "secure" / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
    frames.append(pd.DataFrame({
        "database": "eICU", "record_id": eicu.reference_id.astype("string"), "cluster": eicu.hospitalid.astype("string"),
        "dense": eicu.R_dense.astype(int), "full_observed": eicu.R_longitudinal.astype(int),
        "full_outcome": pd.to_numeric(eicu.Y_longitudinal, errors="coerce"),
        "sex": eicu.gender.astype("string").str.title().fillna("Unknown"), "age": age_group(eicu.age_num),
        "clinical_group": eicu.ethnicity.astype("string").str.strip().str.title().fillna("Unknown"),
        "clinical_group_name": "race_or_ethnicity",
    }))
    return frames


def bootstrap_group_rates(frame, variable, cluster_bootstrap, seed):
    usable = frame.loc[frame[variable].notna()].copy()
    counts = usable.groupby(variable, observed=False).size().sort_values(ascending=False)
    groups = counts.index[counts >= MIN_N].astype(str).tolist()
    if not groups:
        return []
    reference = groups[0]
    rng = np.random.default_rng(seed)
    estimates = {group: {"rate": [], "rr": [], "difference": []} for group in groups}
    if cluster_bootstrap:
        unique = usable.cluster.unique()
        cluster_index = {label: usable.index[usable.cluster.eq(label)].to_numpy() for label in unique}
    for _ in range(N_BOOTSTRAP):
        if cluster_bootstrap:
            drawn = rng.choice(unique, len(unique), replace=True)
            pieces = []
            for draw_number, label in enumerate(drawn):
                piece = usable.loc[cluster_index[label]].copy()
                piece["_draw_cluster"] = draw_number
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
        else:
            sample = usable.iloc[rng.choice(len(usable), len(usable), replace=True)]
        rates = sample.groupby(variable, observed=False).dense.mean()
        ref_rate = float(rates.get(reference, np.nan))
        for group in groups:
            rate = float(rates.get(group, np.nan))
            if np.isfinite(rate) and np.isfinite(ref_rate) and ref_rate > 0:
                estimates[group]["rate"].append(rate)
                estimates[group]["rr"].append(rate / ref_rate)
                estimates[group]["difference"].append(rate - ref_rate)
    rows = []
    for group in groups:
        sub = usable.loc[usable[variable].astype(str).eq(group)]
        rate = float(sub.dense.mean())
        ref_rate = float(usable.loc[usable[variable].astype(str).eq(reference), "dense"].mean())
        row = {
            "database": str(usable.database.iloc[0]), "group_variable": variable, "group": group,
            "reference_group": reference, "n": len(sub), "n_dense": int(sub.dense.sum()),
            "dense_observability": rate, "dense_observability_ci_lower": float(np.quantile(estimates[group]["rate"], .025)),
            "dense_observability_ci_upper": float(np.quantile(estimates[group]["rate"], .975)),
            "risk_ratio_vs_reference": rate / ref_rate,
            "risk_ratio_ci_lower": float(np.quantile(estimates[group]["rr"], .025)),
            "risk_ratio_ci_upper": float(np.quantile(estimates[group]["rr"], .975)),
            "risk_difference_vs_reference": rate - ref_rate,
            "risk_difference_ci_lower": float(np.quantile(estimates[group]["difference"], .025)),
            "risk_difference_ci_upper": float(np.quantile(estimates[group]["difference"], .975)),
            "bootstrap_unit": "hospital" if cluster_bootstrap else "analytic_record",
            "n_bootstrap": N_BOOTSTRAP,
        }
        rows.append(row)
    return rows


def z(values):
    values = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmedian(values))
    sd = values.std()
    return np.zeros(len(values)) if sd < 1e-10 else (values - values.mean()) / sd


def load_stress_database(database):
    predictions = pd.read_csv(ROOT / "secure_work" / "PUBLIC_EXTENDED_TRANSPORT_PREDICTIONS_SECURE.csv.gz")
    predictions = predictions.loc[(predictions.database.eq(database)) & predictions.model_specification.eq("extended_common")].copy()
    if database == "MIMIC-IV":
        d = pd.read_csv(ROOT / "secure_work" / "MIMIC_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
        d["sex_group"] = d.gender.astype("string").str.upper().map({"M": "Male", "F": "Female"}).fillna("Unknown")
        d["age_group"] = age_group(d.age)
        d["ethnicity_group"] = mimic_race(d.race)
        d["age_numeric"] = d.age
        d["sex_numeric"] = d.gender.astype("string").str.upper().map({"M": 1, "F": 0})
        d["stratum_numeric"] = pd.factorize(d.race)[0]
        serial = pd.read_csv(ROOT / "secure_work" / "MIMIC_CREATININE_SERIAL_SECURE.csv.gz")
    else:
        d = pd.read_csv(ROOT / "eicu" / "secure" / "EICU_SURGICAL_ICU_REFERENCE_SECURE.csv.gz", low_memory=False)
        d["sex_group"] = d.gender.astype("string").str.title().fillna("Unknown")
        d["age_group"] = age_group(d.age_num)
        d["ethnicity_group"] = d.ethnicity.astype("string").str.strip().str.title().fillna("Unknown")
        d["age_numeric"] = d.age_num
        d["sex_numeric"] = d.gender.astype("string").str.lower().map({"male": 1, "female": 0})
        d["stratum_numeric"] = pd.factorize(d.ethnicity)[0]
        serial = pd.read_csv(ROOT / "eicu" / "secure" / "EICU_CREATININE_SERIAL_SECURE.csv.gz")
    d = d.merge(predictions[["record_id", "predicted_probability"]], left_on="reference_id", right_on="record_id", validate="one_to_one")
    patient = pd.DataFrame({
        "reference_id": d.reference_id, "baseline_creatinine": d.baseline_creatinine,
        "y_full": d.Y_longitudinal.astype(int), "risk": d.predicted_probability,
        "age_z": z(d.age_numeric), "sex_z": z(d.sex_numeric), "stratum_z": z(d.stratum_numeric),
        "sex": d.sex_group, "age": d.age_group, "race_or_ethnicity": d.ethnicity_group,
    })
    serial = serial.loc[serial.reference_id.isin(patient.reference_id), ["reference_id", "hour", "creatinine"]]
    return patient.reset_index(drop=True), serial.reset_index(drop=True)


def one_measurement_replicate(database, patient, serial, replicate):
    condition = f"fairness|{database}|{replicate}|{SEED}"
    seed = SEED + int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16) % 2_000_000_000
    rng = np.random.default_rng(seed)
    sim = delete_and_reconstruct(patient, serial, MECHANISM, RETENTION, STRENGTH, rng)
    f = sim.patient
    rows = []
    for variable in ("sex", "age", "race_or_ethnicity"):
        for group, sub in f.groupby(variable, observed=False):
            if len(sub) < MIN_N or sub.y_full.sum() < MIN_EVENTS or (len(sub) - sub.y_full.sum()) < MIN_NONEVENTS:
                continue
            observed = sub.R.eq(1) & sub.y_reconstructed.notna()
            if observed.sum() < MIN_N or sub.loc[observed, "y_reconstructed"].sum() < MIN_EVENTS:
                continue
            full = weighted_metrics(sub.y_full, sub.risk)
            apparent = weighted_metrics(sub.loc[observed, "y_reconstructed"], sub.loc[observed, "risk"])
            sensitivity = float((sub.y_reconstructed.eq(1) & sub.y_full.eq(1)).sum() / max(sub.y_full.sum(), 1))
            for metric in ("auc", "oe", "brier", "calibration_intercept", "calibration_slope"):
                rows.append({
                    "database": database, "group_variable": variable, "group": str(group),
                    "replicate": replicate, "seed": seed, "n": len(sub), "events": int(sub.y_full.sum()),
                    "metric": metric, "full_reference": full[metric], "apparent_reconstructed": apparent[metric],
                    "measurement_induced_gap": apparent[metric] - full[metric],
                    "outcome_observed_fraction": float(observed.mean()), "reconstructed_sensitivity": sensitivity,
                })
    return rows


# %%
observability_rows = []
for frame in load_observability():
    database = str(frame.database.iloc[0])
    variables = ["sex", "age", "clinical_group"]
    for index, variable in enumerate(variables):
        rows = bootstrap_group_rates(frame, variable, database == "eICU", SEED + len(database) * 10 + index)
        for row in rows:
            if variable == "clinical_group":
                row["group_variable"] = str(frame.clinical_group_name.iloc[0])
        observability_rows.extend(rows)
pd.DataFrame(observability_rows).to_csv(ROOT / "tables" / "Table_measurement_aware_fairness_observability.csv", index=False)

gap_rows = []
for database in ("MIMIC-IV", "eICU"):
    patient, serial = load_stress_database(database)
    nested = Parallel(n_jobs=8, prefer="processes", batch_size=1)(
        delayed(one_measurement_replicate)(database, patient, serial, replicate)
        for replicate in range(N_SIMULATION)
    )
    gap_rows.extend(row for group in nested for row in group)

raw = pd.DataFrame(gap_rows)
raw.to_csv(ROOT / "secure_work" / "MEASUREMENT_AWARE_FAIRNESS_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
summary_rows = []
keys = ["database", "group_variable", "group", "metric"]
for key, group in raw.groupby(keys, dropna=False):
    base = dict(zip(keys, key))
    first = group.iloc[0]
    row = {**base, "n": int(first.n), "events": int(first.events), "n_replicates": len(group), "independent_unit": "Monte Carlo replicate"}
    for variable in ("full_reference", "apparent_reconstructed", "measurement_induced_gap", "outcome_observed_fraction", "reconstructed_sensitivity"):
        values = pd.to_numeric(group[variable], errors="coerce").dropna().to_numpy()
        row[f"{variable}_mean"] = float(values.mean())
        row[f"{variable}_q025"] = float(np.quantile(values, .025))
        row[f"{variable}_q975"] = float(np.quantile(values, .975))
    summary_rows.append(row)
summary = pd.DataFrame(summary_rows)
summary.to_csv(ROOT / "tables" / "Table_measurement_aware_fairness_calibration_gap.csv", index=False)

disparity_rows = []
for key, group in summary.groupby(["database", "group_variable", "metric"], dropna=False):
    if len(group) < 2:
        continue
    disparity_rows.append({
        "database": key[0], "group_variable": key[1], "metric": key[2],
        "groups_included": "|".join(group.group.astype(str)),
        "full_reference_max_minus_min": float(group.full_reference_mean.max() - group.full_reference_mean.min()),
        "apparent_max_minus_min": float(group.apparent_reconstructed_mean.max() - group.apparent_reconstructed_mean.min()),
        "measurement_gap_max_minus_min": float(group.measurement_induced_gap_mean.max() - group.measurement_induced_gap_mean.min()),
        "interpretation": "descriptive disparity decomposition; not a fairness hypothesis test",
    })
pd.DataFrame(disparity_rows).to_csv(ROOT / "tables" / "Table_measurement_aware_fairness_disparity_summary.csv", index=False)

audit = {
    "analysis": "measurement-aware subgroup audit",
    "observability_bootstrap": N_BOOTSTRAP,
    "observability_bootstrap_unit": {"INSPIRE": "analytic record", "MIMIC-IV": "analytic record", "eICU": "hospital"},
    "measurement_simulation": {"replicates": N_SIMULATION, "mechanism": MECHANISM, "strength": STRENGTH, "target_retention": RETENTION},
    "minimum_reportable_group": f"n>={MIN_N}, events>={MIN_EVENTS}, non-events>={MIN_NONEVENTS}",
    "multiplicity": "no subgroup p values; descriptive effect sizes and intervals only",
    "interpretation_boundary": "The audit separates observability and apparent-performance disparities but is not a causal fairness analysis or certification.",
    "protected_data": "patient-level and replicate-level outputs retained under secure_work only",
}
(ROOT / "outputs" / "MEASUREMENT_AWARE_FAIRNESS_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
print(pd.DataFrame(observability_rows).groupby("database").size())
print(summary.loc[summary.metric.eq("oe"), ["database", "group_variable", "group", "n", "events", "full_reference_mean", "apparent_reconstructed_mean", "measurement_induced_gap_mean"]].to_string(index=False))
