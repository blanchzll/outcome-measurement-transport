# %% [markdown]
# # Pure label-selection control experiment
#
# This experiment deletes otherwise perfectly classified patient-level labels.
# It is a positive control for inverse-probability weighting and is deliberately
# separate from longitudinal measurement coarsening and endpoint reconstruction.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


ROOT = Path(str(_release_path('analysis')))
TABLES, OUTPUTS, SECURE = ROOT / "tables", ROOT / "outputs", ROOT / "secure_work"
BASE_SEED = 20260826
RETENTIONS = (0.35, 0.55, 0.75)
MECHANISMS = ("MCAR", "risk_MAR", "stratum_MAR", "outcome_MNAR", "mixed_MNAR")
STRENGTHS = ("weak", "strong")

spec = importlib.util.spec_from_file_location("simulation", ROOT / "code" / "52_measurement_deletion_simulation.py")
simulation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(simulation)


def standardize(values):
    values = np.asarray(values, dtype=float)
    standard_deviation = np.nanstd(values)
    return np.zeros_like(values) if standard_deviation < 1e-8 else (values - np.nanmean(values)) / standard_deviation


def mean_calibrated_probability(score, target):
    lo, hi = -20.0, 20.0
    for _ in range(80):
        middle = (lo + hi) / 2
        if expit(middle + score).mean() < target:
            lo = middle
        else:
            hi = middle
    return expit((lo + hi) / 2 + score)


def run_database(database: str, replicates: int) -> None:
    preparers = {"INSPIRE": simulation.prepare_inspire, "MIMIC": simulation.prepare_mimic, "EICU": simulation.prepare_eicu}
    patient, _ = preparers[database]()
    patient = patient.reset_index(drop=True)
    truth = simulation.weighted_metrics(patient.y_full, patient.risk)
    risk_score = standardize(logit(patient.risk.clip(1e-6, 1 - 1e-6)))
    stratum_score = 0.6 * patient.age_z.to_numpy() + 0.4 * patient.sex_z.to_numpy() + 0.5 * patient.stratum_z.to_numpy()
    outcome_score = standardize(patient.y_full)
    rows = []
    for retention in RETENTIONS:
        for mechanism in MECHANISMS:
            for strength in STRENGTHS:
                scale = 0.65 if strength == "weak" else 1.35
                if mechanism == "MCAR":
                    score = np.zeros(len(patient))
                elif mechanism == "risk_MAR":
                    score = risk_score
                elif mechanism == "stratum_MAR":
                    score = stratum_score
                elif mechanism == "outcome_MNAR":
                    score = outcome_score
                else:
                    score = 0.4 * risk_score + 0.3 * stratum_score + 0.3 * outcome_score
                probability = mean_calibrated_probability(scale * score, retention)
                for replicate in range(replicates):
                    condition = f"selection|{database}|{retention}|{mechanism}|{strength}|{replicate}"
                    seed = BASE_SEED + int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16) % 2_000_000_000
                    rng = np.random.default_rng(seed)
                    observed = rng.random(len(patient)) < probability
                    if observed.sum() < 20 or patient.loc[observed, "y_full"].nunique() < 2:
                        continue
                    common = {
                        "database": database,
                        "retention_target": retention,
                        "mechanism": mechanism,
                        "strength": strength,
                        "replicate": replicate,
                        "seed": seed,
                        "observed_fraction": float(observed.mean()),
                    }
                    naive = simulation.add_event_rate_inference(
                        simulation.weighted_metrics(patient.loc[observed, "y_full"], patient.loc[observed, "risk"]),
                        truth["event_rate"],
                    )
                    rows.append({**common, "method": "naive", **naive})
                    weights = 1.0 / np.clip(probability[observed], 0.005, 1.0)
                    oracle = simulation.add_event_rate_inference(
                        simulation.weighted_metrics(patient.loc[observed, "y_full"], patient.loc[observed, "risk"], weights),
                        truth["event_rate"],
                    )
                    oracle.update({"weight_max": float(np.max(weights)), "weight_p99": float(np.quantile(weights, 0.99))})
                    rows.append({**common, "method": "oracle_IPW_untruncated", **oracle})

    raw = pd.DataFrame(rows)
    raw.to_csv(SECURE / f"{database}_PURE_LABEL_SELECTION_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    keys = ["database", "retention_target", "mechanism", "strength", "method"]
    summary_rows = []
    for values, group in raw.groupby(keys, dropna=False):
        base = dict(zip(keys, values, strict=True))
        for metric in ("event_rate", "oe", "brier", "auc", "ess", "event_rate_se", "weight_max", "weight_p99"):
            observed_values = group[metric].dropna().to_numpy(dtype=float)
            if not len(observed_values):
                continue
            target = truth.get(metric, np.nan)
            summary_rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_replicates": len(observed_values),
                    "mean": float(observed_values.mean()),
                    "sd": float(observed_values.std(ddof=1)),
                    "q025": float(np.quantile(observed_values, 0.025)),
                    "q975": float(np.quantile(observed_values, 0.975)),
                    "truth": target,
                    "bias": float(observed_values.mean() - target) if np.isfinite(target) else np.nan,
                    "rmse": float(np.sqrt(np.mean((observed_values - target) ** 2))) if np.isfinite(target) else np.nan,
                }
            )
        coverage = group.event_rate_coverage.dropna()
        if len(coverage):
            summary_rows.append({**base, "metric": "event_rate_interval_coverage", "n_replicates": len(coverage), "mean": float(coverage.mean())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / f"Table_{database.lower()}_pure_label_selection_control.csv", index=False)
    audit = {
        "database": database,
        "role": "positive-control experiment with perfectly classified observed labels",
        "replicates_per_condition": replicates,
        "conditions": len(RETENTIONS) * len(MECHANISMS) * len(STRENGTHS),
        "patient_n": len(patient),
        "events": int(patient.y_full.sum()),
        "oracle_probability_uses_unobserved_outcome_under_mnar": True,
        "interpretation": "oracle IPW is an identification benchmark, not an implementable MNAR estimator",
        "rows": len(raw),
    }
    (OUTPUTS / f"{database}_PURE_LABEL_SELECTION_CONTROL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=("INSPIRE", "MIMIC", "EICU", "all"), default="all")
    parser.add_argument("--reps", type=int, default=300)
    arguments = parser.parse_args()
    for selected_database in (("INSPIRE", "MIMIC", "EICU") if arguments.database == "all" else (arguments.database,)):
        run_database(selected_database, arguments.reps)
