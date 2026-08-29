# %% [markdown]
# # Parallel production runner for the ascertainment stress test
# Parallelises prespecified factorial conditions without changing seeds or estimators.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import argparse
import hashlib
import importlib.util
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(str(_release_path('analysis')))
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"

spec = importlib.util.spec_from_file_location("simulation_core", ROOT / "code" / "52_measurement_deletion_simulation.py")
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)

PATIENT = None
SERIAL = None
TRUTH = None
DATABASE = None


def initialise_worker(patient, serial, truth, database):
    global PATIENT, SERIAL, TRUTH, DATABASE
    PATIENT, SERIAL, TRUTH, DATABASE = patient, serial, truth, database


def run_condition(args):
    retention, mechanism, strength, reps = args
    patient, serial, truth, database = PATIENT, SERIAL, TRUTH, DATABASE
    rows = []
    for rep in range(reps):
        condition_id = f"{database}|{retention}|{mechanism}|{strength}|{rep}"
        stable_hash = int(hashlib.sha256(condition_id.encode()).hexdigest()[:8], 16)
        seed = core.BASE_SEED + (stable_hash % 2_000_000_000)
        rng = np.random.default_rng(seed)
        sim = core.delete_and_reconstruct(patient, serial, mechanism, retention, strength, rng)
        f = sim.patient
        obs = f.R.eq(1) & f.y_reconstructed.notna()
        common = {
            "database": database, "retention_target": retention, "mechanism": mechanism,
            "strength": strength, "replicate": rep, "seed": seed,
            "measurement_retention_realized": sim.mean_measurement_retention,
            "outcome_observed_fraction": float(obs.mean()),
            "reconstructed_sensitivity": float(((f.y_reconstructed.eq(1)) & f.y_full.eq(1)).sum() / max(f.y_full.sum(), 1)),
            "reconstructed_specificity": float(((f.y_reconstructed.eq(0)) & f.y_full.eq(0) & obs).sum() / max((f.y_full.eq(0) & obs).sum(), 1)),
            "observability_probability_status": (
                "conditional_realized_history_approximation"
                if mechanism in {"history_MAR", "mixed_MNAR"}
                else "exact_under_independent_design_deletion"
            ),
        }
        rows.append(core.record("full_reference", "full", core.add_event_rate_inference(truth, truth["event_rate"]), common))
        if obs.sum() < 20 or f.loc[obs, "y_reconstructed"].nunique() < 2:
            continue
        yobs, pobs = f.loc[obs, "y_reconstructed"], f.loc[obs, "risk"]
        rows.append(core.record("naive", "reconstructed", core.add_event_rate_inference(core.weighted_metrics(yobs, pobs), truth["event_rate"]), common))
        w_raw = 1 / f.loc[obs, "q_observed"].clip(0.005, 1)
        raw_metrics = core.weighted_metrics(yobs, pobs, w_raw)
        raw_metrics.update({"weight_p99": float(w_raw.quantile(0.99)), "weight_max": float(w_raw.max()), "weight_truncated": 0})
        rows.append(core.record("IPAW_design_probability_untruncated", "reconstructed", core.add_event_rate_inference(raw_metrics, truth["event_rate"]), common))
        w = w_raw.clip(upper=w_raw.quantile(0.99))
        truncated_metrics = core.weighted_metrics(yobs, pobs, w)
        truncated_metrics.update({"weight_p99": float(w.quantile(0.99)), "weight_max": float(w.max()), "weight_truncated": 1})
        rows.append(core.record("IPAW_design_probability_truncated99", "reconstructed", core.add_event_rate_inference(truncated_metrics, truth["event_rate"]), common))
        aipw, aipw_se = core.aipw_event_rate(f)
        aipw_metrics = {k: np.nan for k in truth}
        aipw_metrics.update({"n": len(f), "events": aipw * len(f), "event_rate": aipw,
                             "mean_prediction": float(f.risk.mean()), "oe": aipw / f.risk.mean(),
                             "ess": float((w_raw.sum() ** 2) / np.square(w_raw).sum()),
                             "event_rate_se": aipw_se, "aipw_se": aipw_se})
        rows.append(core.record("AIPW_design_probability", "reconstructed", core.add_event_rate_inference(aipw_metrics, truth["event_rate"]), common))

        p_int_all, ok_int = core.crossfit_recalibration(f, rng, intercept_only=True)
        p_slope_all, ok_slope = core.crossfit_recalibration(f, rng, intercept_only=False)
        if ok_int:
            rows.append(core.record("recalibration_intercept_apparent", "reconstructed", core.weighted_metrics(f.loc[obs, "y_reconstructed"], p_int_all[obs]), common))
            rows.append(core.record("recalibration_intercept_truth", "full", core.weighted_metrics(f.y_full, p_int_all), common))
        if ok_slope:
            rows.append(core.record("recalibration_intercept_slope_apparent", "reconstructed", core.weighted_metrics(f.loc[obs, "y_reconstructed"], p_slope_all[obs]), common))
            rows.append(core.record("recalibration_intercept_slope_truth", "full", core.weighted_metrics(f.y_full, p_slope_all), common))

        reference_order = rng.permutation(len(f))
        for fraction in core.REFERENCE_FRACTIONS:
            sample_size = max(30, int(np.ceil(fraction * len(f))))
            val, evaluation = reference_order[:sample_size], reference_order[sample_size:]
            if np.unique(f.y_full.iloc[val]).size < 2 or len(evaluation) == 0:
                continue
            try:
                _, av, bv = core.recalibrate(f.risk.iloc[val], f.y_full.iloc[val], intercept_only=False)
            except Exception:
                continue
            p_reference = expit(av + bv * logit(f.risk.iloc[evaluation].clip(1e-6, 1 - 1e-6)))
            label = f"reference_{int(round(fraction * 100)):02d}pct_recalibration"
            reference_metrics = core.weighted_metrics(f.y_full.iloc[evaluation], p_reference)
            reference_metrics.update({"reference_sample_n": sample_size, "evaluation_n": len(evaluation)})
            rows.append(core.record(label, "full_heldout", reference_metrics, common))
        lo, hi = core.mnar_event_bounds(f, gamma=2.0)
        bound_metrics = {k: np.nan for k in truth}
        bound_metrics.update({"n": len(f), "event_rate": (lo + hi) / 2, "mnar_lower": lo,
                              "mnar_upper": hi, "mnar_covers_truth": int(lo <= truth["event_rate"] <= hi)})
        rows.append(core.record("Gamma2_prediction_sensitivity_region", "full", bound_metrics, common))
    return rows


def summarise(raw, truth):
    key = ["database", "retention_target", "mechanism", "strength", "method", "evaluation_target"]
    summaries = []
    for keys, g in raw.groupby(key, dropna=False):
        base = dict(zip(key, keys))
        for metric in ["event_rate", "oe", "brier", "auc", "calibration_intercept", "calibration_slope",
                       "outcome_observed_fraction", "reconstructed_sensitivity", "ess", "event_rate_se",
                       "weight_p99", "weight_max", "reference_sample_n", "evaluation_n"]:
            x = g[metric].dropna().to_numpy(float)
            if len(x):
                true_value = core.estimand_truth(base["method"], metric, truth)
                summaries.append({**base, "metric": metric, "n_replicates": len(x), "mean": x.mean(),
                                  "sd": x.std(ddof=1), "q025": np.quantile(x, .025), "q975": np.quantile(x, .975),
                                  "truth": true_value, "bias": x.mean() - true_value if np.isfinite(true_value) else np.nan,
                                  "rmse": np.sqrt(np.mean((x - true_value) ** 2)) if np.isfinite(true_value) else np.nan})
        x = g.mnar_covers_truth.dropna()
        if len(x):
            summaries.append({**base, "metric": "MNAR_event_rate_coverage", "n_replicates": len(x), "mean": x.mean()})
        x = g.event_rate_coverage.dropna()
        if len(x):
            summaries.append({**base, "metric": "event_rate_interval_coverage", "n_replicates": len(x), "mean": x.mean()})
    return pd.DataFrame(summaries)


def run_database(database, reps, workers):
    preparers = {"INSPIRE": core.prepare_inspire, "MIMIC": core.prepare_mimic, "EICU": core.prepare_eicu}
    patient, serial = preparers[database]()
    truth = core.weighted_metrics(patient.y_full, patient.risk)
    conditions = [(r, m, s, reps) for r in core.RETENTIONS for m in core.MECHANISMS for s in core.STRENGTHS]
    context = mp.get_context("fork")
    pieces = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialise_worker,
                             initargs=(patient, serial, truth, database)) as pool:
        for result in pool.map(run_condition, conditions, chunksize=1):
            pieces.extend(result)
    raw = pd.DataFrame(pieces)
    raw_path = SECURE / f"{database}_SIMULATION_REPLICATES_PARALLEL_SECURE.csv.gz"
    summary_path = TABLES / f"Table_{database.lower()}_simulation_summary_parallel.csv"
    raw.to_csv(raw_path, index=False, compression="gzip")
    summarise(raw, truth).to_csv(summary_path, index=False)
    audit = {"database": database, "replicates_per_condition": reps, "conditions": 36,
             "patient_n": len(patient), "events": int(patient.y_full.sum()), "serial_rows": len(serial),
             "full_reference_metrics": truth, "replicate_rows": len(raw), "parallel_workers": workers,
             "seed_definition": "identical SHA256-derived seed as serial production runner"}
    (OUTPUTS / f"{database}_SIMULATION_PARALLEL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", choices=["INSPIRE", "MIMIC", "EICU"], required=True)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    run_database(args.database, args.reps, args.workers)
