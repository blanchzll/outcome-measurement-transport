# %% [markdown]
# # Rebuild simulation summaries with method-specific estimands
# Distinguishes estimation of original-model performance from evaluation of model updating.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(str(_release_path('analysis')))
SECURE, TABLES, OUTPUTS = ROOT / "secure_work", ROOT / "tables", ROOT / "outputs"
spec = importlib.util.spec_from_file_location("simulation_core", ROOT / "code" / "52_measurement_deletion_simulation.py")
core = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(core)

audits = {}
for database in ("INSPIRE", "MIMIC", "EICU"):
    raw = pd.read_csv(SECURE / f"{database}_SIMULATION_REPLICATES_SECURE.csv.gz")
    truth = json.loads((OUTPUTS / f"{database}_SIMULATION_AUDIT.json").read_text())["full_reference_metrics"]
    key = ["database", "retention_target", "mechanism", "strength", "method", "evaluation_target"]
    summaries = []
    for keys, g in raw.groupby(key, dropna=False):
        base = dict(zip(key, keys))
        for metric in ["event_rate", "oe", "brier", "auc", "calibration_intercept", "calibration_slope",
                       "outcome_observed_fraction", "reconstructed_sensitivity", "ess", "event_rate_se",
                       "weight_p99", "weight_max", "reference_sample_n", "evaluation_n"]:
            x = g[metric].dropna().to_numpy(float)
            if not len(x):
                continue
            target = core.estimand_truth(base["method"], metric, truth)
            summaries.append({**base, "metric": metric, "n_replicates": len(x), "mean": x.mean(),
                              "sd": x.std(ddof=1), "q025": np.quantile(x, .025), "q975": np.quantile(x, .975),
                              "truth": target, "bias": x.mean() - target if np.isfinite(target) else np.nan,
                              "rmse": np.sqrt(np.mean((x - target) ** 2)) if np.isfinite(target) else np.nan})
        x = g.mnar_covers_truth.dropna()
        if len(x):
            summaries.append({**base, "metric": "MNAR_event_rate_coverage", "n_replicates": len(x), "mean": x.mean()})
        x = g.event_rate_coverage.dropna()
        if len(x):
            summaries.append({**base, "metric": "event_rate_interval_coverage", "n_replicates": len(x), "mean": x.mean()})
        if {"event_rate_ci_lower", "event_rate_ci_upper"}.issubset(g.columns):
            width = (g.event_rate_ci_upper - g.event_rate_ci_lower).dropna().to_numpy(float)
            if len(width):
                summaries.append({**base, "metric": "event_rate_interval_width", "n_replicates": len(width),
                                  "mean": width.mean(), "sd": width.std(ddof=1),
                                  "q025": np.quantile(width, .025), "q975": np.quantile(width, .975)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(TABLES / f"Table_{database.lower()}_simulation_summary.csv", index=False)
    failures = (
        summary.groupby(["database", "retention_target", "mechanism", "strength", "method", "evaluation_target"],
                        dropna=False, as_index=False)
        .agg(min_available_replicates=("n_replicates", "min"), max_available_replicates=("n_replicates", "max"))
    )
    failures["nominal_replicates"] = 300
    failures["maximum_metric_failure_fraction"] = 1 - failures.min_available_replicates / 300
    failures.to_csv(TABLES / f"Table_{database.lower()}_simulation_failure_diagnostics.csv", index=False)
    audits[database] = {"rows": len(summary), "minimum_replicates": int(summary.n_replicates.min()),
                        "maximum_metric_failure_fraction": float(failures.maximum_metric_failure_fraction.max()),
                        "recalibration_targets": {"oe": 1.0, "calibration_intercept": 0.0, "calibration_slope": 1.0},
                        "original_model_target": truth}
(OUTPUTS / "SIMULATION_ESTIMAND_RESUMMARY_AUDIT.json").write_text(json.dumps(audits, indent=2) + "\n")
print(json.dumps(audits, indent=2))
