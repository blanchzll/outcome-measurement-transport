#!/usr/bin/env python3
"""Compare locked 3,710 and screened 4,014 LOCO prediction analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, roc_auc_score


SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slope(y: np.ndarray, p: np.ndarray) -> float:
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    design = np.column_stack([np.ones(len(z)), z])
    def objective(beta):
        eta = design @ beta
        return -np.sum(y * eta - np.logaddexp(0, eta))
    fit = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return float(fit.x[1]) if fit.success else np.nan


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "oe_ratio": float(y.sum() / p.sum()),
        "calibration_slope": slope(y, p),
    }


def analyse(path: Path, label: str, n_bootstrap: int) -> tuple[dict, dict]:
    frame = pd.read_csv(
        path,
        usecols=["Center", "PostopAKI", "Gender", "pred_PI_restricted_rf"],
        low_memory=False,
    )
    y = frame.PostopAKI.to_numpy(int)
    p = frame.pred_PI_restricted_rf.to_numpy(float)
    centre = frame.Center.to_numpy()
    point = metrics(y, p)
    # Reuse the primary 3,710-cohort seed so its interval is identical to the
    # authoritative main-table rebuild. Use a distinct, declared seed for the
    # separately fitted 4,014-record sensitivity analysis.
    random_seed = SEED if label == "locked_3710" else SEED + 1
    rng = np.random.default_rng(random_seed)
    draws = {name: [] for name in point}
    event_counts = []
    strata = [np.flatnonzero(centre == value) for value in pd.unique(centre)]
    for _ in range(n_bootstrap):
        index = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        event_counts.append(int(y[index].sum()))
        result = metrics(y[index], p[index])
        for name, value in result.items():
            if np.isfinite(value):
                draws[name].append(value)
    row = {
        "analysis_population": label,
        "n": int(len(frame)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "sex_male": int(frame.Gender.eq("Male").sum()),
        "sex_female": int(frame.Gender.eq("Female").sum()),
        "sex_missing_or_unresolved": int(frame.Gender.isna().sum()),
        "model": "perioperative_restricted_rf",
        "predictions": "locked_leave_one_centre_out",
    }
    for name, value in point.items():
        row[name] = value
        row[f"{name}_bootstrap_q025"] = float(np.quantile(draws[name], 0.025))
        row[f"{name}_bootstrap_q975"] = float(np.quantile(draws[name], 0.975))
    diagnostic = {
        "analysis_population": label,
        "bootstrap_unit": "analytic record within centre",
        "bootstrap_replicates": n_bootstrap,
        "random_seed": random_seed,
        "centre_sizes_preserved": True,
        "outcome_stratified": False,
        "event_count_mean": float(np.mean(event_counts)),
        "event_count_sd": float(np.std(event_counts, ddof=1)),
        "event_count_q025": float(np.quantile(event_counts, 0.025)),
        "event_count_q975": float(np.quantile(event_counts, 0.975)),
        "oe_q025": row["oe_ratio_bootstrap_q025"],
        "oe_q975": row["oe_ratio_bootstrap_q975"],
        "uncertainty_included": "validation-sample uncertainty conditional on locked predictions",
        "uncertainty_excluded": "model development, hyperparameter selection and model-selection uncertainty",
    }
    return row, diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-3710", required=True, type=Path)
    parser.add_argument("--cohort-4014", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, diagnostics = [], []
    for path, label in ((args.cohort_3710, "locked_3710"), (args.cohort_4014, "screened_4014_sensitivity")):
        row, diagnostic = analyse(path, label, args.bootstrap)
        rows.append(row)
        diagnostics.append(diagnostic)
    table = args.output_dir / "Table_source_screened_cohort_sensitivity.csv"
    diagnostic_table = args.output_dir / "Table_source_bootstrap_uncertainty_diagnostics.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    pd.DataFrame(diagnostics).to_csv(diagnostic_table, index=False)
    payload = {
        "status": "PASS",
        "inputs": {str(args.cohort_3710): sha256(args.cohort_3710), str(args.cohort_4014): sha256(args.cohort_4014)},
        "outputs": {table.name: sha256(table), diagnostic_table.name: sha256(diagnostic_table)},
        "interpretation": "The 4,014 analysis is a screened-population sensitivity with separately locked LOCO fits; it does not identify the effect of any one exclusion or repair source coding.",
        "patient_level_output_written": False,
    }
    audit = args.output_dir / "SOURCE_SCREENED_COHORT_SENSITIVITY_AUDIT.json"
    audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
