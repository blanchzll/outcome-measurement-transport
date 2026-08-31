#!/usr/bin/env python3
"""Rebuild source-model fixed-prediction intervals with event counts free to vary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.metrics import brier_score_loss, roc_auc_score


MODELS = (
    ("PI", "ridge", "pred_PI_ridge"),
    ("PI", "restricted_rf", "pred_PI_restricted_rf"),
    ("PI", "gradient_boosting", "pred_PI_gradient_boosting"),
    ("H", "restricted_rf", "pred_H_restricted_rf"),
)
SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
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
        "calibration_slope": calibration_slope(y, p),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-table", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    columns = ["Center", "PostopAKI"] + [item[2] for item in MODELS]
    frame = pd.read_csv(args.predictions, usecols=columns, low_memory=False)
    y = frame.PostopAKI.to_numpy(int)
    centre = frame.Center.to_numpy()
    strata = [np.flatnonzero(centre == value) for value in pd.unique(centre)]
    probabilities = {column: frame[column].to_numpy(float) for _, _, column in MODELS}
    points = {column: metrics(y, p) for column, p in probabilities.items()}
    draws = {column: {metric: [] for metric in points[column]} for column in probabilities}
    event_counts = []
    rng = np.random.default_rng(SEED)
    for _ in range(args.bootstrap):
        index = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        event_counts.append(int(y[index].sum()))
        for column, p in probabilities.items():
            result = metrics(y[index], p[index])
            for metric, value in result.items():
                if np.isfinite(value):
                    draws[column][metric].append(value)

    rows = []
    for feature_set, model, column in MODELS:
        point = points[column]
        row = {
            "database": "source_3710",
            "feature_set": feature_set,
            "model": model,
            "n": int(len(frame)),
            "events": int(y.sum()),
        }
        for metric, value in point.items():
            row[metric] = value
            row[f"{metric}_ci_lower"] = float(np.quantile(draws[column][metric], 0.025))
            row[f"{metric}_ci_upper"] = float(np.quantile(draws[column][metric], 0.975))
        rows.append(row)
    order = [
        "database", "feature_set", "model", "n", "events",
        "roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper",
        "brier", "brier_ci_lower", "brier_ci_upper",
        "oe_ratio", "oe_ratio_ci_lower", "oe_ratio_ci_upper",
        "calibration_slope", "calibration_slope_ci_lower", "calibration_slope_ci_upper",
    ]
    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[order].to_csv(args.output_table, index=False)
    payload = {
        "status": "PASS",
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_unit": "analytic record within centre",
        "centre_sizes_preserved": True,
        "outcome_stratified": False,
        "event_count_mean": float(np.mean(event_counts)),
        "event_count_sd": float(np.std(event_counts, ddof=1)),
        "event_count_q025": float(np.quantile(event_counts, 0.025)),
        "event_count_q975": float(np.quantile(event_counts, 0.975)),
        "uncertainty_included": "validation-sample uncertainty conditional on locked outer-fold predictions",
        "uncertainty_excluded": "model development, tuning and model-selection uncertainty",
        "input_sha256": sha256(args.predictions),
        "output_sha256": sha256(args.output_table),
        "patient_level_output_written": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
