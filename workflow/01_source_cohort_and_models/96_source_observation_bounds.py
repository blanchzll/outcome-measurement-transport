#!/usr/bin/env python3
# %% [markdown]
# # Source-cohort post-discharge outcome sensitivity bounds

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260829
FRACTIONS = (0.00, 0.02, 0.05, 0.10, 0.20)


def stable_seed(*parts: object) -> int:
    token = "|".join(map(str, parts))
    return SEED + int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 2_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    for name in ("secure_work", "tables", "outputs"):
        (args.output_root / name).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.base / "code"))
    from ascertainment_stress import weighted_metrics

    usecols = ["MajorID", "Center", "PostopAKI", "PostopHospitalDays", "pred_PI_restricted_rf"]
    frame = pd.read_csv(args.base / "secure_work/SOURCE_4014_LOCKED_LOCO_PREDICTIONS_SECURE.csv.gz", usecols=usecols)
    frame["PostopHospitalDays"] = pd.to_numeric(frame.PostopHospitalDays, errors="coerce")
    short_negative = frame.PostopHospitalDays.lt(7) & frame.PostopAKI.eq(0)
    candidates = np.flatnonzero(short_negative.to_numpy())
    rows = []
    for fraction in FRACTIONS:
        count = int(round(fraction * len(candidates)))
        for mechanism in ("random", "highest_predicted_risk"):
            replicates = 1 if mechanism == "highest_predicted_risk" or fraction == 0 else args.replicates
            for replicate in range(replicates):
                outcome = frame.PostopAKI.to_numpy(int).copy()
                if count:
                    if mechanism == "random":
                        rng = np.random.default_rng(stable_seed(fraction, mechanism, replicate))
                        selected = rng.choice(candidates, size=count, replace=False)
                    else:
                        order = np.argsort(frame.pred_PI_restricted_rf.to_numpy()[candidates])[::-1]
                        selected = candidates[order[:count]]
                    outcome[selected] = 1
                metrics = weighted_metrics(outcome, frame.pred_PI_restricted_rf)
                rows.append({
                    "assumed_postdischarge_event_fraction": fraction,
                    "assignment_mechanism": mechanism,
                    "replicate": replicate,
                    "short_stay_recorded_negative_n": int(len(candidates)),
                    "added_events": count,
                    **metrics,
                })
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_root / "secure_work/SOURCE_POSTDISCHARGE_BOUNDS_REPLICATES_SECURE.csv.gz", index=False, compression="gzip")
    summary_rows = []
    metrics = ["events", "event_rate", "oe", "brier", "auc", "calibration_intercept", "calibration_slope"]
    for keys, group in raw.groupby(["assumed_postdischarge_event_fraction", "assignment_mechanism"]):
        for metric in metrics:
            x = group[metric].dropna().to_numpy(float)
            summary_rows.append({
                "assumed_postdischarge_event_fraction": keys[0],
                "assignment_mechanism": keys[1],
                "metric": metric,
                "n_replicates": len(x),
                "mean": float(x.mean()),
                "q025": float(np.quantile(x, 0.025)),
                "q975": float(np.quantile(x, 0.975)),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_root / "tables/Table_source_postdischarge_sensitivity_bounds.csv", index=False)
    audit = {
        "status": "PASS",
        "source_n": int(len(frame)),
        "recorded_events": int(frame.PostopAKI.sum()),
        "postoperative_stay_below_7d": int(frame.PostopHospitalDays.lt(7).sum()),
        "short_stay_recorded_negative_n": int(len(candidates)),
        "assumed_event_fractions": list(FRACTIONS),
        "interpretation_boundary": "Sensitivity bounds do not identify post-discharge AKI or event timing.",
    }
    (args.output_root / "outputs/SOURCE_POSTDISCHARGE_BOUNDS_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
