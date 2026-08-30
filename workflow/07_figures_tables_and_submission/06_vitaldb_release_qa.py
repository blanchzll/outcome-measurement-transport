# %% [markdown]
# # VitalDB extension release QA

# %%
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_METHODS = {
    "full_reference",
    "naive",
    "IPAW_design_probability_untruncated",
    "IPAW_design_probability_truncated99",
    "AIPW_design_probability",
    "recalibration_intercept_apparent",
    "recalibration_intercept_truth",
    "recalibration_intercept_slope_apparent",
    "recalibration_intercept_slope_truth",
    "reference_05pct_recalibration",
    "reference_10pct_recalibration",
    "reference_20pct_recalibration",
    "reference_30pct_recalibration",
    "Gamma2_prediction_sensitivity_region",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-audit", required=True, type=Path)
    parser.add_argument("--simulation-audit", required=True, type=Path)
    parser.add_argument("--simulation-summary", required=True, type=Path)
    parser.add_argument("--schedule-audit", required=True, type=Path)
    parser.add_argument("--schedule-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    endpoint = json.loads(args.endpoint_audit.read_text(encoding="utf-8"))
    simulation = json.loads(args.simulation_audit.read_text(encoding="utf-8"))
    schedule = json.loads(args.schedule_audit.read_text(encoding="utf-8"))
    sim_table = pd.read_csv(args.simulation_summary)
    schedule_table = pd.read_csv(args.schedule_summary)

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    check("endpoint explicitly creatinine-only", "creatinine-only" in endpoint["endpoint"], endpoint["endpoint"])
    check("dense-reference event count supports mechanism model", endpoint["audit"]["n_dense_reference"] >= 1000, endpoint["audit"]["n_dense_reference"])
    check("simulation uses 300 replicates", simulation["replicates_per_condition"] == 300, simulation["replicates_per_condition"])
    check("held-out model discrimination is informative", simulation["full_reference_metrics"]["auc"] >= 0.65, simulation["full_reference_metrics"]["auc"])
    methods = set(sim_table.method.dropna().astype(str))
    check("simulation method contract complete", EXPECTED_METHODS.issubset(methods), sorted(EXPECTED_METHODS - methods))
    expected_pairs = {
        "EICU->VitalDB", "INSPIRE->VitalDB", "MIMIC->VitalDB",
        "VitalDB->EICU", "VitalDB->INSPIRE", "VitalDB->MIMIC", "VitalDB->VitalDB",
    }
    check("empirical schedule extension has all VitalDB pairs", set(schedule["computed_pairs"]) == expected_pairs, schedule["computed_pairs"])
    check("empirical schedule uses 200 replicates", schedule["replicates_per_condition"] == 200, schedule["replicates_per_condition"])
    check("schedule summary has no duplicate estimand rows", not schedule_table.duplicated([
        "target_database", "donor_schedule_database", "tolerance_hours", "method", "evaluation_target", "metric"
    ]).any(), int(schedule_table.duplicated([
        "target_database", "donor_schedule_database", "tolerance_hours", "method", "evaluation_target", "metric"
    ]).sum()))
    numeric = sim_table[["mean", "q025", "q975"]].apply(pd.to_numeric, errors="coerce")
    check("reported simulation means are finite", np.isfinite(numeric["mean"].dropna()).all(), int(numeric["mean"].notna().sum()))

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
        "claim_boundaries": [
            "VitalDB outcome is a creatinine-only operational reference, not complete KDIGO.",
            "Dense-reference estimates are conditional on intense postoperative measurement.",
            "Empirical schedules are observed measurement patterns and not causal hospital policies.",
            "VitalDB-to-source model transport is endpoint transport, not same-expert-endpoint validation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
