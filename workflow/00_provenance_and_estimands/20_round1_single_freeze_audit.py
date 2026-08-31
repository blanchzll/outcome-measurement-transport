#!/usr/bin/env python3
"""Assert that Round-1 submission artifacts share one declared result freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, checks: list[dict]) -> None:
    checks.append({"check": message, "status": "PASS" if condition else "FAIL"})
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    source = root / "manuscript_sources"
    tables = root / "submission_package_work/delivery"
    qa = root / "submission_package_work/qa"
    manuscript = (source / "MANUSCRIPT_NATURE_COMMUNICATIONS.md").read_text(encoding="utf-8")
    supplement = (source / "SUPPLEMENTARY_INFORMATION.md").read_text(encoding="utf-8")
    combined = manuscript + "\n" + supplement
    checks: list[dict] = []

    for expected in (
        "39, 42, 56, 14 and one",
        "0.6809",
        "0.3746",
        "0.8793",
        "1.1413",
        "1.5704",
        "0.8436-1.1307",
        "295 unresolved sex",
    ):
        require(expected in combined, f"current text contains {expected}", checks)
    for stale in ("39, 43, 56, 16", "0.6861", "0.3576", "1.1460", "1.6045", "0.9532-1.0102"):
        require(stale not in combined, f"stale value absent: {stale}", checks)

    table1_path = tables / "main/tables/Table1_source_model_results.csv"
    table1 = pd.read_csv(table1_path)
    require(table1.n.eq(3710).all(), "all primary source-model rows use n=3710", checks)
    require(table1.events.eq(152).all(), "all primary source-model rows use 152 events", checks)
    rf = table1.loc[(table1.feature_set == "PI") & (table1.model == "restricted_rf")].iloc[0]
    require(abs(rf.oe_ratio_ci_lower - 0.8436146377524048) < 1e-12, "authoritative RF O/E lower interval", checks)
    require(abs(rf.oe_ratio_ci_upper - 1.1307314835073594) < 1e-12, "authoritative RF O/E upper interval", checks)

    schedule_path = qa / "FIGURE5_PRIMARY_12H_AUDIT.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    require(schedule["status"] == "PASS", "Figure 5 audit passes", checks)
    require(schedule["primary_tolerance_hours"] == 12, "Figure 5 uses primary 12-hour tolerance", checks)
    require(schedule["cells"] == 32 and schedule["replicates_per_cell"] == 200, "Figure 5 contains both metrics for 16 cells and 200 replicates", checks)

    roles_path = tables / "supplement/tables/Table_dataset_roles_and_inference_boundaries.csv"
    roles = pd.read_csv(roles_path)
    source_role = roles.loc[roles.dataset.eq("Five-centre source cohort")].iloc[0]
    require("3710-patient locked analysis cohort with 152" in source_role.analysis_population, "source role ledger declares locked primary cohort", checks)
    require("screened denominator 4014" in source_role.analysis_population, "source role ledger declares screened sensitivity denominator", checks)

    sensitivity_path = tables / "supplement/tables/Table_source_screened_cohort_sensitivity.csv"
    sensitivity = pd.read_csv(sensitivity_path)
    require(set(sensitivity.analysis_population) == {"locked_3710", "screened_4014_sensitivity"}, "screened sensitivity contains both declared populations", checks)
    require(int(sensitivity.loc[sensitivity.analysis_population.eq("screened_4014_sensitivity"), "sex_missing_or_unresolved"].iloc[0]) == 295, "screened sensitivity retains 295 unresolved sex records", checks)

    tracked = [
        source / "MANUSCRIPT_NATURE_COMMUNICATIONS.md",
        source / "SUPPLEMENTARY_INFORMATION.md",
        table1_path,
        roles_path,
        sensitivity_path,
        schedule_path,
    ]
    payload = {
        "status": "PASS",
        "freeze": {"primary_n": 3710, "primary_events": 152, "screened_n": 4014, "screened_events": 155},
        "checks": checks,
        "artifact_sha256": {str(path.relative_to(root)): sha256(path) for path in tracked},
        "patient_level_output_written": False,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
