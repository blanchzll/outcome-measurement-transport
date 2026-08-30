# %% [markdown]
# # VitalDB longitudinal creatinine operational reference
#
# Jupytext-compatible `py:percent` script. It creates a case-level internal
# analysis file and an aggregate selection/event ledger.

# %%
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vitaldb_endpoint import build_creatinine_endpoint


EXPECTED = {
    "clinical_data.csv": "7d6edb471e5eee3fde75e417084240c97bdbf6eff41cbd61e5dace44f1585ecf",
    "lab_data.csv": "c6e84fb397afe8182a7e6cc3aac6b34502d6ac0fadf1abf0ef643cc8bd50ea8b",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def cohort_row(label: str, frame: pd.DataFrame) -> dict[str, object]:
    observable = frame.loc[frame["reference_observable"]]
    dense = frame.loc[frame["dense_reference"]]
    return {
        "cohort": label,
        "n_operations": int(len(frame)),
        "n_patients": int(frame["subjectid"].nunique()),
        "n_reference_observable": int(len(observable)),
        "reference_observable_percent": float(100 * len(observable) / len(frame)) if len(frame) else None,
        "n_dense_reference": int(len(dense)),
        "dense_reference_percent": float(100 * len(dense) / len(frame)) if len(frame) else None,
        "n_creatinine_events_observable": int(observable["creatinine_event_168h"].astype(bool).sum()),
        "event_rate_observable_percent": float(100 * observable["creatinine_event_168h"].astype(bool).mean()) if len(observable) else None,
        "n_creatinine_events_dense": int(dense["creatinine_event_168h"].astype(bool).sum()),
        "event_rate_dense_percent": float(100 * dense["creatinine_event_168h"].astype(bool).mean()) if len(dense) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    for name, expected_hash in EXPECTED.items():
        path = args.data_root / name
        observed = sha256(path)
        if observed != expected_hash:
            raise RuntimeError(f"{name} checksum mismatch: {observed}")

    clinical = pd.read_csv(args.data_root / "clinical_data.csv", encoding="utf-8-sig", low_memory=False)
    labs = pd.read_csv(args.data_root / "lab_data.csv", encoding="utf-8-sig", low_memory=False)
    case_level, audit = build_creatinine_endpoint(clinical, labs)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    case_level.to_csv(out / "vitaldb_case_level_creatinine_reference_INTERNAL.csv", index=False)
    opend = pd.to_numeric(clinical.set_index("caseid")["opend"], errors="coerce")
    serial = labs.loc[labs["name"].astype(str).str.lower().eq("cr"), ["caseid", "dt", "result"]].copy()
    for column in ("caseid", "dt", "result"):
        serial[column] = pd.to_numeric(serial[column], errors="coerce")
    serial = serial.dropna(subset=["caseid", "dt", "result"])
    serial = serial.loc[serial["result"].between(0.1, 20.0, inclusive="both")]
    serial = serial.drop_duplicates(["caseid", "dt", "result"], keep="first")
    serial["opend"] = serial["caseid"].map(opend)
    serial["hours_from_opend"] = (serial["dt"] - serial["opend"]) / 3600
    serial = serial.loc[(serial["hours_from_opend"] > 0) & (serial["hours_from_opend"] <= 168)]
    serial = serial.merge(
        case_level[["caseid", "subjectid", "baseline_cr", "reference_observable", "dense_reference"]],
        on="caseid",
        how="inner",
        validate="many_to_one",
    )
    serial.to_csv(out / "vitaldb_creatinine_serial_0_168h_INTERNAL.csv", index=False)

    adults = case_level.loc[case_level["adult"]].copy()
    gi = adults.loc[adults["gi_stomach_colorectal"]].copy()
    ledger = pd.DataFrame(
        [
            cohort_row("all operations", case_level),
            cohort_row("adult operations", adults),
            cohort_row("adult stomach or colorectal operations", gi),
        ]
    )
    ledger.to_csv(out / "VITALDB_ENDPOINT_COHORT_LEDGER.csv", index=False)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "VitalDB 1.0.0",
        "prediction_landmark": "operation end (opend)",
        "endpoint": "creatinine-only postoperative AKI operational reference",
        "endpoint_components": {
            "absolute": ">=0.3 mg/dL increase within 48 hours",
            "ratio": ">=1.5 times the latest timestamped pre-operation-end creatinine within 168 hours",
        },
        "not_available": ["complete postoperative urine output", "timed RRT initiation", "expert KDIGO adjudication"],
        "audit": audit,
        "cohorts": ledger.to_dict(orient="records"),
    }
    (out / "VITALDB_ENDPOINT_AUDIT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# VitalDB creatinine reference endpoint audit",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "The endpoint is a creatinine-only operational reference anchored at operation end. It is not complete expert-adjudicated KDIGO.",
        "",
        "```csv",
        ledger.to_csv(index=False, float_format="%.2f").strip(),
        "```",
        "",
        "## Interpretation boundary",
        "",
        "Dense-reference results estimate performance conditional on intense postoperative creatinine observation. They do not identify performance in the full surgical population without additional assumptions.",
        "",
    ]
    (out / "VITALDB_ENDPOINT_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
