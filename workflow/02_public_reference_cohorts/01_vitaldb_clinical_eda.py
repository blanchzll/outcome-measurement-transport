# %% [markdown]
# # VitalDB clinical table audit
#
# This notebook-style script audits case and patient structure, timestamps,
# eligibility variables, missingness, and surgical mix before outcome creation.

# %%
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CLINICAL_SHA256 = "7d6edb471e5eee3fde75e417084240c97bdbf6eff41cbd61e5dace44f1585ecf"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    desc = values.describe(percentiles=[0.25, 0.5, 0.75])
    return {
        "n": int(desc["count"]),
        "min": float(desc["min"]),
        "p25": float(desc["25%"]),
        "median": float(desc["50%"]),
        "p75": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def count_true(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    observed_hash = sha256(args.clinical_csv)
    if observed_hash != EXPECTED_CLINICAL_SHA256:
        raise RuntimeError(f"clinical_data.csv checksum mismatch: {observed_hash}")

    data = pd.read_csv(args.clinical_csv, encoding="utf-8-sig", low_memory=False)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    required = ["caseid", "subjectid", "opend"]
    missing_required = [name for name in required if name not in data.columns]
    if missing_required:
        raise KeyError(f"Missing required columns: {missing_required}")

    caseid = data["caseid"]
    subjectid = data["subjectid"]
    cases_per_subject = data.groupby("subjectid", dropna=False)["caseid"].nunique().sort_values(ascending=False)
    repeat_subjects = cases_per_subject[cases_per_subject > 1]

    missingness = (
        data.isna().mean().mul(100).rename("missing_percent").to_frame()
        .assign(n_missing=data.isna().sum())
        .sort_values(["missing_percent", "n_missing"], ascending=False)
    )
    missingness.to_csv(out / "vitaldb_clinical_missingness.csv", index_label="variable")

    category_rows: list[dict[str, object]] = []
    for column in ("department", "optype", "sex", "emop", "asa"):
        if column not in data.columns:
            continue
        counts = data[column].fillna("<MISSING>").astype(str).value_counts(dropna=False).head(30)
        category_rows.extend(
            {"variable": column, "level": level, "n": int(n), "percent": float(100 * n / len(data))}
            for level, n in counts.items()
        )
    pd.DataFrame(category_rows).to_csv(out / "vitaldb_clinical_category_counts.csv", index=False)

    opend = pd.to_numeric(data["opend"], errors="coerce")
    opstart = pd.to_numeric(data["opstart"], errors="coerce") if "opstart" in data else pd.Series(np.nan, index=data.index)
    caseend = pd.to_numeric(data["caseend"], errors="coerce") if "caseend" in data else pd.Series(np.nan, index=data.index)
    dis = pd.to_numeric(data["dis"], errors="coerce") if "dis" in data else pd.Series(np.nan, index=data.index)
    postop_los_days = (dis - opend) / 86400

    optype = data["optype"].fillna("").astype(str) if "optype" in data else pd.Series("", index=data.index)
    gi_mask = optype.str.contains(r"stomach|colorectal", case=False, regex=True)
    age = pd.to_numeric(data["age"], errors="coerce") if "age" in data else pd.Series(np.nan, index=data.index)
    preop_cr = pd.to_numeric(data["preop_cr"], errors="coerce") if "preop_cr" in data else pd.Series(np.nan, index=data.index)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "clinical_sha256": observed_hash,
        "n_rows": int(len(data)),
        "n_columns": int(data.shape[1]),
        "n_unique_caseid": int(caseid.nunique(dropna=True)),
        "n_duplicate_caseid_rows": int(caseid.duplicated(keep=False).sum()),
        "n_unique_subjectid": int(subjectid.nunique(dropna=True)),
        "n_missing_subjectid": int(subjectid.isna().sum()),
        "n_subjects_with_multiple_cases": int(len(repeat_subjects)),
        "n_cases_in_repeat_subjects": int(cases_per_subject[repeat_subjects.index].sum()) if len(repeat_subjects) else 0,
        "maximum_cases_per_subject": int(cases_per_subject.max()),
        "n_age_18_or_older": count_true(age >= 18),
        "n_valid_preop_cr_0_1_to_20": count_true(preop_cr.between(0.1, 20.0, inclusive="both")),
        "n_gi_stomach_or_colorectal": count_true(gi_mask),
        "timestamp_checks": {
            "n_opend_missing": int(opend.isna().sum()),
            "n_opend_not_after_opstart": count_true(opend <= opstart),
            "n_caseend_before_opend": count_true(caseend < opend),
            "n_discharge_before_opend": count_true(dis < opend),
        },
        "postoperative_los_days": numeric_summary(postop_los_days),
        "age": numeric_summary(age),
        "preop_creatinine": numeric_summary(preop_cr),
        "top_missing_variables": [
            {"variable": str(idx), "missing_percent": float(row["missing_percent"]), "n_missing": int(row["n_missing"])}
            for idx, row in missingness.head(20).iterrows()
        ],
    }
    (out / "VITALDB_CLINICAL_EDA.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    lines = [
        "# VitalDB clinical table EDA",
        "",
        f"Generated: {audit['generated_at_utc']}",
        "",
        "## Cohort and identifiers",
        "",
        f"- Rows / unique cases: {audit['n_rows']} / {audit['n_unique_caseid']}",
        f"- Unique patients: {audit['n_unique_subjectid']}",
        f"- Patients with multiple cases: {audit['n_subjects_with_multiple_cases']} ({audit['n_cases_in_repeat_subjects']} cases)",
        f"- Maximum cases per patient: {audit['maximum_cases_per_subject']}",
        f"- Adults (age >=18): {audit['n_age_18_or_older']}",
        f"- Stomach or colorectal operations: {audit['n_gi_stomach_or_colorectal']}",
        "",
        "## Prediction-landmark checks",
        "",
        f"- Missing operation end: {audit['timestamp_checks']['n_opend_missing']}",
        f"- Operation end not after operation start: {audit['timestamp_checks']['n_opend_not_after_opstart']}",
        f"- Case end before operation end: {audit['timestamp_checks']['n_caseend_before_opend']}",
        f"- Discharge before operation end: {audit['timestamp_checks']['n_discharge_before_opend']}",
        "",
        "## Endpoint readiness",
        "",
        f"- Valid preoperative creatinine field (0.1-20 mg/dL): {audit['n_valid_preop_cr_0_1_to_20']}",
        "- Timestamped longitudinal creatinine eligibility will be audited only after `lab_data.csv` passes its official checksum.",
        "- This dataset supports a creatinine-only operational endpoint, not complete expert-adjudicated KDIGO.",
        "",
        "## Repeated-operation estimand",
        "",
        "Because absolute calendar dates are removed and case IDs are random, primary model and simulation analyses will use patients with exactly one recorded operation. All-operation sensitivity analyses will use patient-clustered uncertainty.",
        "",
    ]
    (out / "VITALDB_CLINICAL_EDA.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
