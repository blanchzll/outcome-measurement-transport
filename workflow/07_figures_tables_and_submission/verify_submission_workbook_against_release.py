#!/usr/bin/env python3
"""Verify workbook sheets semantically against frozen aggregate release CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(value).strip() for value in result.columns]
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--release-tables", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    workbook = load_workbook(args.workbook, read_only=True, data_only=True)
    index_rows = list(workbook["INDEX"].values)
    index = pd.DataFrame(index_rows[1:], columns=index_rows[0])
    compared = []
    conflicts = []
    for row in index.itertuples(index=False):
        source = args.release_tables / str(row.source_file)
        if not source.is_file():
            continue
        values = list(workbook[str(row.sheet)].values)
        observed = clean(pd.DataFrame(values[1:], columns=values[0]))
        expected = clean(pd.read_csv(source))
        try:
            pd.testing.assert_frame_equal(
                observed,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-10,
                atol=1e-12,
            )
        except AssertionError as error:
            conflicts.append({
                "source_file": source.name,
                "sheet": str(row.sheet),
                "observed_shape": list(observed.shape),
                "expected_shape": list(expected.shape),
                "difference": str(error)[:1500],
            })
        compared.append({"source_file": source.name, "sheet": str(row.sheet)})
    workbook.close()

    release_commit = subprocess.run(
        ["git", "-C", str(args.release_tables.parents[1]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    payload = {
        "status": "PASS" if not conflicts else "FAIL",
        "workbook": str(args.workbook),
        "workbook_sha256": sha256(args.workbook),
        "release_tables": str(args.release_tables),
        "release_commit": release_commit,
        "tables_compared": len(compared),
        "conflicts": conflicts,
        "comparison": "column names, row order, strings and numeric values; numeric tolerance rtol=1e-10 and atol=1e-12",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "tables_compared": len(compared), "conflicts": len(conflicts)}, indent=2))
    if conflicts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
