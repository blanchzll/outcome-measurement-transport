# %% [markdown]
# # VitalDB download and integrity audit
#
# Jupytext-compatible `py:percent` script. It verifies the official PhysioNet
# checksums without modifying raw data.

# %%
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


CORE_TABLES = (
    "clinical_data.csv",
    "clinical_parameters.csv",
    "lab_data.csv",
    "lab_parameters.csv",
)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        digest, name = parts
        checksums[name.lstrip("*./")] = digest.lower()
    return checksums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checksum_path = args.data_root / "SHA256SUMS.txt"
    expected = parse_checksum_file(checksum_path) if checksum_path.exists() else {}
    audited_names = sorted(set(CORE_TABLES) | {"track_names.csv"})
    records = []
    for name in audited_names:
        path = args.data_root / name
        record = {
            "file": name,
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "expected_sha256": expected.get(name),
            "observed_sha256": None,
            "checksum_match": False,
        }
        if path.exists() and record["expected_sha256"]:
            record["observed_sha256"] = sha256(path)
            record["checksum_match"] = record["observed_sha256"] == record["expected_sha256"]
        records.append(record)

    by_name = {item["file"]: item for item in records}
    core_pass = all(by_name[name]["checksum_match"] for name in CORE_TABLES)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(args.data_root),
        "status": "PASS_CORE_TABLES" if core_pass else "WAITING_FOR_COMPLETE_CORE_TABLES",
        "core_tables": list(CORE_TABLES),
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "files": records}, indent=2))


if __name__ == "__main__":
    main()
