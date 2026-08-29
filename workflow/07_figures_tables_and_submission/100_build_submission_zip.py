# %% [markdown]
# # Build the Nature Communications submission archive

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import json
import zipfile
from pathlib import Path


NC = Path(
    '<external-path-redacted>'
    "ascertainment_framework_20260826/nature_communications"
)
PACKAGE = NC / "submission_package"
TEMP = NC / "Nature_Communications_submission_package_crossdb_v2.tmp.zip"
FINAL = NC / "Nature_Communications_submission_package.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


files = sorted(path for path in PACKAGE.iterdir() if path.is_file())
if TEMP.exists():
    TEMP.unlink()
with zipfile.ZipFile(TEMP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in files:
        archive.write(path, arcname=path.name)

with zipfile.ZipFile(TEMP) as archive:
    bad_member = archive.testzip()
    names = sorted(archive.namelist())
if bad_member is not None:
    raise RuntimeError(f"Archive integrity failure: {bad_member}")
if names != sorted(path.name for path in files):
    raise RuntimeError("Archive member list does not match submission package")

TEMP.replace(FINAL)
payload = {
    "status": "PASS",
    "archive": str(FINAL),
    "archive_bytes": FINAL.stat().st_size,
    "archive_sha256": sha256(FINAL),
    "file_count": len(files),
    "members": names,
    "patient_level_files": [
        name for name in names
        if any(token in name.lower() for token in ("patient", "prediction_secure", "subject_id"))
    ],
}
if payload["patient_level_files"]:
    raise RuntimeError("Patient-level-looking file name found in submission package")
(NC / "qa/SUBMISSION_ARCHIVE_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
