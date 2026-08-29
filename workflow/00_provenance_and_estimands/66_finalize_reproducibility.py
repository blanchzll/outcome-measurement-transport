# %% [markdown]
# # Reproducibility manifest
# Records the runtime, core package versions and SHA-256 hashes of analysis artifacts.

# %%
from __future__ import annotations
from release_paths import release_path as _release_path

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path

ROOT = Path(str(_release_path('analysis')))
OUT = ROOT / "outputs"
REPRO = ROOT / "delivery" / "reproducibility"
PACKAGE = ROOT / "package" / "ascertainment-stress-test"
REPRO.mkdir(parents=True, exist_ok=True)

# Keep the release package synchronised with the audited analysis core. The dedicated
# package regression test is maintained beside this script and copied during finalisation.
shutil.copy2(ROOT / "code" / "ascertainment_stress.py", PACKAGE / "ascertainment_stress.py")
contract_test = ROOT / "code" / "test_simulation_contract.py"
if contract_test.exists():
    shutil.copy2(contract_test, PACKAGE / "tests" / "test_simulation_contract.py")

packages = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "statsmodels", "duckdb", "pyarrow", "joblib"]
versions = {}
for name in packages:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None

(REPRO / "requirements-core.txt").write_text(
    "".join(f"{name}=={version}\n" for name, version in versions.items() if version is not None)
)

include_roots = [ROOT / "code", ROOT / "protocol", ROOT / "manuscript", ROOT / "package",
                 ROOT / "tables", ROOT / "figures", ROOT / "outputs",
                 ROOT / "eicu" / "code", ROOT / "eicu" / "tables", ROOT / "eicu" / "outputs"]
exclude_parts = {"__pycache__", ".pytest_cache"}
manifest = []
skipped_unreadable: list[str] = []
for base in include_roots:
    if not base.exists():
        continue
    discovered: list[Path] = []
    for directory, directory_names, file_names in os.walk(base):
        directory_names[:] = sorted(name for name in directory_names if name not in exclude_parts)
        discovered.extend(Path(directory) / name for name in sorted(file_names))
    for path in sorted(discovered):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except PermissionError:
            # Cache paths are pruned above; any other unreadable generated file is recorded
            # separately rather than aborting the aggregate release manifest.
            skipped_unreadable.append(str(path.relative_to(ROOT)))
            continue
        manifest.append((digest, str(path.relative_to(ROOT))))
(REPRO / "MANIFEST_SHA256.txt").write_text("".join(f"{d}  {p}\n" for d, p in manifest))

audit = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "packages": versions,
    "manifest_files": len(manifest),
    "patient_level_files_in_manifest": [p for _, p in manifest if p.startswith(("secure_work/", "eicu/secure/"))],
    "skipped_unreadable_files": skipped_unreadable,
    "remote_workspace": str(ROOT),
    "release_core_matches_analysis_core": hashlib.sha256(
        (PACKAGE / "ascertainment_stress.py").read_bytes()
    ).hexdigest() == hashlib.sha256((ROOT / "code" / "ascertainment_stress.py").read_bytes()).hexdigest(),
}
(OUT / "REPRODUCIBILITY_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
