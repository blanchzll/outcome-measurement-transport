# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: '.py'
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # VitalDB extension: core-analysis rerun decision
#
# Compare the release baseline with the current revision and distinguish
# VitalDB-specific extension files from shared scientific code. The output is a
# machine-readable decision ledger; it does not run or alter an analysis.

# %%
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


VITALDB_SPECIFIC_PATHS = {
    "protocols/VITALDB_WAVEFORM_EXTENSION_ACCEPTANCE_GATES.md",
    "protocols/VITALDB_WAVEFORM_EXTENSION_PROTOCOL.md",
    "tests/test_vitaldb_waveform_extension.py",
    "workflow/00_provenance_and_estimands/13_vitaldb_waveform_extension_qa.py",
    "workflow/00_provenance_and_estimands/16_vitaldb_waveform_result_digest.py",
    "workflow/00_provenance_and_estimands/17_vitaldb_core_rerun_decision.py",
    "workflow/02_public_reference_cohorts/10_vitaldb_waveform_feature_extraction.py",
    "workflow/04_measurement_deletion_simulation/11_vitaldb_waveform_model_comparison.py",
    "workflow/04_measurement_deletion_simulation/12_vitaldb_waveform_measurement_stress.py",
    "workflow/07_figures_tables_and_submission/15_plot_vitaldb_waveform_extension.py",
}
RELEASE_METADATA_PATHS = {
    "README.md",
    "WORKFLOW_MANIFEST.csv",
    "RELEASE_MANIFEST.json",
    "requirements-full.txt",
}


def run_git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.STDOUT
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_changed_paths(changed: list[str]) -> dict[str, list[str]]:
    vitaldb = sorted(path for path in changed if path in VITALDB_SPECIFIC_PATHS)
    metadata = sorted(path for path in changed if path in RELEASE_METADATA_PATHS)
    shared = sorted(path for path in changed if path not in VITALDB_SPECIFIC_PATHS | RELEASE_METADATA_PATHS)
    return {
        "vitaldb_specific": vitaldb,
        "release_metadata": metadata,
        "shared_scientific_or_unclassified": shared,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--baseline", default="v1.3.0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo.resolve()
    baseline_commit = run_git(repo, "rev-parse", args.baseline)
    head_commit = run_git(repo, "rev-parse", "HEAD")
    changed = sorted(
        line for line in run_git(repo, "diff", "--name-only", f"{args.baseline}..HEAD").splitlines()
        if line
    )
    classification = classify_changed_paths(changed)
    shared = classification["shared_scientific_or_unclassified"]
    status_lines = run_git(repo, "status", "--porcelain").splitlines()
    tracked_hashes = {
        path: sha256(repo / path)
        for path in changed
        if (repo / path).is_file()
    }
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "repository": repo.name,
        "baseline_ref": args.baseline,
        "baseline_commit": baseline_commit,
        "head_commit": head_commit,
        "worktree_clean": not status_lines,
        "changed_paths": changed,
        "classification": classification,
        "changed_file_sha256": tracked_hashes,
        "core_analysis_rerun_required": bool(shared),
        "decision": (
            "Rerun every database affected by the listed shared scientific code changes."
            if shared
            else "Do not rerun unaffected core databases; run regression and hash audits, then integrate the VitalDB-specific extension."
        ),
        "boundary": (
            "This code-dependency decision does not determine whether the formal VitalDB result belongs in the main text, supplementary information, or is non-reportable."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
