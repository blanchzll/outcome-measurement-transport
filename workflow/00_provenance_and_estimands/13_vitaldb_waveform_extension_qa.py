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
# # VitalDB waveform extension acceptance-gate audit
#
# This deterministic audit applies the criteria frozen before formal waveform
# extraction. It reports evidence and an integration category; it does not alter
# data, refit models, or choose a more favourable analysis.

# %%
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_FILES = 6394
EXPECTED_TEST_N = 324
EXPECTED_TEST_EVENTS = 46
EXPECTED_BASELINE_AUC = 0.7044103847356897


def gate(name: str, passed: bool, evidence: dict[str, object]) -> dict[str, object]:
    return {"gate": name, "passed": bool(passed), "evidence": evidence}


def calibration_robustness(stress: pd.DataFrame, model: str) -> dict[str, object]:
    subset = stress.loc[
        stress.model.eq(model)
        & stress.method.eq("recalibration_intercept_slope_truth")
        & stress.evaluation_target.eq("full")
    ].copy()
    targets = {"calibration_intercept": 0.0, "calibration_slope": 1.0}
    metrics: dict[str, object] = {}
    for metric, target in targets.items():
        values = pd.to_numeric(subset[metric], errors="coerce").dropna()
        deviation = values - target
        positive = float((deviation > 0).mean()) if len(deviation) else np.nan
        negative = float((deviation < 0).mean()) if len(deviation) else np.nan
        metrics[metric] = {
            "n_replicates": int(len(deviation)),
            "mean": float(values.mean()) if len(values) else None,
            "mean_deviation_from_ideal": float(deviation.mean()) if len(deviation) else None,
            "mean_absolute_deviation_from_ideal": float(deviation.abs().mean()) if len(deviation) else None,
            "directional_consistency": max(positive, negative) if len(deviation) else None,
            "dominant_direction": (
                "above_ideal" if positive >= negative else "below_ideal"
            ) if len(deviation) else None,
        }
    passed = any(
        item["n_replicates"] == 300
        and item["directional_consistency"] >= 0.80
        and item["mean_absolute_deviation_from_ideal"] >= 0.10
        for item in metrics.values()
    )
    return {"passed": bool(passed), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-verification", required=True, type=Path)
    parser.add_argument("--waveform-audit", required=True, type=Path)
    parser.add_argument("--model-audit", required=True, type=Path)
    parser.add_argument("--stress-audit", required=True, type=Path)
    parser.add_argument("--stress-replicates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest_verification.read_text(encoding="utf-8"))
    waveform = json.loads(args.waveform_audit.read_text(encoding="utf-8"))
    model = json.loads(args.model_audit.read_text(encoding="utf-8"))
    stress_audit = json.loads(args.stress_audit.read_text(encoding="utf-8"))
    stress = pd.read_csv(args.stress_replicates, low_memory=False)

    manifest_pass = (
        manifest.get("status") == "PASS"
        and manifest.get("manifest_entries") == EXPECTED_FILES
        and manifest.get("verified_files") == EXPECTED_FILES
        and not manifest.get("missing")
        and not manifest.get("mismatched")
    )
    cohort_pass = (
        model.get("test_n") == EXPECTED_TEST_N
        and model.get("test_events") == EXPECTED_TEST_EVENTS
        and abs(model.get("baseline_auc_reproduced", np.nan) - EXPECTED_BASELINE_AUC) <= 1e-6
        and "opend" in model.get("leakage_boundary", "")
    )
    coverage = float(model.get("test_usable_art_map_percent", np.nan))
    waveform_quality_pass = np.isfinite(coverage) and coverage >= 60.0
    auc_metrics = model.get("comparison_metrics", {}).get("point_metrics", {})
    enhanced_auc = float(auc_metrics.get("waveform_enhanced_ridge", {}).get("auc", np.nan))
    duration_adjusted_auc = float(
        auc_metrics.get("duration_adjusted_clinical_ridge", {}).get("auc", np.nan)
    )
    auc_delta = (
        model.get("comparison_metrics", {})
        .get("paired_deltas", {})
        .get("waveform_minus_duration_adjusted_clinical_paired_delta", {})
        .get("auc", {})
    )
    auc_delta_lower = float(auc_delta.get("ci_lower", np.nan))
    model_strength_pass = (
        np.isfinite(enhanced_auc)
        and enhanced_auc >= 0.70
        and np.isfinite(auc_delta_lower)
        and auc_delta_lower > -0.02
    )
    stress_contract_pass = (
        stress_audit.get("condition", {}).get("mechanism") == "mixed_MNAR"
        and stress_audit.get("condition", {}).get("target_measurement_retention") == 0.35
        and stress_audit.get("condition", {}).get("strength") == "strong"
        and stress_audit.get("condition", {}).get("replicates") == 300
        and stress_audit.get("heldout_n") == EXPECTED_TEST_N
        and stress_audit.get("heldout_events") == EXPECTED_TEST_EVENTS
    )
    robustness = calibration_robustness(stress, "waveform_enhanced_ridge")

    gates = [
        gate("full_official_manifest_integrity", manifest_pass, manifest),
        gate(
            "frozen_cohort_timing_and_baseline_reproduction",
            cohort_pass,
            {
                "test_n": model.get("test_n"),
                "test_events": model.get("test_events"),
                "baseline_auc_reproduced": model.get("baseline_auc_reproduced"),
                "expected_baseline_auc": EXPECTED_BASELINE_AUC,
                "leakage_boundary": model.get("leakage_boundary"),
            },
        ),
        gate(
            "waveform_quality",
            waveform_quality_pass,
            {
                "heldout_usable_art_map_percent": coverage,
                "minimum_percent": 60.0,
                "overall_waveform_audit": waveform,
            },
        ),
        gate(
            "prespecified_real_model_strength",
            model_strength_pass,
            {
                "waveform_model_auc": enhanced_auc,
                "duration_adjusted_clinical_auc": duration_adjusted_auc,
                "minimum_auc": 0.70,
                "paired_auc_delta_lower_95": auc_delta_lower,
                "minimum_noninferiority_bound": -0.02,
                "paired_auc_delta": auc_delta,
                "primary_comparator": "duration_adjusted_clinical_ridge",
            },
        ),
        gate("measurement_stress_contract", stress_contract_pass, stress_audit.get("condition", {})),
        gate("measurement_process_robustness", robustness["passed"], robustness),
    ]
    all_pass = all(item["passed"] for item in gates)
    integrity_timing_pass = manifest_pass and cohort_pass
    if all_pass:
        integration = "eligible_for_main_text_if_scientifically_material"
    elif integrity_timing_pass:
        integration = "supplementary_only"
    else:
        integration = "do_not_report_until_integrity_or_timing_failure_is_resolved"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_timing_pass else "FAIL",
        "all_prespecified_gates_passed": all_pass,
        "manuscript_integration": integration,
        "gates": gates,
        "interpretation_boundary": (
            "A passed extension supports robustness of the measurement-process thesis only; it is not "
            "prospective clinical impact evidence, strict source-model external validation, or causal "
            "evidence for intraoperative haemodynamics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not integrity_timing_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
