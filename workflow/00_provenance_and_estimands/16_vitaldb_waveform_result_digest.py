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
# # VitalDB waveform extension result digest
#
# Build a machine-readable numerical ledger and a concise manuscript drafting
# aid from the audited aggregate outputs. This script never edits the manuscript.

# %%
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


MODEL_LABELS = {
    "clinical_table_ridge": "historical clinical-table ridge",
    "duration_adjusted_clinical_ridge": "duration-adjusted clinical ridge",
    "waveform_enhanced_ridge": "clinical-plus-waveform ridge",
}
DELTA_LABELS = {
    "waveform_minus_duration_adjusted_clinical_paired_delta": (
        "waveform minus duration-adjusted clinical ridge"
    ),
    "waveform_minus_historical_clinical_paired_delta": (
        "waveform minus historical clinical-table ridge"
    ),
}


def interval(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n_replicates": int(len(x)),
        "mean": float(x.mean()),
        "q025": float(x.quantile(0.025)),
        "q975": float(x.quantile(0.975)),
    }


def format_interval(item: dict[str, float | int], digits: int = 3) -> str:
    return (
        f"{item['mean']:.{digits}f} "
        f"({item['q025']:.{digits}f}-{item['q975']:.{digits}f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waveform-audit", required=True, type=Path)
    parser.add_argument("--model-table", required=True, type=Path)
    parser.add_argument("--model-audit", required=True, type=Path)
    parser.add_argument("--stress-replicates", required=True, type=Path)
    parser.add_argument("--qa-audit", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args()

    waveform = json.loads(args.waveform_audit.read_text(encoding="utf-8"))
    model_audit = json.loads(args.model_audit.read_text(encoding="utf-8"))
    qa = json.loads(args.qa_audit.read_text(encoding="utf-8"))
    if qa.get("status") != "PASS":
        raise RuntimeError("Extension integrity/timing QA must pass before result digestion")
    model_table = pd.read_csv(args.model_table)
    stress = pd.read_csv(args.stress_replicates, low_memory=False)

    performance: dict[str, object] = {}
    for model in MODEL_LABELS:
        subset = model_table.loc[
            model_table.comparison.eq("model_performance") & model_table.model.eq(model)
        ]
        performance[model] = {
            row.metric: {
                "estimate": float(row.estimate),
                "ci_lower": float(row.ci_lower),
                "ci_upper": float(row.ci_upper),
                "bootstrap_replicates": int(row.bootstrap_replicates),
            }
            for row in subset.itertuples(index=False)
        }
    paired_deltas = {}
    for comparison, label in DELTA_LABELS.items():
        delta = model_table.loc[model_table.comparison.eq(comparison)]
        paired_deltas[comparison] = {
            "label": label,
            "metrics": {
                row.metric: {
                    "estimate": float(row.estimate),
                    "ci_lower": float(row.ci_lower),
                    "ci_upper": float(row.ci_upper),
                    "bootstrap_replicates": int(row.bootstrap_replicates),
                }
                for row in delta.itertuples(index=False)
            },
        }

    stress_digest: dict[str, object] = {}
    methods = {
        "apparent_recalibration": (
            "recalibration_intercept_slope_apparent",
            "reconstructed",
        ),
        "retained_reference_after_apparent_recalibration": (
            "recalibration_intercept_slope_truth",
            "full",
        ),
    }
    for model in MODEL_LABELS:
        stress_digest[model] = {}
        for label, (method, target) in methods.items():
            subset = stress.loc[
                stress.model.eq(model)
                & stress.method.eq(method)
                & stress.evaluation_target.eq(target)
            ]
            stress_digest[model][label] = {
                metric: interval(subset[metric])
                for metric in ("oe", "calibration_intercept", "calibration_slope")
            }
        reference = stress.loc[
            stress.model.eq(model)
            & stress.method.eq("full_reference")
            & stress.evaluation_target.eq("full")
        ]
        stress_digest[model]["measurement_process"] = {
            metric: interval(reference[metric])
            for metric in ("outcome_observed_fraction", "reconstructed_sensitivity")
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "VitalDB 1.0.0",
        "analysis_role": "prespecified waveform robustness extension",
        "manuscript_integration": qa.get("manuscript_integration"),
        "cohort": {
            "eligible_n": model_audit.get("eligible_n"),
            "eligible_events": model_audit.get("eligible_events"),
            "heldout_n": model_audit.get("test_n"),
            "heldout_events": model_audit.get("test_events"),
        },
        "waveform_quality": {
            "overall_usable_art_map_n": waveform.get("n_usable_art_map_duration_features"),
            "overall_usable_art_map_percent": waveform.get("usable_art_map_percent"),
            "heldout_usable_art_map_n": model_audit.get("test_usable_art_map_n"),
            "heldout_usable_art_map_percent": model_audit.get("test_usable_art_map_percent"),
            "operation_windows_truncated_n": waveform.get("n_operation_windows_truncated"),
            "operation_windows_truncated_percent": waveform.get("operation_window_truncated_percent"),
            "operation_window_available_fraction_minimum": waveform.get(
                "operation_window_available_fraction_minimum"
            ),
            "art_map_track_counts": waveform.get("art_map_track_counts"),
        },
        "model_performance": performance,
        "paired_waveform_deltas": paired_deltas,
        "measurement_stress": stress_digest,
        "qa_gate_status": {
            item["gate"]: item["passed"] for item in qa.get("gates", [])
        },
        "interpretation_boundary": qa.get("interpretation_boundary"),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    waveform_auc = performance["waveform_enhanced_ridge"]["auc"]
    auc_delta = paired_deltas[
        "waveform_minus_duration_adjusted_clinical_paired_delta"
    ]["metrics"]["auc"]
    waveform_stress = stress_digest["waveform_enhanced_ridge"]
    lines = [
        "# VitalDB waveform extension: audited result drafting aid",
        "",
        f"Reporting category: `{qa.get('manuscript_integration')}`.",
        "",
        "## Results sentence",
        "",
        (
            f"In the frozen VitalDB held-out set ({model_audit.get('test_n')} operations; "
            f"{model_audit.get('test_events')} creatinine-reference events), the prespecified "
            f"clinical-plus-waveform ridge model had an AUC of {waveform_auc['estimate']:.3f} "
            f"(95% CI {waveform_auc['ci_lower']:.3f}-{waveform_auc['ci_upper']:.3f}); the paired "
            f"difference from the duration-adjusted clinical ridge was {auc_delta['estimate']:+.3f} "
            f"({auc_delta['ci_lower']:+.3f} to {auc_delta['ci_upper']:+.3f})."
        ),
        (
            "After updating against the reconstructed endpoint under the prespecified strong "
            "mixed-MNAR 35% retention condition, apparent O/E was "
            f"{format_interval(waveform_stress['apparent_recalibration']['oe'])}, whereas O/E "
            "against the retained reference for the same updated probabilities was "
            f"{format_interval(waveform_stress['retained_reference_after_apparent_recalibration']['oe'])}."
        ),
        "",
        "## Required boundary sentence",
        "",
        str(qa.get("interpretation_boundary")),
        "",
        "All numbers above are generated from audited aggregate outputs; edit prose for clarity, not values.",
    ]
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "json": str(args.json_output), "markdown": str(args.markdown_output)}, indent=2))


if __name__ == "__main__":
    main()
