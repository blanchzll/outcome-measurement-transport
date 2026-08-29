# %% [markdown]
# # Manuscript release audit
#
# Aggregate-only checks for word limits, evidence-role boundaries, corrected simulation
# language, numerical anchors, and unresolved author-controlled fields.

# %%
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript" / "MANUSCRIPT_FINAL.md"
FACT_BASE = ROOT / "outputs" / "PUBLICATION_FACT_BASE.json"
OUTPUT = ROOT / "outputs" / "MANUSCRIPT_RELEASE_AUDIT.json"

text = MANUSCRIPT.read_text(encoding="utf-8")
lower = text.lower()
facts = json.loads(FACT_BASE.read_text(encoding="utf-8"))


def section_between(start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def fact_row(database: str, method: str, metric: str, retention: float = 0.35) -> dict:
    rows = facts["apparent_vs_reference_recalibration"]
    matches = [
        row for row in rows
        if row["database"] == database and row["method"] == method
        and row["metric"] == metric and abs(float(row["retention_target"]) - retention) < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError((database, method, metric, retention, len(matches)))
    return matches[0]


summary = section_between("## Summary", "## Research in context")
article = section_between("## Introduction", "## Contributors")
title = text.splitlines()[0].lstrip("# ").strip()

dynamic_anchors = {
    "source_n": str(facts["source_primary"]["n"]),
    "source_events": str(facts["source_primary"]["events"]),
    "source_auc": f'{facts["source_pi_restricted_rf"]["roc_auc"]:.3f}',
    "inspire_recalibration_apparent_oe": f'{fact_row("INSPIRE", "recalibration_intercept_slope_apparent", "oe")["mean"]:.3f}',
    "inspire_recalibration_reference_oe": f'{fact_row("INSPIRE", "recalibration_intercept_slope_truth", "oe")["mean"]:.3f}',
    "mimic_recalibration_apparent_oe": f'{fact_row("MIMIC", "recalibration_intercept_slope_apparent", "oe")["mean"]:.3f}',
    "mimic_recalibration_reference_oe": f'{fact_row("MIMIC", "recalibration_intercept_slope_truth", "oe")["mean"]:.3f}',
    "eicu_recalibration_apparent_oe": f'{fact_row("EICU", "recalibration_intercept_slope_apparent", "oe")["mean"]:.3f}',
    "eicu_recalibration_reference_oe": f'{fact_row("EICU", "recalibration_intercept_slope_truth", "oe")["mean"]:.3f}',
    "monte_carlo_replicates": "300",
    "master_seed": "20260826",
}

overclaim_patterns = {
    "gold_standard_assertion": r"(?:is|was|provides?) (?:an? |the )?gold standard",
    "full_kdigo_public_assertion": r"public (?:database )?.{0,40}full kdigo",
    "practice_changing": r"practice[- ]changing",
    "deployment_ready": r"deployment[- ]ready",
    "mimic_source_external_validation": r"external validation of the source model.{0,30}mimic",
    "clinical_benefit_claim": r"(?:improved|reduced) patient outcomes",
}

required_phrases = {
    "retained_reference_target": "retained operational reference",
    "cross_fitted_recalibration": "two-fold cross-fitted",
    "positive_control": "pure label-selection",
    "endpoint_component_audit": "component-availability audit",
    "icu_replication_role": "both icu databases used admission landmarks and database-native models",
    "clinical_impact_boundary": "not clinical impact",
    "source_kdigo": "2012 kdigo",
    "source_two_nephrologists": "two nephrologists",
    "source_masking": "masked to model predictions and candidate predictors",
    "source_third_reviewer": "third nephrologist at the coordinating centre",
    "source_not_central": "rather than by a single central panel",
    "eicu_unseen_hospital_role": "unseen-hospital computational replication",
    "three_database_claim": "across all three public databases",
    "retrospective_nonregistration": "not prospectively registered",
    "harmonized_transport_boundary": "not external validation of the source model",
    "record_bootstrap_unit": "analytic-record bootstrap",
}

unresolved_author_fields = sorted(set(re.findall(r"\[[A-Z][^\]]+\]", text)))
checks = {
    "summary_words_le_300": len(summary.split()) <= 300,
    "article_words_le_3500": len(article.split()) <= 3500,
    "title_does_not_use_paradox": "paradox" not in title.lower(),
    "no_stale_false_alert_language": "false alerts" not in lower,
    "no_stale_in_sample_recalibration_claim": "same-sample recalibration" not in lower,
    "no_draft_result_placeholders": not any(
        marker in text for marker in ("[INSERT", "PRODUCTION SIMULATION RESULTS", "TO BE COMPLETED")
    ),
    "all_required_phrases_present": all(phrase in lower for phrase in required_phrases.values()),
    "all_dynamic_anchors_present": all(value in text for value in dynamic_anchors.values()),
    "overclaim_patterns_absent": not any(re.search(pattern, lower) for pattern in overclaim_patterns.values()),
    "references_include_doi_or_persistent_url": len(re.findall(r"(?:doi:10\.|https://)", text)) >= 15,
}

analysis_passed = all(checks.values())
payload = {
    "manuscript": str(MANUSCRIPT),
    "title": title,
    "summary_word_count": len(summary.split()),
    "article_introduction_through_discussion_word_count": len(article.split()),
    "persistent_reference_identifiers": len(re.findall(r"(?:doi:10\.|https://)", text)),
    "required_phrases": required_phrases,
    "dynamic_numerical_anchors": dynamic_anchors,
    "unresolved_author_fields": unresolved_author_fields,
    "analysis_release_checks": checks,
    "analysis_release_passed": analysis_passed,
    "submission_ready": analysis_passed and not unresolved_author_fields,
    "patient_level_data_read": False,
}

OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, ensure_ascii=False))
if not analysis_passed:
    raise SystemExit(1)
