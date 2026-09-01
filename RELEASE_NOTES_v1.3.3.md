# Release notes: v1.3.3

This release closes the final cross-file reproducibility finding from blind pre-submission review.

## Workbook authority and semantic verification

- Makes the tagged aggregate CSV tables the final authority when submission workbooks are rebuilt, preventing an older delivery snapshot from repopulating corrected source-cohort results.
- Rebuilds the Supplementary Tables workbook and confirms semantic agreement with all 33 matching tagged release tables, including temporal validation, fixed-geography validation, observation restriction, preoperative-AKI sensitivity and the public-model clinical bridge.
- Adds an executable workbook-to-release semantic audit with strict column, row, string and numeric comparisons.

## Presentation

- Replaces overlapping model labels in Supplementary Figure 15 with compact two-line labels and rechecks the combined Supplementary Information PDF.

No statistical estimand, patient-level result or scientific claim changed relative to v1.3.2.
