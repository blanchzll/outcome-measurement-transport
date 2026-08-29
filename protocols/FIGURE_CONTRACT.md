# Figure Contract

Core conclusion: A fixed clinical risk model can appear locally miscalibrated when
center-specific longitudinal outcome measurement changes endpoint observability; this
distortion is not reliably repaired by local recalibration, whereas a small complete
reference sample and explicit sensitivity bounds distinguish risk drift from measurement drift.

Figure archetype: quantitative grid delivered as separate panel PDFs in one folder per figure.

Target journal/output: high-impact clinical digital-health journal; Python-only drawing;
editable SVG, vector PDF, and 600-dpi TIFF; 180-mm full-width or 89-mm single-column panels.

Panel map:

- Figure 1a: source-to-reference cohort and endpoint observability counts.
- Figure 1b: observed versus IPW-balanced covariate standardized differences.
- Figure 1c: monitoring density versus operational event detection.
- Figure 2a: event-rate bias heatmap across deletion mechanisms and retention levels.
- Figure 2b: O/E bias heatmap across correction strategies.
- Figure 2c: reconstruction sensitivity versus measurement retention in both databases.
- Figure 3a: apparent versus full-reference calibration after local recalibration.
- Figure 3b: bias/RMSE comparison for IPAW, AIPW, local recalibration, reference-sample calibration and MNAR bounds.
- Figure 3c: independent INSPIRE/MIMIC replication concordance.
- Figure 4a: 200-refit risk-ranking and top-20% stability.
- Figure 4b: preoperative-to-perioperative incremental value.
- Figure 4c: portability/performance frontier.
- Figure 5a: monitoring burden versus event capture.
- Figure 5b: subgroup discrimination/calibration representativeness audit.

Evidence hierarchy:

- hero evidence: two-database complete-reference deletion experiments;
- validation evidence: independent database replication and reference-sample correction;
- controls/robustness: weight diagnostics, MNAR envelopes, refit stability, subgroup audit.

Statistics needed: 300 Monte Carlo replicates/condition; mean, Monte Carlo SD, 2.5th/97.5th
percentiles, bias, RMSE, ESS and coverage; 1000 center/outcome-stratified bootstrap intervals;
200 center/outcome-stratified refits.

Source data needed: aggregate CSV for every panel; no patient-level source data in figure folders.

Image-integrity notes: numerical plots only; no image enhancement, cropping, AI-generated
quantitative panels or manual point removal.

Reviewer risks: operational creatinine reference is not full KDIGO or blinded adjudication;
MIMIC landmark differs from surgery end; only five source centers support hierarchical inference;
utility and fairness panels are scenario/representativeness audits, not clinical-impact or fairness certification.
