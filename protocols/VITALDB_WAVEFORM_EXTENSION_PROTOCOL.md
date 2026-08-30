# Prespecified VitalDB waveform extension protocol

Date frozen: 30 August 2026, before complete waveform extraction.

## Scientific role

This extension tests whether the apparent-versus-retained calibration divergence
persists when the fixed risk score contains high-resolution intraoperative
physiology. It is not an algorithm search and is not intended to turn the source
AKI model into a deployment-ready product.

## Primary question

Among the existing VitalDB adult, single-operation dense-reference cohort, does
transporting or deleting postoperative creatinine measurements produce the same
direction of apparent-versus-retained calibration divergence for a waveform-
enhanced fixed ridge score as for the existing clinical-table ridge score?

## Cohorts and landmark

- Dataset: VitalDB 1.0.0, with all files verified against the official
  `SHA256SUMS.txt` manifest before analysis.
- Prediction landmark: recorded operation end (`opend`).
- Primary cohort: adults with exactly one recorded operation and the existing
  dense creatinine reference.
- Secondary cohort: adults with exactly one operation and an observable 0-168 h
  creatinine reference.
- The original patient-disjoint 70/30 split and seed 20260830 will be reused.
  No outcome from the held-out set will inform feature or model selection.

## Waveform feature extraction

Only measurements between `opstart` and `opend` are eligible. Values outside the
following physiological ranges are set missing before summary:

- mean arterial pressure (MAP): 20-200 mmHg;
- heart rate: 20-250 beats per minute.

Arterial MAP is selected in the following frozen order when available:
`Solar8000/ART_MBP`, `EV1000/ART_MBP`, then `Solar8000/FEM_MBP`. Non-invasive
`Solar8000/NIBP_MBP` is summarized separately and is not carried forward as if it
were continuously observed. Heart rate uses `Solar8000/HR`, with
`Solar8000/PLETH_HR` as a labelled fallback.

For time-weighted arterial-MAP and heart-rate summaries, a valid monitor value is
carried forward for at most five seconds; longer gaps remain missing. Raw sample
counts and the resulting covered seconds are both retained. This rule is fixed
before extraction and does not use the outcome.

Prespecified features are:

- valid arterial-MAP observation minutes and proportion of operative duration;
- median, 5th percentile and standard deviation of arterial MAP;
- minutes and proportion of observed arterial MAP below 65, 60 and 55 mmHg;
- time-weighted average deficit and area below 65 mmHg;
- median and standard deviation of heart rate;
- observed proportion with heart rate above 100 beats per minute;
- count and median of non-invasive MAP measurements.

Hypotension-duration features are treated as missing when valid arterial MAP
covers less than 30 minutes or less than 20% of operative duration. Coverage is
retained as a predictor and reported by outcome and split. No waveform feature is
derived after `opend`.

## Models and evaluation

Two fixed ridge logistic models will be compared:

1. the existing VitalDB clinical-table feature set;
2. the same feature set plus the prespecified waveform summaries.

Preprocessing remains fold-contained. There is no broad model comparison,
stacking, feature selection against held-out performance or hyperparameter search.
Held-out AUC, Brier score, O/E, calibration intercept and slope are reported with
patient bootstrap intervals. Incremental performance is descriptive and is not a
claim of clinical benefit.

## Outcome-measurement stress test

The primary condition is strong mixed-MNAR deletion at 35% target
per-measurement retention with 300 replicates. The existing reconstructed and
retained operational endpoints, cross-fitted apparent-target recalibration,
IPAW/AIPW, reference sampling and sensitivity-region definitions are unchanged.
The empirical schedule-transport experiment is repeated for both fixed scores.

The primary contrast is the retained-reference O/E after apparent-target local
recalibration. The relevant conclusion is whether its direction and material
separation from apparent O/E persist, not whether one risk model has the largest
AUC.

## Decision rules and reporting boundary

- If fewer than 300 held-out records, fewer than 40 held-out retained-reference
  events, or less than 60% usable arterial-MAP coverage remain, the waveform
  comparison is labelled exploratory feasibility evidence.
- The waveform result enters the main text only if it changes the central
  inference or materially strengthens the real-model generality claim; otherwise
  it remains supplementary.
- The existing clinical/laboratory analyses are rerun only if an official core
  file hash changes or shared analysis code changes. Downloading waveforms alone
  is not a reason to rerun unchanged core analyses.
- No result may be described as full KDIGO adjudication, a causal effect of
  hypotension, clinical utility, prospective impact or proof of deployment
  readiness.
