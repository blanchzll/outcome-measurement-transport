# VitalDB waveform extension: frozen acceptance and reporting gates

Frozen on 2026-08-30 before complete waveform extraction and before any formal
waveform-model result was available. The one-replicate, case-ID-only smoke test
was used only to verify interfaces and exact reconstruction of the previously
reported clinical-table ridge model; it was not used to set performance gates.

## Data-integrity gate

- The official VitalDB 1.0.0 `SHA256SUMS.txt` must contain 6,394 objects.
- Every object must exist and independently match its official SHA-256 digest.
- Waveform extraction must stop if the full-manifest result is not `PASS`.
- Patient-level waveform features and predictions remain in the mode-700 secure
  workspace and are not included in the public repository.

## Cohort and timing gate

- Preserve the existing adult, dense-creatinine-reference, single-operation
  cohort and the patient-disjoint 70/30 split with seed 20260830.
- The held-out set must remain 324 patients and 46 creatinine-reference events.
- The baseline clinical-table ridge AUC must reproduce exactly within 1e-6 of
  0.7044103847356897.
- All waveform summaries end at `opend`; the retained creatinine outcome begins
  after `opend`. No held-out outcome may inform feature selection, preprocessing,
  tuning, or fitting.
- Report the number and percentage of operation windows truncated by waveform
  record boundaries and the minimum available-window fraction. Do not use these
  quantities for post hoc patient exclusion or model selection.

## Waveform-quality gate

- Use only prespecified tracks, physiological ranges, five-second maximum carry
  forward, and operation-window definitions in the protocol.
- Duration-dependent arterial-pressure summaries are usable only with at least
  30 minutes and at least 20% operation-window coverage.
- If fewer than 60% of held-out patients have usable arterial-pressure duration
  features, the waveform analysis is labelled exploratory and cannot support a
  main-text claim of incremental physiological information.

## Model-comparison gate

- Compare only the frozen historical clinical-table ridge, a clinical ridge with
  operation duration, and the duration-adjusted ridge with prespecified waveform
  summaries.
- Report AUC, Brier score, O/E, calibration intercept, and calibration slope with
  1,000 patient-level paired-bootstrap intervals.
- The primary paired difference is waveform minus duration-adjusted clinical.
  Waveform minus historical clinical is secondary continuity evidence. Do not
  infer superiority from overlapping or non-overlapping marginal intervals.
- The waveform model may be described as a stronger real risk engine only when
  its held-out AUC is at least 0.70 and the lower 95% paired-bootstrap bound for
  its AUC difference from the duration-adjusted comparator is greater than
  -0.02. Otherwise it is a descriptive sensitivity model, not evidence of
  predictive improvement.

## Measurement-process robustness gate

- Primary condition: strong mixed-MNAR deletion, target measurement retention
  35%, 300 independently seeded replicates.
- The central finding is considered robust to the waveform model only if the
  apparent-versus-full-reference calibration bias has the same prespecified
  direction in at least 80% of replicates and its mean absolute bias is at least
  0.10 for either calibration intercept or calibration slope.
- IPAW/AIPW results must be interpreted against their identifying assumptions;
  neither may be called a recovery of truth under outcome-dependent MNAR.
- Apparent-outcome recalibration is evaluated both against the reconstructed
  outcome and the retained full reference. Improvement against the former does
  not establish improvement against the latter.

## Manuscript-integration gate

- Enter the main Results only if data integrity, cohort/timing, waveform quality,
  model comparison, and robustness gates all pass and the result materially
  strengthens the central measurement-process thesis.
- If integrity and timing pass but waveform quality or model-strength gates fail,
  report the extension transparently in the Supplementary Information and retain
  the existing main conclusion.
- Rerun the existing INSPIRE, MIMIC-IV, eICU-CRD, source-cohort, tables, and figures
  only if shared code/data hashes changed or the VitalDB result changes a shared
  estimand. Otherwise perform hash and regression checks rather than redundant
  recomputation.
- No VitalDB result is evidence of prospective clinical impact, expert-adjudicated
  AKI transportability, or a causal effect of intraoperative hypotension.
