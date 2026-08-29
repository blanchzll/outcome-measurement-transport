# Statistical analysis and simulation protocol

Version: 1.1, protocol amendment dated 2026-08-27 after independent code-to-protocol audit.

This is a retrospective, non-preregistered enhancement analysis. Version 1.0 was written after the existing source/INSPIRE cohort and endpoint audits were known but before the new analyses. Version 1.1 prospectively records corrections triggered by independent review: the 48–96 h second-window implementation, sequential observed-history deletion, untruncated and truncated design-probability IPAW, held-out cross-fitted recalibration, multi-fraction reference sampling, a pure label-selection positive control, and a public KDIGO-component audit. All affected simulations were rerun; no old and corrected result was selected on statistical favourability.

## Scientific objective

Estimate how incomplete, risk-dependent, and institution-dependent postoperative creatinine observation changes apparent calibration and decision utility of locked clinical risk predictions, and determine when complete-case evaluation, inverse-probability methods, local recalibration, and sensitivity bounds recover or distort performance relative to a densely observed longitudinal creatinine reference.

## Notation and estimands

- `Y_full`: operational longitudinal creatinine AKI reference derived from all available postoperative creatinine values within 0–168 hours. It is not biological truth, urine-output KDIGO, or clinician adjudication.
- `M_i(t)`: whether patient `i` has a retained/observed creatinine measurement at postoperative time `t`.
- `R_i`: whether the observed record contains all measurements required by the target two-slot endpoint.
- `Y_obs`: two-slot AKI label reconstructed from retained measurements.
- `p_i`: model probability frozen before evaluation.
- Primary performance estimands relative to `Y_full`: calibration-in-the-large, calibration slope, O/E, Brier score, and decision-curve net benefit. AUROC and average precision are secondary.
- Primary observability estimands in the 7,135-operation INSPIRE candidate cohort: reference observability fraction, standardized differences between observable and non-observable patients, stabilized inverse-observation weights, effective sample size, IPW/AIPW event rate, Brier score, O/E, risk-bin calibration, and net benefit.

The independent Monte Carlo unit is one seeded measurement-deletion replicate. Patients are sampling units within a replicate and are never counted as independent simulation replicates.

## Analysis populations

### Source clinical cohort

- Authoritative 4,014-operation table.
- Primary model-development and LOCO population: all 4,014 operations with a recorded site-level nephrologist-adjudicated KDIGO 2012 outcome, including 155 events.
- Five centres are retained as clusters for LOCO and hierarchical/partial-pooling calibration.

### INSPIRE 1.4.2

- Candidate cohort: 7,135 first qualifying gastric or colorectal cancer gastrointestinal operations.
- Two-slot observable cohort: baseline plus at least one creatinine in >0–48 h and >48–96 h.
- Longitudinal reference cohort: baseline plus at least one creatinine in >0–168 h.
- Dense simulation cohort: baseline, at least three postoperative creatinine measurements, at least one in >0–48 h and one in >48–96 h, and postoperative measurement span of at least 72 h. If this leaves fewer than 100 events, the span requirement is relaxed but reported.
- Primary locked probability: source-developed restricted random forest. Ridge and gradient boosting are algorithm-sensitivity comparators.

### MIMIC-IV 3.1

- Independent methodological test bed, not external validation of the gastric/colorectal model.
- Adults, first ICU stay per hospitalization, surgical service active at ICU admission, and baseline creatinine available in the preceding 30 days.
- Longitudinal landmark: ICU admission.
- `Y_full`: creatinine increase >=0.3 mg/dL within 48 h or ratio >=1.5 within 168 h relative to latest pre-landmark baseline.
- Dense simulation cohort uses the same measurement-density rule as INSPIRE.
- A database-specific ridge probability is trained on the earlier 60% of admissions by calendar time and locked before evaluation in the later 40%. It is a measurement-bias test instrument, not a proposed clinical model.

## Outcome-observability analysis

Cross-fitted propensity models estimate `Pr(R=1|X)` from prediction-time variables only. Primary propensity model: regularized logistic regression with nonlinear continuous terms; gradient boosting is a sensitivity model. Five folds are stratified by `R`; folds are kept fixed.

Report:

- propensity and stabilized-weight quantiles;
- positivity plots;
- weight truncation at none, 0.5/99.5%, 1/99%, and 2.5/97.5%;
- effective sample size `(sum w)^2/sum(w^2)`;
- observed versus non-observed standardized differences before and after weighting;
- complete-case, IPW, outcome-regression, and AIPW estimates for event rate, Brier score, O/E, risk-bin calibration, and net benefit.

IPW is not assumed to correct outcome misclassification. AIPW is considered doubly robust only for the specified missing-outcome estimand under MAR and positivity, not under latent-outcome-dependent measurement.

## MNAR and partial-identification analysis

For patients without the target outcome, shift outcome odds from the observed-data outcome model using sensitivity odds ratios `Gamma = 1/3, 1/2, 2/3, 1, 1.5, 2, 3`. Report the resulting event-rate, O/E, Brier and net-benefit envelope. Also report nonparametric worst-case prevalence bounds. The primary conclusion is the range of conclusions compatible with plausible `Gamma`, not a single corrected estimate.

## Measurement-deletion simulation

Full factorial factors:

- database: INSPIRE, MIMIC;
- target marginal measurement retention: 0.35, 0.55, 0.75;
- mechanism: `MCAR`, `stratum_MAR`, `risk_MAR`, `history_MAR`, `outcome_MNAR`, `mixed_MNAR`;
- dependence strength: weak, strong;
- Monte Carlo repetitions: 300 per scenario;
- master seed: 20260826; scenario and replicate seeds are deterministically derived.

Postoperative measurements are sampled conditional on the mechanism, with intercepts numerically calibrated to the target retention. The baseline measurement is never deleted. MCAR, stratum-MAR, risk-MAR and outcome-MNAR use conditionally independent measurement draws. History-MAR and mixed-MNAR are generated sequentially: each history term uses the most recent retained creatinine, never a hidden value from the complete trajectory. For these adaptive mechanisms, the recorded pattern probability is a conditional realised-history approximation and is not described as a marginal oracle probability. The model probability and patient cohort never change within a replicate.

Each retained record is converted to the same two-slot label. `R=1` requires both postoperative slots. The following strategies are compared:

1. retained-reference performance (`Y_full`; simulation benchmark);
2. naive complete-case evaluation against `Y_obs`;
3. untruncated design-probability IPAW and a 99th-percentile-truncated sensitivity analysis;
4. augmented design-probability IPAW for estimands linear in the outcome;
5. two-fold cross-fitted intercept-only local recalibration to `Y_obs`;
6. two-fold cross-fitted intercept-plus-slope local recalibration to `Y_obs`;
7. randomly selected 5%, 10%, 20% and 30% reference samples, each evaluated only in held-out patients;
8. a Gamma=2 prediction-based sensitivity region, not a formal nonparametric identification bound.

Apparent and reference-target recalibration use the same out-of-fold predictions. No recalibration performance is evaluated on the labels used to estimate that update.

## Pure label-selection positive control

In a separate experiment, complete patient-level labels are hidden without deleting measurements or altering observed label values. MCAR, risk-MAR, stratum-MAR, outcome-MNAR and mixed-MNAR mechanisms use the same retention targets and dependence strengths. Naive evaluation is compared with untruncated IPW using the exact simulated selection probability. Under outcome-dependent selection this oracle probability uses the otherwise unobserved outcome and is an identification benchmark, not an implementable estimator. This experiment tests whether residual bias in the longitudinal experiment arises from selection alone or from endpoint coarsening and reconstruction.

## Public KDIGO-component compatibility audit

INSPIRE and MIMIC-IV are audited for postoperative creatinine, urine-output-duration and RRT components over 0–168 h. A multicomponent MIMIC endpoint is an algorithmic sensitivity definition only. INSPIRE urine-output criteria are considered non-estimable unless duration coverage is sufficient. Neither public definition reproduces source-centre nephrologist adjudication, and neither is labelled a same-endpoint external validation.

For each strategy/scenario report Monte Carlo bias, absolute bias, RMSE, empirical standard deviation, mean standard error when defined, 95% interval coverage when defined, failure fraction, weight effective sample size, and positivity violations. Failure conditions are results, not runs to be silently discarded.

## Calibration precision and stability

- Empirical precision audit conditional on the observed linear-predictor distribution, with simulated outcomes and sample-size multipliers 1, 1.5, 2, and 3.
- Precision targets: AUROC 95% interval width <=0.10, O/E width <=0.20, calibration-slope width <=0.30, and standardized net-benefit width <=0.20 at prespecified thresholds.
- Source-model stability: 200 centre-stratified bootstrap refits using frozen hyperparameters; report external prediction SD, rank stability, performance distribution, and patient-level instability by risk band.
- Centre calibration: fixed common slope with centre-specific intercepts, followed by empirical-Bayes normal-normal partial pooling. With only five centres, heterogeneity estimates are descriptive and reported with uncertainty; no centre-level significance claims are made.

## Decision-analytic monitoring evaluation

- Probability thresholds: 0.02 to 0.15 in increments of 0.01.
- Resource policies: top 10%, 20%, 30%, and 40% of predicted risk.
- Outcomes: net benefit, standardized net benefit, sensitivity, PPV, events captured, patients monitored per event detected, false-positive monitoring, and additional creatinine tests assuming 1, 2, or 3 extra tests per monitored patient.
- Source and INSPIRE analyses are reported separately. INSPIRE includes complete-case, IPW/AIPW where identified, and MNAR-envelope results.
- Results are retrospective decision-analytic evidence, not clinical-impact evidence.

## Fairness and subgroup audit

Prespecified groups: sex, age (<65/≥65 years or database-compatible bins), gastric/colorectal cancer, open/minimally invasive approach, source centre, and INSPIRE observability stratum. Report n/events, O/E, Brier, calibration-in-the-large, AUROC where estimable, threshold sensitivity/PPV, monitoring rate and net benefit with bootstrap intervals. No subgroup-specific model is fit. Sparse groups use partial pooling or are labelled not estimable.

## Performance-portability frontier

Prespecified source models:

1. preoperative harmonized feature set;
2. end-of-surgery harmonized feature set;
3. source-native extended end-of-surgery feature set selected from prediction-time variables before model fitting and documented with missingness/units.

Compare source LOCO and external portability, incremental Brier, log loss, AUROC, decision utility, missingness burden, and predictor harmonization coverage. No search over additional algorithm families is allowed.

## Multiplicity and interpretation

The six methodological mechanisms and correction methods are prespecified. Simulation comparisons emphasize effect sizes, uncertainty, bias and coverage. Clinical subgroup results are exploratory and use false-discovery-rate control for any collection of formal interaction tests; descriptive performance intervals are not converted into significance claims.

## Reproducibility and release

All code is Jupytext `py:percent`, seeded, parameterized by CLI, and produces aggregate machine-readable audits. Patient-level source/INSPIRE/MIMIC data remain under `secure_work/`. The public package contains only generic algorithms, synthetic examples, aggregate tests, protocol, environment manifest and documentation.

## Claim boundaries

- `Y_full` is a densely observed EHR creatinine reference, not biological truth or full urine-output KDIGO.
- Source `PostopAKI` is an author-confirmed site-level dual-nephrologist KDIGO 2012 adjudication; individual component forms and inter-rater agreement are unavailable in the analytic extract.
- MIMIC is an independent methodological replication, not same-population external validation of the gastric/colorectal model.
- MIMIC's multicomponent sensitivity endpoint is algorithmic and cannot substitute for source adjudication.
- Weighting does not identify latent-outcome-dependent missingness without sensitivity assumptions.
- Local recalibration to `Y_obs` may fit institutional observation practice rather than latent clinical risk.
- No retrospective analysis establishes clinical benefit, safety, workflow uptake, or practice change.
