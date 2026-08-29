# TRIPOD+AI / PROBAST+AI Reporting and Bias Audit

This is a project-specific working audit, not a substitute for the official submission
checklists. Status refers to the current reanalysis package on 2026-08-27.

| Domain | Current evidence | Status | Required action before submission |
|---|---|---|---|
| Study identification | Title and structured summary identify prediction evaluation and outcome-observation stress testing | Ready after results | Add final design terms and verified estimates |
| Data sources and dates | Five-centre source, INSPIRE 1.4.2, and MIMIC-IV 3.1 roles are separated | Partial | Authors must add source-centre recruitment dates and ethics details |
| Participants and eligibility | Exact 4014 source count; historical 3710 sex-code exclusion reconstructed; public cohort flow tables generated | Ready with limitation | Add clinical eligibility criteria exactly as approved in the source protocol |
| Prediction time and horizon | Source landmark frozen at end of surgery; operational public endpoint spans 0–168 h | Ready | Ensure every predictor is documented as available by the relevant landmark |
| Outcome definition | Source `PostopAKI` was adjudicated at each centre by two nephrologists using 2012 KDIGO creatinine, urine-output and RRT criteria; disagreements were resolved by a third coordinating-centre nephrologist; public creatinine-only operational definitions are executable | Ready with audit limitation | Distinguish site-level expert adjudication from a new central readjudication; do not call the public outcome full KDIGO |
| Outcome assessment blinded to predictors | Source adjudicators were masked to model predictions and candidate predictors | Ready with documentation requirement | Authors should retain the adjudication protocol or case-report form and state that centres adjudicated separately |
| Candidate predictors | P, PI and harmonised H sets frozen; final aggregate variable-definition/unit table generated; garbled sex token is missing rather than female | Ready with author check | Authors confirm source-laboratory units and clinical coding against the governed codebook |
| Sample-size rationale | Events, bootstrap precision, calibration intervals and model stability reported | Ready after final audit | Report effective sample sizes for weighted analyses and low-event-centre limitations |
| Missing predictors | Fold-internal imputation and preprocessing; unresolved sex retained in 4014 primary analysis | Ready | Report missingness by predictor and centre in supplement |
| Missing/partially observed outcomes | Empirical observability models, balance/weight diagnostics, AIPW, prediction-based MNAR sensitivity regions, a pure label-selection positive control, and controlled longitudinal measurement deletion | Ready after corrected simulation | State identification assumptions and distinguish feasible models from oracle design probabilities |
| Model specification | Ridge, restricted random forest and gradient boosting are primary families; voting is secondary | Ready | Publish exact locked hyperparameters and executable prediction pipeline |
| Internal/external validation | Five-centre LOCO primary; INSPIRE same-feature transport test; MIMIC database-native temporal mechanism replication | Ready with boundary | Never describe MIMIC as same-model clinical external validation |
| Performance measures | AUC, Brier, O/E, calibration intercept/slope, decision curves, intervals and centre/subgroup results | Ready after final audit | Include smoothed calibration curves only where event counts permit; label low-information estimates |
| Model updating | Two-fold cross-fitted apparent-target recalibration is evaluated against both reconstructed and retained-reference outcomes; held-out reference-sample recalibration uses 5%, 10%, 20%, and 30% samples; a 1000-replicate audit reports realised reference-event counts and identity-anchored penalised updating | Ready with boundary | State which endpoint each update targets; design by reference events rather than fraction alone; do not present the penalised sensitivity rule as externally validated |
| Clinical utility | Threshold, top-fraction, test burden and recorded-event capture scenarios | Scenario only | Do not claim clinical impact; call selected outcome-negative patients non-events rather than false alerts, and specify the prospective impact study needed |
| Model stability | 200 analytic-record bootstrap refits within centre, using frozen modal hyperparameters, with risk-rank and high-risk-set overlap | Ready with boundary | Report algorithm-specific instability; do not call this a full tuning-and-development confidence interval |
| Subgroups and equity | Sex, age, cancer site and surgical approach with minimum information rules and 500 bootstrap replicates | Descriptive | Call this a representativeness/performance audit, not fairness certification |
| Transparency and reproducibility | Versioned SAP amendment, executable py:percent scripts, aggregate outputs, open stress-test package, simulation-contract tests, and an immutable audit trail preserving the pre-correction analysis | Ready after package QA | Add repository URL, version tag, license after author approval, and environment lock |
| Protocol and registration | Retrospective reanalysis SAP is versioned | Partial | State that the expanded analyses were not prospectively preregistered |
| Data/code availability | Public database terms documented; source governance unresolved; aggregate/code release prepared | Author input | Add approved source-data statement and persistent repository URL |
| Ethics, consent, funding, conflicts, contributors | Not present in verified data or code | Blocked on author | Authors must supply exact statements; no placeholders may remain in submission |
| AI assistance disclosure | Draft contains a dedicated disclosure placeholder | Author input | Confirm target-journal wording and authors' verification responsibility |

## PROBAST+AI risk-of-bias posture

- Participants: some concern. Retrospective selection can be reconstructed, but the source
  eligibility protocol and dates still require author confirmation.
- Predictors: low-to-some concern after fold-contained preprocessing, provided all units,
  availability times, and coding are confirmed.
- Outcome: low-to-some concern for the source-model case study because site-level dual
  nephrologist adjudication used a uniform 2012 KDIGO protocol and was masked to model
  information. Residual concern remains because the analytic extract cannot support a new
  central readjudication, component-level audit or empirical assessment of between-centre
  adjudication agreement. Public operational creatinine endpoints serve a different role
  and do not retrospectively revalidate the source outcome. INSPIRE has insufficient
  postoperative urine-output density for a full KDIGO reconstruction; the MIMIC
  multicomponent endpoint is algorithmic and highly target-dependent, not expert adjudication.
- Analysis: some concern. Restricted model families, LOCO validation, precision and
  stability analyses reduce overfitting risk; only 155 events and one very-low-event centre
  constrain calibration and subgroup inference. Corrected simulations use sequential
  observed-history deletion, untruncated and truncated oracle weights, two-fold cross-fitted
  recalibration, held-out reference samples, and a pure-selection positive control. Their
  conclusions remain conditional on the stated data-generating mechanisms. Simulated
  observability probabilities are exact for independent deletion mechanisms but only
  conditional realised-history approximations for sequential history and mixed deletion.
- Applicability: high concern for direct deployment. The model is population- and
  workflow-specific, public databases serve different validation roles, and clinical impact
  is not observed.
- Reference-cohort selection: high concern for direct population transport. INSPIRE dense
  reference retained 23.5% of candidates and differed strongly in measurement count and
  cancer site; MIMIC retained 76.7% and also differed in measurement count. These selected
  operational cohorts support controlled stress tests, not unbiased incidence estimates.

The methodological claim can be stronger than the bedside-model claim. The manuscript
must maintain that separation in the title, abstract, figures, and conclusion.
