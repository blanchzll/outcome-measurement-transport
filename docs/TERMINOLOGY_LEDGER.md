# Terminology ledger

| Canonical term | First-use definition | Variants retired | Decision |
|---|---|---|---|
| outcome measurement transport | Cross-setting transport of the process by which measurements are obtained and an endpoint is reconstructed | ascertainment transport; measurement alignment | Use for the paper's central construct. |
| retained operational reference endpoint | Binary creatinine endpoint derived from all available measurements in the prespecified 0–168 h window | complete endpoint; true endpoint; gold standard | Never call this biological truth or full KDIGO. |
| reconstructed endpoint | Endpoint recomputed after simulated measurement deletion | apparent endpoint; observed endpoint | Use when the longitudinal record has been coarsened. |
| apparent calibration | Calibration assessed against the reconstructed endpoint | local calibration | Reserve “local recalibration” for the updating procedure. |
| retained-reference calibration | Calibration of the same predictions against the retained operational reference endpoint | true calibration; corrected calibration | Use to distinguish target choice from model updating. |
| outcome observability | Whether the measurements required to determine an endpoint are present | ascertainment intensity; monitoring completeness | Measurement density is a component, not a synonym. |
| inverse-probability-of-ascertainment weighting (IPAW) | Weighting by estimated outcome-observability probability | IPW; inverse probability weighting | Use IPAW for measurement-pattern analyses; use oracle IPW only for the pure label-selection control. |
| acute kidney injury (AKI) | Source cohort outcome adjudicated under 2012 Kidney Disease Improving Global Outcomes criteria | postoperative renal injury | Public-database endpoints are operational creatinine endpoints, not expert-adjudicated AKI. |
| observed-to-expected ratio (O/E) | Observed event count divided by the sum of predicted probabilities | calibration ratio | Define once and use O/E thereafter. |
| leave-one-centre-out (LOCO) validation | Internal–external validation holding out one participating centre at a time | external validation | Never call it untouched or independent external validation. |
| dense-reference cohort | Public-database subset meeting prespecified longitudinal measurement-density requirements | complete-case cohort; gold-standard cohort | State that it is a selected, highly monitored population. |
| reference sample | Random subset retaining the operational reference endpoint for model-updating evaluation | adjudication sample | Public references are algorithmic, not clinician-adjudicated. |
| locked public-database external validation | INSPIRE-developed serialized model applied unchanged at the same ICU-admission landmark and against the same creatinine endpoint in MIMIC-IV and eICU | public transport test; public replication | This is genuine same-model external validation for the public model, not for the five-centre source model. |
| endpoint-transport clinical bridge | INSPIRE gastrointestinal model applied unchanged to the five-centre cohort while the reference endpoint changes from creatinine-only to expert full KDIGO | source external validation; clinical validation | Never call this strict same-endpoint external validation. |
| empirical measurement-schedule transport | Semi-synthetic mapping of observed donor measurement times to unchanged target trajectories | hospital-policy transfer; natural experiment | State the matching tolerance and never give the schedules a causal interpretation. |
| probability reference sample | Reference sample drawn with known, positive inclusion probabilities and evaluated outside sampled records | targeted adjudication; enriched validation set | Non-random designs require inverse known inclusion probabilities and risk-coverage diagnostics. |
| operational haemoglobin-decline endpoint | Haemoglobin decrease of at least 2 g/dL after a harmonised peri-landmark baseline through 168 h | postoperative bleeding; haemorrhage | Never call this adjudicated bleeding or a surgical complication. |
| post-discharge sensitivity bound | Performance under explicit assumed event fractions among short-stay recorded negatives | corrected incidence; recovered outcome | This analysis does not identify whether or when post-discharge events occurred. |
| retained trajectory \(L^*\) | All available longitudinal measurements retained for the operational reference | complete biological trajectory | Retained data are still observational and may miss biological events. |
| measurement indicator \(R(t)\) | Whether a measurement is retained at time \(t\) | missingness flag | Use to distinguish longitudinal coarsening from whole-label selection. |

## Locked claim boundary

The work establishes a reproducible evaluation failure mode and stress-testing framework. It does not establish that the source AKI model is implementation-ready, that measurement heterogeneity caused the observed five-centre calibration differences, or that retrospective decision analysis demonstrates clinical benefit.
