# Reporting checklist crosswalk

This crosswalk prepares the author responses for the official Nature Portfolio smart PDFs and clinical reporting forms. It does not replace the official forms or corresponding-author sign-off.

## Nature Portfolio reporting summary

| Domain | Draft response and manuscript pointer | Status |
|---|---|---|
| Exact sample sizes | Source screening denominator n=4014 with 155 events; locked analysis n=3710 with 152 events. Public cohort flows and events are reported in Results, Figure 1 and Supplementary Information section 4. | Complete with selection-provenance limitation |
| Unit of analysis | One unique patient and one eligible operation in the source cohort; first eligible stay in MIMIC-IV and eICU; Monte Carlo replicate is the simulation unit. Methods, all cohort subsections. | Complete |
| Data exclusions | Landmark-based variable exclusions, invalid date sensitivity, dense-reference rules and database exclusions are reported in Methods and Supplementary Information sections 2-4. | Complete |
| Sample size | No prospective power calculation. Available retrospective cohorts were used; uncertainty and reference-event precision were quantified. A centre with one event and temporal validation with 35 events are reported as limitations. | Complete with limitation |
| Randomisation | Not applicable to the retrospective cohorts. Simulation seeds and random reference sampling were prespecified and reproducible. | Complete |
| Blinding | Source adjudicators were reportedly masked to predictions and candidate predictors. Analysts could not be blinded to computational conditions. Original adjudication records remain unavailable. | Author verification required |
| Replication | Longitudinal stress tests used INSPIRE, MIMIC-IV, eICU and VitalDB; extensions transported empirical schedules and repeated the dual-target analysis for a haemoglobin-decline endpoint. | Complete |
| Sex and gender | The locked 3710-patient cohort uses valid source sex codes. Among the 304 screened records not present in that file, 295 had unresolved sex coding and nine were female-coded. Public database fields are administrative sex variables; the basis of assignment is not available from the extracts. | Complete with selection-provenance limitation |
| Ethics | Five approval identifiers and the retrospective consent waiver are reported. Investigators confirmed coverage of the analysed gastric and colorectal cohorts through 25 June 2024; supporting local-governance records remain author-controlled. | Complete subject to portal attestation |
| Data availability | Public database access routes, source-data restrictions and the governed request route through the corresponding author are stated. | Complete |
| Code availability | Frozen code, tests and audits are released under the MIT License at GitHub tag v1.3.1. An archival DOI is optional for review and can be added if deposited. | Complete for review |

## Nature machine-learning reporting summary

| Domain | Draft response and manuscript pointer | Status |
|---|---|---|
| Intended use | Evaluation of outcome-measurement transport and calibration, not clinical deployment. Abstract, Discussion and Model Card. | Complete |
| Data provenance and versions | Five-centre source data; INSPIRE 1.4.2; MIMIC-IV 3.1; eICU 2.0; VitalDB 1.0.0. Methods. | Complete |
| Prediction landmark | End of surgery for the source and VitalDB. The locked public cross-database model uses ICU admission in INSPIRE, MIMIC-IV and eICU; database-native mechanism experiments retain their declared index times. Methods. | Complete |
| Target definition | Source `PostopAKI` is expert-reported KDIGO adjudication; public targets are creatinine-only operational endpoints. Methods and Supplementary Information section 4. | Complete with source audit limitation |
| Train-validation separation | LOCO internal-external validation, within-centre chronological validation and unseen-hospital public replication; no external outcomes used for model selection. | Complete |
| Leakage prevention | Post-landmark and timing-ambiguous variables excluded; imputation, encoding, scaling and tuning confined to training folds. | Complete |
| Model selection | Ridge logistic regression, shallow random forest and histogram gradient boosting were prespecified; voting was secondary; no 100-model search. | Complete |
| Hyperparameters | Frozen search spaces and modal locks are documented in Supplementary Information and executable code. | Complete |
| Missing data | Fold-contained preprocessing; centre-structured missing variables excluded from primary models; observability and weighting treated separately from predictor imputation. | Complete |
| Evaluation metrics | AUC, Brier score, O/E, calibration intercept and slope, decision curves, bias, RMSE, coverage, positivity and effective sample size. | Complete |
| Uncertainty | Patient or hospital bootstrap, Monte Carlo percentiles and Bayesian credible intervals are labelled by inferential unit. | Complete |
| Independent data | The INSPIRE-locked public model underwent unchanged same-endpoint validation in MIMIC-IV and eICU, with weak transport. No equivalent independent validation exists for the source clinical model. | Complete and prominently limited |
| Reproducibility | Fixed seeds, Jupytext `py:percent` code, input hashes, machine-readable audits, 51 figure-source CSVs grouped into 21 final-display sheets and synthetic tests. | Complete for review |

## STROBE observational-study mapping

| STROBE domain | Location | Status |
|---|---|---|
| Design in title or abstract | Abstract states five-centre cohort and public-database stress tests; Methods states retrospective design. | Complete |
| Background and objectives | Introduction. | Complete |
| Setting and dates | Methods, clinical cohort; exact source dates 4 December 2017 to 25 June 2024. | Complete |
| Participants and eligibility | Methods and Supplementary Information sections 2 and 4 report the source denominator, unique patients, study dates and consecutive enrolment status. The stated missingness rule still does not reproduce the supplied 3710-patient file and is retained as a provenance limitation. | Complete with limitation |
| Variables and measurement | Methods; Supplementary Information sections 2-6; 110-variable dictionary in Supplementary Tables. | Complete with source outcome audit limitation |
| Bias | Dense-reference selection, observability, preoperative-AKI bounds, source-data quality and missingness audits. | Complete |
| Study size | Available cohorts plus precision analyses and limitations. | Complete |
| Quantitative variables | Predictor definitions, units and range audits in Supplementary Tables. | Complete |
| Statistical methods | Methods and Supplementary Information sections 1, 3 and 5-8, including identification conditions, empirical schedule transport and probability reference sampling. | Complete |
| Participant flow and descriptive data | Figure 1, Supplementary Figure 5, and Supplementary Tables. | Complete |
| Outcome data and main results | Results and Tables 1-4. | Complete |
| Other analyses | Sensitivity, subgroup, temporal and simulation analyses labelled. | Complete |
| Limitations and interpretation | Discussion. | Complete |
| Funding | Grant identifiers and funder-role statement are included from the supplied source manuscript. | Complete subject to author sign-off |

## TRIPOD+AI mapping

| Domain | Location | Status |
|---|---|---|
| Intended use and users | Evaluation framework; source model not implementation-ready. Abstract, Discussion and Model Card. | Complete |
| Data sources and clusters | Five source centres and four public databases. Methods. | Complete |
| Outcome and prediction time | End-of-surgery source landmark; 0-7 day source outcome; public landmark differences and 0-168 h endpoint stated. | Complete with source verification need |
| Predictor availability | Landmark-role audit and 110-column dictionary. | Complete |
| Sample size and events | Results and Supplementary Information. | Complete |
| Missing data and preprocessing | Fold-contained procedures and centre-structured exclusion rules. | Complete |
| Model specification | Restricted engines, hyperparameter locks and reproducible code. | Complete |
| Validation design | LOCO and chronological internal validation; public mechanisms are not source-model external validation. | Complete |
| Performance and calibration | AUC, Brier, O/E, intercept, slope and uncertainty. | Complete |
| Model updating | Cross-fitted apparent-target and reference-sample updating; targets explicitly separated. | Complete |
| Subgroups and fairness | Measurement-aware descriptive audit; no fairness certification. | Complete with limitation |
| Clinical utility | Retrospective decision analysis only; no prospective impact claim. | Complete with limitation |
| Open science | Aggregate Source Data and tagged release v1.3.1 are prepared for review; an archive DOI is optional and can be added later. | Complete for review |

## PROBAST+AI self-audit

| Domain | Judgement | Reason |
|---|---|---|
| Participants | High concern for source clinical prediction claim | No untouched clinical external validation; the 4014-to-3710 selection does not reproduce from the stated missingness rule; source denominator and consecutive enrolment require confirmation; dense public cohorts are selected. |
| Predictors | Low to moderate concern | Landmark frozen; post-landmark and timing-ambiguous variables excluded; centre-structured variables removed; units still require author verification for some source fields. |
| Outcome | High concern for source clinical prediction claim | Reported expert adjudication is not independently auditable; preoperative AKI cannot be reliably excluded; public references are operational creatinine endpoints. |
| Analysis | Low to moderate concern for the methodological experiments | Fold containment, fixed designs, positive controls and replication are strong; simulated mechanisms and conditional dense estimand limit generality. |
| Applicability | High concern for clinical deployment; moderate for methodological use | Framework is suitable for retrospective stress testing but not a validated clinical AKI system. |

## MI-CLAIM-GEN mapping

| Domain | Location | Status |
|---|---|---|
| Clinical problem and intended use | Abstract, Introduction and Model Card. | Complete |
| Data provenance and cohort construction | Methods, Figure 1 and Supplementary Information. | Complete with author-controlled source gaps and a transparent selection-provenance limitation |
| Reference standard | Source adjudication description and public operational endpoint definitions are separated. | Complete with source audit limitation |
| Feature timing and leakage | Landmark-role audit and fold-contained pipeline. | Complete |
| Model development and baselines | Three restricted engines; no broad model fishing; equal-weight vote secondary. | Complete |
| Evaluation and calibration | Discrimination, calibration, loss, uncertainty and transport analyses. | Complete |
| External validation | Explicitly absent for the source model; public databases replicate mechanism only. | Complete and limited |
| Explainability | Not a central claim; feature stability and incremental value are supplementary. | Not applicable |
| Fairness | Measurement-aware descriptive audits; sex uncertainty retained rather than imputed. | Complete with limitation |
| Clinical deployment and impact | Not performed; monitoring workload is decision-analytic only. | Complete and limited |
| Reproducibility | Code, Source Data and audits are prepared in GitHub tagged release v1.3.1; an archive DOI is optional for peer-review access. | Complete for review |

## Author actions required on the official forms

1. Insert corresponding-author identity and date in both Nature smart PDFs.
2. Confirm ethics and consent-waiver wording verbatim.
3. Confirm sex-field provenance and whether sex or gender was collected.
4. Verify the final repository tag and software licence; add an archive DOI only if one is created.
5. Verify that every response matches the manuscript and portal after author information is inserted.
