# Model card and evaluation framework card

## Intended use

The source probability engines estimate recorded postoperative AKI risk at the end of surgery. In this manuscript they serve as a clinical case study. The primary contribution is an evaluation framework that tests whether a longitudinal outcome-measurement process changes calibration conclusions.

## Out-of-scope use

- Direct clinical deployment or automated treatment decisions.
- Claims of prospective patient benefit.
- Substitution of public creatinine endpoints for expert-adjudicated full KDIGO.
- Use of MIMIC-IV or eICU as external validation of the five-centre source model.
- Interpretation of simulated measurement deletion as proof of a particular hospital's monitoring policy.

## Models

Primary source engines are ridge logistic regression, a shallow random forest and histogram gradient boosting. An equal-weight probability mean is secondary. Public measurement experiments use fixed database-native ridge models and synthetic calibrated scores for controlled discrimination tests. The haemoglobin replication uses separately trained database-native ridge models and does not claim cross-database clinical transport.

## Data and landmarks

- Source cohort: 4014 unique patients, one gastric or colorectal cancer operation each, end-of-surgery landmark, 155 recorded adjudicated events.
- INSPIRE 1.4.2: surgery-end landmark and perioperative longitudinal creatinine.
- MIMIC-IV 3.1 and eICU 2.0: ICU-admission landmark and database-native replications.
- Dense-reference cohorts are selected, highly monitored subsets and do not identify population-wide performance without additional assumptions.

## Performance summary

The source models had AUC 0.693-0.703 under LOCO validation. Their small differences do not establish algorithmic superiority. Under strong mixed measurement deletion at 35% retention, apparent O/E after local updating was approximately 1.000 in all three public databases, while retained-reference O/E was 0.485, 0.771 and 0.736. Empirical creatinine schedules produced apparent O/E of 0.999-1.002 and retained-reference O/E of 1.057-1.292. The haemoglobin replication produced retained-reference O/E of 1.094-1.157 after apparent O/E was restored to 1.000.

## Known limitations

The source outcome lacks component-level adjudication records and reliable preoperative-AKI exclusion. A centre has one event, and chronological validation has 37 events. The source model lacks independent same-model clinical external validation. Public retained endpoints are operational laboratory references. Empirical schedules are semi-synthetic timing distributions rather than causal hospital policies; the haemoglobin endpoint is not adjudicated bleeding. Simulated deletion probabilities and outcome-dependent oracle weights are unavailable in practice.

## Monitoring and update rule

Before local updating, define the target endpoint, audit the measurements needed to determine it, quantify positivity and measurement density, and keep undetermined outcomes separate from negative outcomes. Use local recalibration only when the endpoint is comparable. Otherwise retain a reference sample or report explicit sensitivity analyses. No universal automatic update rule is claimed.

## Reproducibility

Analyses use fixed seeds, Jupytext `py:percent` files, input hashes and machine-readable audits. Aggregate Source Data and code are available under the MIT License in GitHub release v1.1.0 at https://github.com/blanchzll/outcome-measurement-transport/releases/tag/v1.1.0. [AUTHOR INPUT NEEDED: independent archive DOI.]
