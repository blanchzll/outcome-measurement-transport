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

Primary source engines are ridge logistic regression, a shallow random forest and histogram gradient boosting. An equal-weight probability mean is secondary. Public measurement experiments use fixed database-native ridge models and synthetic calibrated scores for controlled discrimination tests. The haemoglobin replication uses separately trained database-native ridge models and does not claim cross-database clinical transport. VitalDB provides a non-ICU perioperative replication with a separately trained held-out ridge model.

## Data and landmarks

- Source cohort: 4014 screened operations; the locked analysis contains 3710 unique patients, one gastric or colorectal cancer operation each, an end-of-surgery landmark and 152 recorded adjudicated events.
- INSPIRE 1.4.2: surgery-end landmark and perioperative longitudinal creatinine.
- MIMIC-IV 3.1 and eICU 2.0: ICU-admission landmark and database-native replications.
- VitalDB 1.0.0: surgery-end landmark, perioperative creatinine trajectories and a creatinine-only operational reference.
- Dense-reference cohorts are selected, highly monitored subsets and do not identify population-wide performance without additional assumptions.

## Performance summary

The primary source random forest had leave-one-centre-out AUC 0.715 and O/E 0.981. Small differences between prespecified model families do not establish algorithmic superiority. Under strong mixed measurement deletion at 35% retention, apparent O/E after local updating was near 1 while retained-reference O/E remained displaced in the ICU testbeds and VitalDB. Cross-database empirical schedules produced apparent O/E of 0.999-1.027 and retained-reference O/E of 0.963-1.484 with 12-h matching. The haemoglobin replication also separated apparent from retained-reference calibration.

## Known limitations

The source outcome lacks component-level adjudication records and reliable preoperative-AKI exclusion. A centre has one event, and chronological validation has 35 events. The source model lacks independent same-model clinical external validation. The 4014-to-3710 selection does not reproduce from the stated row-missingness rule. Public retained endpoints are operational laboratory references. Dense-reference populations are selected. Empirical schedules are semi-synthetic timing distributions rather than causal hospital policies; the haemoglobin endpoint is not adjudicated bleeding. Simulated deletion probabilities and outcome-dependent oracle weights are unavailable in practice.

## Monitoring and update rule

Before local updating, define the target endpoint, audit the measurements needed to determine it, quantify positivity and measurement density, and keep undetermined outcomes separate from negative outcomes. Use local recalibration only when the endpoint is comparable. Otherwise retain a reference sample or report explicit sensitivity analyses. No universal automatic update rule is claimed.

## Reproducibility

Analyses use fixed seeds, Jupytext `py:percent` files, input hashes and machine-readable audits. Aggregate Source Data and code are available under the MIT License in GitHub tagged release v1.3.3 at https://github.com/blanchzll/outcome-measurement-transport/tree/v1.3.3. [AUTHOR INPUT NEEDED: independent archive DOI.]
