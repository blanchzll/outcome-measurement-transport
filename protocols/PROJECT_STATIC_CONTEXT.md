# AKI resubmission static context

This file contains only project facts and constraints expected to remain stable. Dynamic
results, logs, errors, and manuscript estimates belong in dated audit files and should not
be copied here.

## Research question and claim boundary

- Clinical case study: postoperative AKI risk estimated at the end of surgery in gastric
  or colorectal cancer surgery.
- Primary methodological question: how longitudinal outcome-observation processes alter
  apparent calibration and model updating across health-care settings.
- Main thesis: outcome-measurement transport is distinct from algorithm transport.
- The study is retrospective. It does not establish clinical benefit, safety,
  implementation readiness, or a practice-changing effect.

## Source cohort

- Primary analysis population: 4014 unique patients, one operation per patient, across five
  centres; 155 recorded AKI events. `MajorID` links one-to-one between authoritative Sheet1
  and the deidentified cohort, with no centre or outcome discrepancies.
- Observed surgery dates run from 4 December 2017 to 25 June 2024. Admission and surgery
  dates are complete; discharge date is missing for one patient. Seventeen patients have
  admission after surgery and six have surgery after discharge; the primary results retain
  them and a full-refit sensitivity analysis excludes them.
- Recorded postoperative inpatient stay is at least seven days for 3843 patients and under
  seven days for 171. Length of stay audits opportunity for inpatient observation but does
  not prove complete creatinine or urine-output surveillance through day 7.
- Authoritative Sheet1 has 110 columns. Columns 5-7 are direct identifiers and never leave
  governed source storage; columns 8-10 are dates used only for linkage, temporal splitting,
  and observation-opportunity audits. The deidentified analytic cohort has 104 columns.
- The end-of-surgery role audit classifies 41 variables as postlandmark or outcome-adjacent
  and 14 as timing-ambiguous; all are excluded from prediction. Fourteen raw source fields
  supply 13 engineered P/PI predictors.
- BMI, ASA grade, operation time, intraoperative fluid, and intraoperative vasoactive use
  are structurally missing in centre 1 and substantially missing in centre 4. They are not
  eligible for addition to the primary model through ordinary imputation.
- `PostopAKI` and `AKIStage` disagree in three patients. The RRT field has four values outside
  the documented 0/1 code and ten non-AKI patients coded RRT=1. These require source-system
  resolution; they are not silently corrected.
- Historical 3710 cohort: sensitivity analysis reproducing a legacy valid-sex-code filter,
  not the primary missingness rule.
- Cancer site: `Gastrocolorectal=1` means gastric cancer; `2` means colorectal cancer.
- Gender: `1` means male; `2` means female in the original dictionary; malformed text such
  as `濂` is an encoding artefact and must not be assigned a clinical meaning.
- Primary validation: five-centre leave-one-centre-out internal-external validation.
- Temporal validation: within each centre, the earliest approximately 70% by surgery date
  form development data and the later approximately 30% form validation data; same-day
  operations remain together. A pre-2022 versus 2022-or-later split is secondary because
  it is confounded by changing centre participation.
- Secondary legacy split: centres 3/4/5 for development and 1/2 for evaluation.
- Prediction landmark: end of surgery. Only variables available by that point are eligible.

## Source outcome

- `PostopAKI` is the source binary postoperative AKI outcome.
- At each centre, two nephrologists applied KDIGO 2012 creatinine, urine-output, and RRT
  criteria; disagreements were resolved by a third nephrologist at the coordinating centre.
- Adjudicators were masked to model predictions and candidate predictors.
- Centres adjudicated separately. The analytic extract does not contain component-level
  case forms, agreement statistics, or material for a new central readjudication.

## Model policy

- Primary model families: ridge logistic regression, restricted random forest, and one
  gradient-boosting model.
- Equal-weight soft voting is a secondary comparator.
- No mass screening of 100 engines, large-scale stacking, or performance-driven algorithm
  selection.
- Imputation, outlier handling, scaling, feature encoding, and tuning are fitted only in
  each training fold.
- Discrimination, calibration, precision, stability, and applicability boundaries are all
  required; no model-superiority claim is prespecified.

## Public-database roles

- INSPIRE 1.4.2: perioperative harmonised-feature transport and primary longitudinal
  measurement-deletion testbed.
- MIMIC-IV 3.1: different ICU landmark and database-native model; independent replication
  of the measurement-evaluation mechanism, not external validation of the source model.
- Public creatinine endpoints are operational references, not expert-adjudicated full KDIGO
  or biological ground truth.
- Public data may extend methodologic evidence but do not substitute for prospective
  clinical-impact evidence.

## Technical and release conventions

- Canonical workspace: `t630:${ANALYSIS_ROOT}/`.
- All substantive execution and all retained project files reside on `t630`.
- Analysis programs are Jupytext-compatible `py:percent` files.
- Patient-level outputs remain in `secure_work/`; delivery contains aggregate artefacts only.
- Every figure panel is saved separately as vector PDF and SVG plus 600-dpi TIFF, with a
  matching source-data CSV and colour-blind-safe styling.
- Fixed simulation master seed: 20260826. Production conditions use 300 Monte Carlo
  replicates.
- Original code and outputs are preserved before correction; fixes require an audit trail
  and a full non-selective rerun.
