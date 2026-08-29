# Author input required before submission

The analysis package can be completed without further clinical data, but the following
items cannot be inferred safely from datasets or code. They are submission blockers and
must be supplied or approved verbatim by the corresponding author.

## Identity and responsibility

1. Final author order, full names, degrees, affiliations, corresponding-author postal
   address, email, and ORCID identifiers.
2. CRediT contribution statement approved by every author.
3. Confirmation that every listed author meets authorship criteria and accepts final
   responsibility for the manuscript.

## Ethics and governance

The uploaded protocol and initiation materials now establish an original multicentre
retrospective gastric-cancer protocol (V1.1, 6 November 2022) and state that waiver of
informed consent was approved. The scanned approval letter is present. The following
submission fields and scope reconciliations remain unresolved:

1. Full name of each approving ethics committee or institutional review board, the exact
   approval or protocol number, and approval date. The scanned letter was not
   machine-readable in the supplied copy.
2. Final verbatim informed-consent waiver statement for the manuscript.
3. Provide the amendment, centre-specific approval, or written governance confirmation
   covering colorectal cancer surgery, the actual five-centre analysis set, and operations
   through June 2024. Protocol V1.1 describes gastric cancer only, 2013-22, and a planned
   participating-centre list that is not fully concordant with the analysed cohort.
4. Confirm whether enrolment was consecutive, document how the approximately 5000 planned
   source records became the 4014 authoritative operations, and whether the Sheet1 range
   of 4 December 2017 to 25 June 2024 is the intended study period. The previously stated
   2015-2024 period is not present in Sheet1 and must not be used without source verification.
5. Confirmation that the stated site-level adjudication procedure is documented in the
   protocol or case-report materials: KDIGO 2012; creatinine, urine output, and RRT; two
   nephrologists per case; disagreement resolved by a third coordinating-centre
   nephrologist; adjudicators masked to predictions and candidate predictors.
6. Approved statement governing access to the five-centre patient-level data.
7. Source correction, if available, for the 17 records with admission after surgery and
   the six records with surgery after discharge. The sensitivity analysis already excludes
   them, but corrected dates would improve the audit trail.
8. Exact outcome horizon confirmation (currently reported as postoperative day 0-7),
   handling of death/discharge before day 7, and whether every operation had complete
   laboratory and urine-output surveillance opportunity. In the current data, 171 patients
   had fewer than seven recorded postoperative inpatient days; clarify outpatient, transfer,
   or post-discharge follow-up for these patients if available.
9. Counts of first-versus-second adjudicator disagreement and third-review resolution,
   plus the original adjudication protocol or case-report form if governance permits.
10. Source resolution of the three records in which binary `PostopAKI` and `AKIStage`
    disagree. Until reconciled, `PostopAKI` remains primary and stage analyses are descriptive.
11. Definition and timing of the RRT and ventilator fields. Each contains values 2-5 despite
    a documented 0/1 code; ten patients coded RRT=1 are binary non-AKI. Confirm whether these
    values are durations, preoperative/chronic treatment, data-entry errors, or shifted fields.
12. Confirm that surgical-approach code 4 denotes robotic surgery. The source header lists
    only codes 1-3, whereas 60 records contain code 4 and 13 contain corrupted or combined codes.
13. Confirm whether the field labelled `术前POD1 Hb` is postoperative day-1 haemoglobin.
    Its position among postoperative laboratory variables is inconsistent with the word
    `术前`; it is currently treated as postoperative and excluded from prediction.
14. Confirm laboratory units and resolve the small number of implausible values, including
    preoperative WBC values 661 and 7027, BUN values 225 and 4004, haemoglobin below 30 or
    above 220, albumin below 10 or above 70, and creatinine below 20 in the stated units.
15. Explain why BMI, ASA grade, operation time, intraoperative fluid, and intraoperative
    vasoactive use are wholly absent in centre 1 and approximately 61-86% absent in centre 4.
    State whether collection began later, source systems differed, or records were unavailable.

## Funding, conflicts, and assistance

1. Every funding source and grant number, or an explicit statement that no specific
   funding was received.
2. Funder role in design, data collection, analysis, interpretation, writing, and the
   decision to submit.
3. Individual competing-interest declarations for all authors.
4. Final journal-compliant disclosure of AI-assisted coding, statistical checking, and
   language editing. Authors must confirm that they verified all generated code, numerical
   results, references, and prose and retain responsibility for the work.

## Reproducibility and submission administration

1. The public repository URL, version tag and MIT License are fixed at
   https://github.com/blanchzll/outcome-measurement-transport/releases/tag/v1.1.0.
   Add the archive DOI after author-controlled deposit.
2. Confirmation that public-data use statements and required INSPIRE and MIMIC
   acknowledgements meet the applicable data-use agreements.
3. The target journal and article type are now fixed as a Nature Communications Article.
   Confirm acceptance of the open-access publication charge and licence choice, then
   recheck portal fields on the day of submission.
4. Approval of the final title, 155-word abstract, cover letter, Data Availability statement,
   Code Availability statement, and transparent-peer-review choice.
5. Add the archive DOI for figure source data and release-safe code. Nature Communications
   verifies accessions and persistent links before publication.
6. Completed Nature Portfolio Reporting Summary and machine-learning reporting summary,
   signed off by the corresponding author rather than treated as administrative checklists.

Until these fields are completed, the correct status is **analysis-complete and formatted
for Nature Communications, but submission-blocked on author-controlled information**.
