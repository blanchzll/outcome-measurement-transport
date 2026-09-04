# Final reference and citation audit

Date: 4 September 2026  
Target: Journal of Translational Medicine Methodology article  
Manuscript: `MANUSCRIPT_JTM.md`

## Scope and result

All 19 numbered references and every in-text bibliographic callout were re-audited. Sixteen DOI-bearing references resolved in Crossref with exact normalised title matches. OpenAlex returned no retraction flag for those 16 records, and Crossref returned no `update-to` relation. Three references appropriately have no DOI: the AMIA proceedings article (verified by PMID 37350883 and PMCID PMC10283136), the complete KDIGO 2012 guideline (verified against the official 141-page KDIGO PDF), and the tagged GitHub software release. The DOI/retraction screen is a reproducibility aid, not an exhaustive guarantee that no later editorial notice exists.

Final status: **PASS**.

## Corrections made

1. Restored numbered-reference order by first appearance. KDIGO and the four public-dataset papers are now references 12–16; the validation-sample-size and clinical-implementation papers are references 17–18.
2. Changed bibliographic callouts from inherited superscript style to numbered square brackets used in Journal of Translational Medicine articles. Author affiliations and measurement-unit superscripts were left unchanged.
3. Corrected reference 11 to the exact Crossref/BMJ title: *Transparent reporting of multivariable prediction models developed or validated using clustered data: TRIPOD-Cluster checklist*.
4. Removed DOI `10.1038/kisup.2012.1` from the complete KDIGO guideline citation. That DOI identifies the guideline's one-page “Notice,” not the 1–138 guideline. Reference 12 now links to the complete official KDIGO PDF with an access date.
5. Corrected the exact INSPIRE title in reference 13.
6. Corrected the exact Nature Medicine colorectal-surgery implementation title in reference 18.
7. Moved reference 8 to a sentence it directly supports. The study's own designed-discrimination result is no longer attributed to that AKI modelling paper.
8. Removed the unsupported STROBE claim from the reporting sentence and removed MI-CLAIM-GEN, which is designed for generative-AI studies and is not applicable to this ridge/random-forest/gradient-boosting methodology study. A STROBE mapping remains available as an internal observational-study completeness check, not as a cited manuscript claim.
9. Updated the manuscript and reproducibility references to immutable GitHub tag `v1.3.6`.

## Automated checks

- Reference numbering is sequential from 1 to 19.
- Every listed reference is cited, and no undefined citation appears.
- First appearances follow 1 through 19 without reversal.
- All 16 DOI references resolve and match title metadata exactly after normalisation.
- No OpenAlex `is_retracted=true` flag was returned for a DOI reference.
- No Crossref `update-to` relation was returned for a DOI reference.
- The three no-DOI references are intentional and have an official or persistent source.

Machine-readable evidence: `REFERENCE_CROSSREF_AUDIT.json`.

## Evidence boundaries

This audit verifies bibliographic identity, citation order, nearby claim compatibility and available retraction indicators. It does not convert cited literature into project-level evidence, independently reproduce the cited studies, or resolve limitations of the manuscript's underlying clinical data.
