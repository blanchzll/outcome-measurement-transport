# Literature Evidence Map

Search date: 2026-08-27. Primary route: PubMed and official journal pages. Queries
covered outcome ascertainment/informative presence, calibration with incomplete data,
TRIPOD+AI/PROBAST+AI, AKI surveillance, and adaptive clinical-AI validation.

## Reproducible search log

This was a targeted contextual search, not a systematic review. PubMed ESearch was run
on 27 August 2026 with the exact title/abstract queries below; counts are the live counts
returned on that date. Titles and abstracts of the small focused sets were screened for
direct relevance. The broad AKI and reference-sample sets were used to locate conceptual
and design precedents, not to support a categorical claim of uniqueness.

| Search family | Exact PubMed query | Records returned | Screening use |
|---|---|---:|---|
| Informative observation | `(\"informative presence\"[Title/Abstract] OR \"informative observation\"[Title/Abstract] OR \"informative visiting process\"[Title/Abstract]) AND (\"prediction model\"[Title/Abstract] OR calibration[Title/Abstract])` | 1 | One directly relevant review (Sisk et al.) |
| Endpoint error | `(\"outcome misclassification\"[Title/Abstract] OR \"label selection\"[Title/Abstract] OR \"outcome measurement error\"[Title/Abstract]) AND (\"prediction model\"[Title/Abstract] OR \"model evaluation\"[Title/Abstract])` | 4 | One directly relevant prediction-model validation paper (Zou et al.); three off-scope false positives |
| Weighting and calibration | `(\"inverse probability\"[Title/Abstract] OR AIPW[Title/Abstract]) AND calibration[Title/Abstract] AND \"prediction model\"[Title/Abstract]` | 12 | Method-neighbourhood screening; no claim that weighting itself is novel |
| AKI surveillance | `(AKI[Title/Abstract] OR \"acute kidney injury\"[Title/Abstract]) AND (surveillance[Title/Abstract] OR monitoring[Title/Abstract]) AND (detection[Title/Abstract] OR ascertainment[Title/Abstract])` | 351 | Plausibility and clinical-context search; neonatal protocol study retained as an example, not direct evidence for this cohort |
| Reference/validation sample | `(\"reference sample\"[Title/Abstract] OR \"validation sample\"[Title/Abstract] OR \"gold standard sample\"[Title/Abstract]) AND \"prediction model\"[Title/Abstract] AND (calibration[Title/Abstract] OR misclassification[Title/Abstract])` | 63 | Design-neighbourhood search for chart-reviewed or gold-standard subsamples |

Hand searching of the official BMJ pages added TRIPOD+AI, PROBAST+AI, and
TRIPOD-Cluster. Reference-list checking added Corbin et al. and Shin et al. The novelty
statement is deliberately conjunctive and descriptive: it identifies what this package
combines, not a provable claim that no earlier implementation exists.

| Evidence | Project use | Boundary | DOI/URL |
|---|---|---|---|
| Collins et al., TRIPOD+AI, BMJ 2024 | Requires explicit target population/outcome, missing-data methods, discrimination, calibration and clinical utility | Reporting guidance, not proof that this study is unbiased | https://doi.org/10.1136/bmj-2023-078378 |
| Moons et al., PROBAST+AI, BMJ 2025 | Separates model development quality from bias in model-performance evaluation; outcome measurement and applicability are explicit domains | Audit framework, not a numerical correction method | https://doi.org/10.1136/bmj-2024-082505 |
| Debray et al., TRIPOD-Cluster, BMJ 2023 | Requires transparent handling and reporting of clustered prediction-model studies, including internal-external validation | Reporting guidance; does not make a five-centre meta-regression reliable when one centre has one event | https://doi.org/10.1136/bmj-2022-071018 |
| McGee et al., Epidemiology 2022 | Informative healthcare contact/measurement can act as misclassification and simple adjustment for visit counts can fail | General EHR mechanism; not specific evidence that source-center AKI is underdetected | https://doi.org/10.1097/EDE.0000000000001432 |
| Sisk et al., JAMIA 2021 | Shows that informative presence and observation patterns in EHR data can change predictive associations and performance | Establishes prior work on informative observation; our novelty cannot be merely that test ordering is informative | https://doi.org/10.1093/jamia/ocaa242 |
| Corbin et al., 2023 | Formalises label-selection bias in model evaluation and shows that inverse-probability weighting can recover target performance when selection probabilities are known | Pure label selection differs from longitudinal measurement deletion followed by endpoint reconstruction; used as the positive-control benchmark | https://pmc.ncbi.nlm.nih.gov/articles/PMC10283136/ |
| Shin, Gail and Pfeiffer, Biostatistics 2022 | Weighting and auxiliary pseudo-risk can estimate O/E with incomplete validation information | Missing covariates, not longitudinal outcome misclassification; motivates but does not validate IPAW here | https://doi.org/10.1093/biostatistics/kxaa060 |
| Zou et al., Statistics in Medicine 2026 | Uses a small chart-reviewed gold-standard sample to correct outcome misclassification in prediction-model evaluation | Closest reference-sample precedent; our added question is longitudinal measurement deletion, calibration-target mismatch, and held-out local updating | https://doi.org/10.1002/sim.70377 |
| Quantitative bias analyses for time-to-event endpoint measurement error, AJE 2026 | Validation samples and quantitative bias analysis are principled responses when outcome measurement differs | Time-to-event context; our binary 168-h endpoint needs tailored implementation | https://doi.org/10.1093/aje/kwag027 |
| Neonatal AKI surveillance study, 2024 | A standardized creatinine screening protocol increased AKI detection (6% to 16%) | Neonatal population; supports plausibility only | https://pubmed.ncbi.nlm.nih.gov/38084834/ |
| Lim et al., INSPIRE, Scientific Data 2024 | Documents the perioperative public dataset and its longitudinal laboratory coverage | Single Korean academic center; not a multicenter clinical replication | https://doi.org/10.1038/s41597-024-03517-4 |
| Johnson et al., MIMIC-IV, Scientific Data 2023 | Documents the independent EHR testbed | ICU-admission landmark differs from surgery end | https://pubmed.ncbi.nlm.nih.gov/36596836/ |
| Adaptive validation strategies for real-world clinical AI, Nature Computational Science 2025 | Supports validation designs matched to translational stage and continued reassessment | Perspective/framework; does not replace clinical-impact evidence | https://doi.org/10.1038/s43588-025-00901-x |
| Riley et al., uncertainty of risk estimates, BMJ 2025 | Calibration precision depends on validation sample size, event count and the distribution of predicted risks; motivates interval and stability reporting | Methodological guidance; does not repair outcome misclassification | https://doi.org/10.1136/bmj-2024-080749 |
| Riley et al., external validation guide, BMJ 2024 | Calibration must be examined across clinically relevant risk regions and interpreted with uncertainty, not reduced to a pooled scalar | General validation guidance; assumes the observed outcome is an adequate evaluation target | https://doi.org/10.1136/bmj-2023-074820 |
| Clinical implementation of AI decision support in colorectal surgery, Nature Medicine 2025 | Provides a contemporary contrast between retrospective model evaluation and actual clinical implementation evidence | Different prediction target and workflow; cited only to define the evidence gap, not to claim comparable impact | https://doi.org/10.1038/s41591-025-03942-x |

## Claim rule

External literature supports the plausibility and importance of measurement-induced
evaluation bias. The study does not claim to be the first description of informative
observation, label selection, weighting, or reference-sample correction. Its narrower
contribution is the combination of time-stamped longitudinal measurement deletion,
endpoint reconstruction, cross-fitted double-target evaluation of recalibration, a pure
selection control, component-availability auditing, and replication in two public EHR
databases. Only project analyses support quantitative claims about these cohorts; external
associations are not treated as project-level causal evidence.
