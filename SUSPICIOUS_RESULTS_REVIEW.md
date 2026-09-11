# Suspicious or high-risk findings requiring manual review

This file lists observations that may indicate non-realistic, overfitted, or weakly validated results and should be reviewed before publication.

## 1. Extremely high performance values

- [ ] mRNA-only AUC = 1.0000 and BA = 1.0000 are unusually strong for a biological classification task and should be reviewed carefully.
- [ ] Late Fusion AUC = 1.0000 should be checked for leakage, feature leakage, or a train/test contamination issue.
- [ ] Very high metrics can still be valid, but they require a clear explanation of split strategy, preprocessing isolation, and nested validation.

## 2. Risk of leakage via feature selection

- [ ] If PSO or ANOVA filtering was run on the full dataset before splitting, the reported metrics may be inflated.
- [ ] Confirm that the feature-selection step is nested inside training folds and not applied to the full cohort before evaluation.
- [ ] Check whether the same subjects appear across train and test sets after harmonization, split generation, or external validation preparation.

## 3. External validation may be incomplete

- [ ] The external validation dataset appears to contain only tumor samples in some reports (for example, 307 predicted tumor samples and zero normal samples).
- [ ] This is not necessarily a problem, but it means the evaluation may not be a balanced tumor-vs-normal test for the external cohort.
- [ ] Confirm whether the task is truly binary detection in the external dataset or whether a different target definition was applied.

## 4. Path-specific or historical workflow mismatch

- [ ] The provided command list references old paths like `D:\LUAD_MultiOmics\LUAD_MultiOmics\`, `data/`, `datasets/`, and `venv/`.
- [ ] These older paths may not match the current repository layout and could be stale historical notes rather than final instructions.
- [ ] Verify whether the commands were run in the same environment and version of the repo as the final artifacts.

## 5. Multiple experiment versions create ambiguity

- [ ] Several experiment names such as `pso_tumor_normal_v3`, `pso_tumor_normal_target50`, `late_fusion_mrna_mirna_final`, and `late_fusion_3modality_final` suggest a long trial-and-error process.
- [ ] Without a clear experiment registry, it is hard to know which result is the official final result.
- [ ] The final manuscript should clearly specify the accepted experiment and the reasoning behind choosing it.

## 6. Unclear standardization of preprocessing

- [ ] It should be verified whether clinical features, mRNA, and miRNA were harmonized using the same patient IDs and filtered using the same inclusion criteria.
- [ ] The procedure for sample alignment across modalities should be documented in detail.
- [ ] If harmonization was done in a way that depends on outcome labels, that would be a major risk factor.

## 7. Need for statistical confidence reporting

- [ ] Only reporting a single BA/AUC value is not enough for a strong Q1-level paper.
- [ ] The metrics should be accompanied by confidence intervals and repeated-validation statistics.
- [ ] Without variance estimates, a perfect or near-perfect score can look suspicious and may reflect a favorable split rather than true robustness.

## 8. Questions to resolve before any publication claim

- [ ] Was PSO applied only to training folds, or to the full cohort?
- [ ] Did the external validation cohort use the same preprocessing as the discovery cohort?
- [ ] Was any sample filtering performed using label information?
- [ ] Are there any duplicated patient records across modalities or across folds?
- [ ] Are the reported metrics based on the same dataset used for hyperparameter selection?
- [ ] Is the training/validation split truly blinded to the final evaluation set?

## 9. Recommended review actions

- [ ] Re-run the final pipeline from a clean environment with locked dependencies.
- [ ] Generate a single reproducible experiment manifest for the official result.
- [ ] Compare the official result with nested-CV and repeated-split validation.
- [ ] Check whether the same high scores persist under a more realistic, less optimistic split strategy.
- [ ] If the metrics remain unusually high, document the reason clearly and provide a transparent discussion around the possibility of data leakage or cohort bias.

## 10. Summary

The project looks promising, but the currently reported scores are high enough that they require a careful leak-check and validation audit before being considered final publication-grade evidence. The major risks are feature-selection leakage, sample contamination during harmonization, and unclear experiment versioning.
