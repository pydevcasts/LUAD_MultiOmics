# Q1 publication readiness checklist

This document summarizes the main actions needed to improve the project from a strong exploratory pipeline into a statistically robust, publication-ready multi-omics study.

## 1. Validation and reproducibility

- Use nested cross-validation or repeated validation rather than a single train/test split.
- Report mean ± standard deviation for all major metrics.
- Include confidence intervals for AUC, balanced accuracy, F1, sensitivity, and specificity.
- Fix and record all random seeds for preprocessing, sampling, and model training.
- Keep a complete record of data versions, code commit hashes, and configuration files for each experiment.

## 2. Statistical rigor

- Add significance testing for model comparisons, such as DeLong or paired bootstrap tests.
- Report sample counts, exclusion criteria, class imbalance, and missingness transparently.
- Clearly separate primary and secondary endpoints.
- Include ablation studies to quantify the contribution of each modality and feature-selection stage.
- Compare the fusion model against strong, relevant baselines under the same conditions.

## 3. External validation

- Validate the model on at least one independent external cohort.
- Ensure cohort harmonization is applied consistently across discovery and validation datasets.
- Report whether the external validation task matches the same biological objective and label definition.
- Document differences in preprocessing and their potential effect on generalization.

## 4. Interpretability and biological relevance

- Prepare publication-quality SHAP figures and explain modality-level importance clearly.
- Validate selected biomarkers against established LUAD biology and prior literature.
- Add pathway or gene-set enrichment analysis for the candidate markers.
- Distinguish clearly between model importance and biological causality.

## 5. Manuscript quality and reporting

- Write a clear introduction identifying the scientific gap and novelty.
- Provide a complete methods section covering preprocessing, harmonization, feature selection, and fusion strategy.
- Add tables and figures with consistent naming, labels, and statistical annotations.
- Include a limitations section and future-work discussion.
- Add supplementary materials for hyperparameters, code access, and dataset provenance.

## 6. Practical project improvements

- Remove exploratory or historical scripts that do not belong to the final pipeline.
- Standardize experiment names, artifact folders, and result files.
- Create a single entry point for the end-to-end workflow.
- Add CI checks for syntax validation and package import smoke tests.
- Version model checkpoints and selected feature sets for reproducible downstream analysis.

## 7. Priority order

1. Strong validation strategy and nested CV
2. External validation and generalization checks
3. Statistical significance and confidence intervals
4. Reproducibility and environment locking
5. Publication-ready figures and manuscript preparation

## 8. Final assessment

The project is promising and scientifically relevant, but to be competitive in a Q1 journal it should be converted from a strong exploratory workflow into a reproducible, statistically rigorous, externally validated, and publication-ready study. The main risks to address are leakage, validation transparency, and unclear statistical reporting.
