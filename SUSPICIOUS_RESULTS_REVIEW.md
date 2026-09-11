# Results audit and risk review

This document records the main findings that should be checked before making any strong publication claim. The goal is to identify potential problems such as leakage, over-optimism, or weak validation.

## 1. High-performance findings that need scrutiny

- The reported mRNA-only AUC and balanced accuracy near 1.0000 are unusually high for a biological classification task and should be reviewed carefully.
- The late-fusion AUC of 1.0000 is a major flag for possible overfitting or contamination.
- Such scores may still be valid, but they require clear evidence that preprocessing, feature selection, and model tuning were kept strictly inside training folds.

## 2. Leakage risk during feature selection

- Feature selection must not be performed on the full dataset before the train/test split.
- PSO or ANOVA-based filtering should be nested inside cross-validation if it is used for model optimization.
- Confirm that patient samples are not duplicated or misaligned during harmonization across modalities.

## 3. External validation concerns

- Some external validation outputs appear to contain only tumor predictions and zero normal predictions.
- This does not necessarily invalidate the result, but it limits the interpretation of balanced classification performance.
- The target definition used in the external cohort must match the discovery setting exactly.

## 4. Historical workflow mismatch

- Some command examples reference old repository paths and legacy folders, suggesting historical or exploratory runs rather than the final pipeline.
- These old paths should not be interpreted as the final validated workflow without checking the exact commit and environment used.

## 5. Experiment version ambiguity

- Multiple experiment names suggest a long cycle of tuning and testing.
- Without a clear experiment manifest, it is difficult to know which result is the official final result.
- A publication-grade study requires a single clearly documented final configuration.

## 6. Statistical reporting gaps

- A single BA or AUC value is not sufficient for a strong journal-level analysis.
- Report means, standard deviations, confidence intervals, and repeated validation outcomes.
- Without variance estimates, unexpectedly perfect scores can appear suspicious.

## 7. Questions that must be answered before publication

- Was feature selection applied only inside training folds?
- Was the same preprocessing pipeline used for discovery and external validation?
- Were any inclusion or exclusion rules derived from outcome labels?
- Were patient IDs harmonized consistently across modalities?
- Did the evaluation use the same data split for hyperparameter tuning and final reporting?

## 8. Recommended review actions

- Re-run the final project in a clean environment with locked dependencies.
- Generate one canonical experiment record for the official result.
- Compare the final metric with nested cross-validation and repeated splitting.
- Check whether the result remains stable under a less optimistic validation design.
- If the metric remains near-perfect, provide a transparent discussion of why this is biologically and statistically plausible.

## 9. Final assessment

The pipeline is promising, but the strongest reported results should be treated as provisional until leakage and validation risks are checked systematically. The most important review points are feature-selection leakage, data contamination during harmonization, and unclear final experiment provenance.
