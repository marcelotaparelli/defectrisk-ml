# Controlled experiment: classification thresholds

## Method and controls

This measures seven thresholds using the **same out-of-fold defective
probabilities**, without fitting a separate model for each threshold. The
unchanged M3 group-aware split supplies only its 8,708 training rows: 7,023 clean
and 1,685 defective. Final test outputs are immediately discarded; they are not
inspected, fitted, predicted, scored, or used to select a threshold.

The existing `training_folds()` supplies the same five
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` folds used in the
previous experiments. Raw feature groups include all feature columns, exclude
the target and row index, and treat matching missing values as equal. Identical
vectors, including conflicting-label groups, cannot cross a fold boundary. All
duplicate rows remain in the data.

Each fold fits exactly one fresh existing balanced pipeline:

```python
SimpleImputer(strategy="median")
StandardScaler()
LogisticRegression(random_state=42, max_iter=5000, class_weight="balanced")
```

All other classifier settings remain at their existing defaults. The imputer,
scaler, and class weights learn only from that fold's fitting data. The code
calls `predict_proba()` once on its validation rows and identifies the defective
probability column using the fitted `classes_`, rather than assuming column
order. Every training row receives exactly one held-out probability.

After those **five fits and five probability calls**, thresholds 0.20, 0.30,
0.40, 0.50, 0.60, 0.70, and 0.80 are applied to the stored probabilities:

```python
predicted_defective = probability_defective >= threshold
```

Exact threshold ties count as defective in this measurement. The original
pipeline's `predict()` behavior and production/default threshold are unchanged.
There is no resampling, class-weight change, additional model, hyperparameter
search, or automatic threshold selection.

## Evaluation-integrity evidence

Before each pipeline is created, an independent audit compares all raw feature
values across fitting and validation data and rejects shared vectors. Results:

| Fold | Fit rows | Validation rows | Validation defective | Shared vectors | Fit rows with a match | Validation rows with a match | Matching row pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,966 | 1,742 | 337 | 0 | 0 | 0 | 0 |
| 2 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |
| 3 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |
| 4 | 6,965 | 1,743 | 337 | 0 | 0 | 0 | 0 |
| 5 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |

Coverage is checked to require exactly five folds and exactly one validation
appearance per input row before any model fitting occurs.

## Complete real JM1 results

These metrics are calculated by pooling all 8,708 held-out predictions per
threshold. They are **not averages of fold metrics**. Precision, recall, and F1
treat defective (`"true"`) as positive. Undefined metrics are reported as zero.
The review count is `TP + FP`, and its percentage uses all 8,708 training rows
as the denominator.

| Threshold | Precision | Recall | F1 | Accuracy | TP | FP | TN | FN | Flagged modules | Flagged % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.1933 | 0.9970 | 0.3239 | 0.1944 | 1,680 | 7,010 | 13 | 5 | 8,690 | 99.7933 |
| 0.30 | 0.2030 | 0.9638 | 0.3354 | 0.2609 | 1,624 | 6,375 | 648 | 61 | 7,999 | 91.8581 |
| 0.40 | 0.2605 | 0.7881 | 0.3916 | 0.5262 | 1,328 | 3,769 | 3,254 | 357 | 5,097 | 58.5324 |
| 0.50 | 0.3477 | 0.5365 | 0.4219 | 0.7155 | 904 | 1,696 | 5,327 | 781 | 2,600 | 29.8576 |
| 0.60 | 0.4366 | 0.3596 | 0.3944 | 0.7863 | 606 | 782 | 6,241 | 1,079 | 1,388 | 15.9394 |
| 0.70 | 0.4993 | 0.2255 | 0.3107 | 0.8064 | 380 | 381 | 6,642 | 1,305 | 761 | 8.7391 |
| 0.80 | 0.5489 | 0.1300 | 0.2102 | 0.8110 | 219 | 180 | 6,843 | 1,466 | 399 | 4.5820 |

At 0.50, the confusion counts exactly reproduce the previous balanced experiment:
`[[5327, 1696], [781, 904]]`, ordered `[[TN, FP], [FN, TP]]`. The pooled precision
0.3477 and F1 0.4219 differ slightly from its equal-fold means 0.3475 and 0.4217
because pooling weights the denominators differently. Recall remains 0.5365.

## Product trade-offs

| Threshold | Meaning for defect review |
| --- | --- |
| 0.20 | Catches 1,680 of 1,685 defects, missing only 5. However, it sends almost every module for review; precision is essentially the dataset's defective prevalence. It offers little prioritization or workload reduction. |
| 0.30 | Catches 96.38% of defects but reviews 91.86% of modules. It still behaves much like reviewing nearly everything. |
| 0.40 | Raises recall to 78.81%. Compared with 0.50, it catches 424 more defects and produces 2,073 more false positives, adding 2,497 review flags. Reviewing 58.53% of modules is a substantial workload; only 26.05% of flags are defective. |
| 0.50 | Highest observed F1 among these seven thresholds: 0.4219. It catches 53.65% of defects while reviewing 29.86% of modules, with 34.77% precision. This describes the current operating point; no new default is selected. |
| 0.60 | Reduces review workload to 15.94%, with 43.66% precision, but recall falls to 35.96%. Compared with 0.50, it saves 1,212 reviews while missing 298 additional defective modules. |
| 0.70 | Reviews 8.74% of modules and reaches roughly 50% precision, but catches only 22.55% of defects. It provides a smaller priority queue with many missed defects. |
| 0.80 | Reviews only 4.58% of modules and has the highest precision (54.89%), but detects just 13.00% of defects and misses 1,466. Its highest accuracy does not make it the best fit for the stated objective of reducing missed defects. |

**Which catches the most defects?** Threshold **0.20**, with 1,680 true positives
and 99.70% recall, at the cost of reviewing 99.79% of modules.

**Which gives the best F1?** Threshold **0.50**, with pooled F1 **0.4219**, among
the seven tested values. This is an observed result on these training-side
OOF predictions, not a claim of an optimal threshold over all possible values
or independently confirmed final test performance.

**What happens as the threshold decreases?** Using fixed probabilities makes
the flagged sets nested. True positives and recall cannot decrease; false
negatives cannot increase. False positives and review workload cannot decrease.
Here, decreasing from 0.50 to 0.40 reduces false negatives from 781 to 357 and
increases false positives from 1,696 to 3,769. Precision falls in these measured
results; monotonic precision is not mathematically guaranteed in general.

**Can we improve recall beyond 53.65% with a reasonable workload?** Of the tested
values, **0.40 is the nearest recall-improving option**, but it nearly doubles
review flags (2,600 to 5,097) and requires reviewing about 59% of modules. It is
reasonable only if the team can sustain that workload and values the additional
424 detected defects enough to justify it. The project has not specified a
review budget, so none of the tested lower thresholds can confidently be called
a practical improvement with modest additional workload. Thresholds 0.30 and
0.20 require reviewing almost everything. No threshold is permanently changed.

## Execution and verification

Implementation: `src/defectrisk/threshold_experiment.py`.
Complete console output: [threshold-experiment-results.txt](threshold-experiment-results.txt).

```sh
python -m defectrisk.threshold_experiment
python -m pytest -q
```

The real experiment used cached OpenML JM1 and scikit-learn 1.9.1. No convergence
warning was observed. The complete suite passed: **85 tests**, without exclusions.
New tests cover exact threshold ties and endpoints, invalid thresholds and
probabilities, correct defective-class probability-column selection, metric and
confusion-count arithmetic, zero predicted positives, review percentages,
monotonic counts, label/probability alignment, and preserved input data. They
also prove only five fits and five probability calls for all seven thresholds,
fresh balanced pipelines, fitting-only preprocessing, correct OOF row alignment,
rejection of contamination and invalid coverage, and that the command never
inspects its final test outputs. Existing real JM1 integrity tests remain included.

The workspace virtual environment's interpreter symlinks remain broken.
Execution used the temporary compatible Python 3.14 interpreter under `/tmp`,
existing packages through `PYTHONPATH`, and the existing dataset cache. No
dependencies were installed or replaced.

This milestone is measurement only. No subsequent experiment or final test
evaluation was implemented.
