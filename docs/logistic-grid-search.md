# Balanced Logistic Regression: group-aware C grid search

## Scope and fixed settings

Tune only the regularization-strength parameter:

```python
param_grid = {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
```

The existing pipeline remains median imputation, StandardScaler, and
`LogisticRegression(random_state=42, max_iter=5000, class_weight="balanced")`.
All other classifier settings retain their existing defaults. The threshold
remains exactly 0.50. No preprocessing, grouping, final split, class weights,
thresholds, dataset rows, or other models are tuned or changed.

Only the existing 8,708-row M3 training partition enters the search, including
all duplicate rows. Final test outputs are immediately discarded; they are not
inspected, fitted, predicted, scored, or used for candidate selection.

## GridSearchCV methodology

`src/defectrisk/logistic_grid_search.py` constructs
[`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
with the requested five C values and four scoring metrics: defective recall,
defective precision, defective F1, and accuracy. The splitter is
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. Groups are
passed explicitly to the real search:

```python
groups = feature_groups(X_train)
search.fit(X_train, y_train, groups=groups)
```

Group identity includes every raw feature column and excludes the target and
row index. Matching missing values compare equal. Conflicting-label duplicate
groups stay together. The splitter uses the same data, row order, groups, and
integer seed as earlier experiments. Before fitting, every splitter index pair
is explicitly compared with `training_folds()` and independently audited for
exact raw-feature overlap. The integer-seeded splitter reproduces these same
indices when GridSearchCV calls it.

The scoring dictionary uses `make_scorer(..., response_method="predict_proba")`
with defective (`"true"`) as the positive class. All four scorers apply the
existing rule `P(defective) >= 0.50`, including exact ties. This fixes the same
classification rule across candidates without threshold tuning. Undefined
precision, recall, or F1 is reported as zero.

Every candidate/fold combination fits a separate pipeline. Median imputation,
scaling, class-weight calculation, and classifier fitting use only that fold's
fitting rows. There are **25 CV fits**: five C values times five folds.
`error_score="raise"` prevents failed fits from being silently reported as
missing scores. No external resampling is performed.

**Primary selection metric is mean defective F1**, not accuracy. The highest
mean F1 candidate is reported; exact ties resolve to the first C in the supplied
grid. Recall is reviewed separately. `refit=False` avoids creating a selected
estimator fitted on all training data. No model is promoted, persisted, or
installed as a new default.

## Fold-integrity proof

These folds were verified identical to the trusted existing CV folds. Every
boundary has zero shared vectors, zero affected rows, and zero matching pairs.
The audit happens before the search's fitting step.

| Fold | Fit rows | Validation rows | Fit defective | Validation defective | Shared vectors | Fit rows with match | Validation rows with match | Matching pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,966 | 1,742 | 1,348 | 337 | 0 | 0 | 0 | 0 |
| 2 | 6,967 | 1,741 | 1,348 | 337 | 0 | 0 | 0 | 0 |
| 3 | 6,967 | 1,741 | 1,348 | 337 | 0 | 0 | 0 | 0 |
| 4 | 6,965 | 1,743 | 1,348 | 337 | 0 | 0 | 0 | 0 |
| 5 | 6,967 | 1,741 | 1,348 | 337 | 0 | 0 | 0 | 0 |

## Complete real JM1 results

All entries are **equal-fold mean ± population standard deviation** (`ddof=0`)
from GridSearchCV's `cv_results_`. Scores and ranks use unrounded values. SD
describes variability among these five folds, not a confidence interval.

| C | Accuracy | Defective precision | Defective recall | Defective F1 | F1 rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 0.718305 ± 0.005959 | 0.348482 ± 0.010969 | 0.524629 ± 0.024713 | 0.418722 ± 0.015176 | 5 |
| **0.1** | **0.719224 ± 0.006313** | **0.351317 ± 0.013176** | **0.534125 ± 0.031738** | **0.423779 ± 0.019276** | **1** |
| 1.0 (current) | 0.715549 ± 0.004045 | 0.347464 ± 0.009655 | 0.536499 ± 0.029638 | 0.421675 ± 0.015988 | 3 |
| 10.0 | 0.713368 ± 0.006054 | 0.345817 ± 0.011560 | 0.540653 ± 0.030226 | 0.421729 ± 0.017172 | 2 |
| 100.0 | 0.711989 ± 0.006447 | 0.344333 ± 0.012373 | 0.541246 ± 0.032592 | 0.420791 ± 0.018428 | 4 |

The C=1 row reproduces the previous balanced Logistic Regression reference
to its reported precision. C=0.1 has the highest mean defective F1.

## Best candidate versus current C=1

| Metric | Current C=1 | F1-selected C=0.1 | Change (C=0.1 minus C=1) |
| --- | ---: | ---: | ---: |
| Accuracy mean | 0.715549 | 0.719224 | +0.003675 (+0.3675 percentage points) |
| Defective precision mean | 0.347464 | 0.351317 | +0.003853 (+0.3853 points) |
| Defective recall mean | 0.536499 | 0.534125 | -0.002374 (-0.2374 points) |
| Defective F1 mean | 0.421675 | 0.423779 | +0.002104 (+0.2104 points) |

Differences are calculated before rounding. The F1 gain is **negligible in
practical size**, with only a small precision/accuracy increase and a small
recall decrease. The gain is much smaller than the observed fold F1 SDs
(approximately 0.016–0.019 for these two candidates); this comparison does not
establish statistical significance or a reliable improvement outside these
folds. Fold results share fitting data, so SD is not an uncertainty interval
for their difference.

The recall reduction is small, from about **53.65% to 53.41%**, rather than a
substantial sacrifice of defect detection. With 337 defective rows in every
validation fold, this difference corresponds to four fewer true-positive
out-of-fold predictions across the training partition. The project has not
defined a hard acceptable-recall floor, so this report does not automatically
approve the loss in exchange for a tiny F1 gain.

Larger C values produce slightly more recall but do not meaningfully improve
F1; even the highest measured recall, at C=100, is only about 54.12%. None of
the five values materially changes the baseline's defect-detection behavior.

**Conclusion:** C=0.1 is the requested F1-selected candidate for review, but the
observed benefit over C=1 is too small to justify automatic promotion. The
current configuration remains unchanged. These are training-side selection
scores on the same folds used for search, not an independent estimate of the
selected model's future or final test performance.

## Execution and verification

Implementation: `src/defectrisk/logistic_grid_search.py`.
Complete console output: [logistic-grid-search-results.txt](logistic-grid-search-results.txt).

```sh
python -m defectrisk.logistic_grid_search
python -m pytest -q
```

The real search ran on cached OpenML JM1, ID 1053, with scikit-learn 1.9.1.
No convergence or runtime warning was observed. The complete suite passed:
**102 tests**, without exclusions.

New tests verify the exact five-value C-only grid, unchanged balanced LR and
preprocessing, five-fold stratified group settings, fixed threshold and positive
label scoring, F1 selection rather than accuracy or recall selection, explicit
groups passed into the real GridSearchCV fit, all 25 fits restricted to the
trusted fold fitting rows, fitting-only imputation/scaling, no full-training
refit, mean/population SD calculations, unchanged input data, rejection of
contamination or different folds before search fitting, and a command boundary
that never inspects final test outputs. All earlier real JM1 integrity tests
remain included.

The workspace virtual environment's interpreter symlinks remain broken.
Execution used the compatible temporary Python 3.14 interpreter under `/tmp`,
existing packages through `PYTHONPATH`, and the existing JM1 cache. No
dependencies were installed or replaced. No Random Forest tuning or subsequent
experiment was implemented.
