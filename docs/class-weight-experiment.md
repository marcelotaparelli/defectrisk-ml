# Controlled experiment: class-weighted Logistic Regression

## Question and controls

Does `class_weight="balanced"` improve defect detection enough to justify the
additional clean modules flagged for review? The existing product objective is
to prioritize modules for review/testing before release, with false negatives
provisionally considered more costly than false positives (see
[problem-definition.md](problem-definition.md)).

The experiment compares exactly two configurations:

```python
# A: existing baseline
LogisticRegression(random_state=42, max_iter=5000)

# B: balanced baseline
LogisticRegression(random_state=42, max_iter=5000, class_weight="balanced")
```

`create_experiment_pipeline()` creates a fresh existing baseline pipeline and,
for B only, sets `classifier__class_weight="balanced"`. The original baseline
factory and default baseline commands remain unchanged. Both pipelines keep
median imputation, StandardScaler, and every other Logistic Regression setting.
The existing `predict()` classification rule is used without a threshold change.

Scikit-learn's [balanced class weights](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
are inversely proportional to the class frequencies in the fitting data:
`n_samples / (n_classes * class_count)`. Each fold's classifier computes these
weights from its own fitting labels; validation labels do not set the weights.
Class weighting changes the training objective, without resampling any rows.

## Evaluation integrity

The same cached OpenML JM1 dataset (ID 1053) and unchanged deterministic
group-aware M3 split are used. Only the 8,708 M3 training rows enter the
experiment; its final test outputs are discarded and never passed to fitting,
prediction, scoring, or the comparison.

The experiment calls the existing `training_folds()` once, using
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. For each fold,
both models receive the exact same fitting and validation rows. Group identity
uses all raw features, excludes the target and row index, and treats matching
missing values as equal. Duplicate vectors, including conflicting-label groups,
stay together. All rows are retained.

Before either model is created, the independent exact-feature overlap audit is
run on the fold's partitions. A contaminated fold raises an error before model
fitting. The real experiment reports **zero shared feature vectors, zero affected
rows on either side, and zero matching row pairs in all five folds**.

| Fold | Fit rows | Validation rows | Fit defective | Validation defective | Shared vectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,966 | 1,742 | 1,348 | 337 | 0 |
| 2 | 6,967 | 1,741 | 1,348 | 337 | 0 |
| 3 | 6,967 | 1,741 | 1,348 | 337 | 0 |
| 4 | 6,965 | 1,743 | 1,348 | 337 | 0 |
| 5 | 6,967 | 1,741 | 1,348 | 337 | 0 |

Every row receives exactly one held-out prediction per model. A fresh pipeline
is fitted for each model/fold combination, so neither preprocessing nor model
state is carried between folds. Defective (`"true"`) is the positive class;
undefined precision, recall, or F1 is reported as zero.

## Real per-fold results

| Model | Fold | Accuracy | Defective precision | Defective recall | Defective F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: baseline | 1 | 0.8020 | 0.4286 | 0.0712 | 0.1221 |
| B: balanced | 1 | 0.7210 | 0.3586 | 0.5608 | 0.4375 |
| A: baseline | 2 | 0.8191 | 0.6667 | 0.1306 | 0.2184 |
| B: balanced | 2 | 0.7186 | 0.3596 | 0.5816 | 0.4444 |
| A: baseline | 3 | 0.8139 | 0.5942 | 0.1217 | 0.2020 |
| B: balanced | 3 | 0.7105 | 0.3372 | 0.5134 | 0.4071 |
| A: baseline | 4 | 0.8084 | 0.5246 | 0.0950 | 0.1608 |
| B: balanced | 4 | 0.7114 | 0.3398 | 0.5223 | 0.4117 |
| A: baseline | 5 | 0.8110 | 0.5714 | 0.0950 | 0.1628 |
| B: balanced | 5 | 0.7163 | 0.3421 | 0.5045 | 0.4077 |

The rerun of A reproduces the previous
[cross-validation baseline](cross-validation.md) to the reported precision.

## Mean and standard deviation

| Metric | A mean | A SD | B mean | B SD | B minus A (percentage points) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 0.8109 | 0.0057 | 0.7155 | 0.0040 | -9.53 |
| Defective precision | 0.5571 | 0.0789 | 0.3475 | 0.0097 | -20.96 |
| Defective recall | 0.1027 | 0.0212 | 0.5365 | 0.0296 | +43.38 |
| Defective F1 | 0.1732 | 0.0339 | 0.4217 | 0.0160 | +24.84 |

Means weight the five folds equally; SD is the population standard deviation
across folds (`ddof=0`). All summaries and differences use unrounded metrics,
so subtracting displayed rounded means can differ in the last digit. SD describes
fold variability, not a confidence interval. The fitting sets overlap.

## Aggregated held-out confusion matrices

These sum the five validation matrices, counting every M3 training row once
per model. They are cross-validation predictions, not final test predictions.
Rows are actual classes and columns are predicted classes.

A: existing baseline

| Actual / predicted | Clean | Defective |
| --- | ---: | ---: |
| Clean | 6,888 | 135 |
| Defective | 1,512 | 173 |

B: balanced baseline

| Actual / predicted | Clean | Defective |
| --- | ---: | ---: |
| Clean | 5,327 | 1,696 |
| Defective | 781 | 904 |

Pooled matrix metrics may differ slightly from equally weighted fold means,
especially precision, because the number of positive predictions varies by fold.

## Product interpretation

1. **Recall increased substantially:** from 10.27% to 53.65%, an increase of
   **43.38 percentage points**, approximately 5.23 times the baseline recall.
   It improves in every fold. Across held-out predictions, B catches 904 of
   1,685 defective modules versus 173 for A: **731 more defects detected** and
   731 fewer false negatives. B still misses 781 defective modules.
2. **Precision and accuracy decreased:** mean precision falls by **20.96
   percentage points** (55.71% to 34.75%), and accuracy by **9.53 percentage
   points** (81.09% to 71.55%). There are **1,561 additional false positives**.
   Total review flags rise from 308 to 2,600, or approximately 3.54% to 29.86%
   of these rows. About two thirds of B's flagged modules are clean.
3. **Class weighting better fits the current recall-oriented product objective**
   because it detects many more defective modules and reduces costly misses.
   F1 also rises from 0.1732 to 0.4217 in the same controlled comparison.
   This comes with a substantially larger review workload, so the judgment
   depends on the project's provisional error-cost preference and review capacity.
   It does not establish that B is universally better or deployment-ready.

Under a simple additive false-positive/false-negative cost model, B has lower
error cost when each missed defect costs more than `1561 / 731`, approximately
**2.14 times** a false alarm. This is a break-even calculation from the observed
counts, not an assumed or measured business cost. The project has not specified
that cost ratio or a review budget. No default model was replaced as part of
this experiment.

## Execution and tests

Run from the project directory:

```sh
python -m defectrisk.class_weight_experiment
python -m pytest -q
```

The real experiment ran with scikit-learn 1.9.1. Complete output is retained in
[class-weight-experiment-results.txt](class-weight-experiment-results.txt).
No convergence warning was observed. The complete test suite passed:
**68 tests**, without exclusions.

New tests verify that class_weight is the sole pipeline-parameter difference,
that both models receive the existing identical folds, that all ten pipelines
are fresh, that preprocessing learns only from fitting rows, and that input
data is preserved. They also verify defective-class metrics, population SD,
aggregate matrices counting each row once, rejection of contamination before
either model runs, and the command's M3-training-only boundary. Existing real
JM1 fold-integrity and reproducibility tests remain included.

The workspace virtual environment's Python symlinks remain broken. Execution
used the compatible temporary Python 3.14 interpreter under `/tmp`, existing
packages through `PYTHONPATH`, and the existing JM1 cache. No dependencies were
installed or replaced.

No threshold change, resampling, Random Forest, GridSearchCV, other parameter
tuning, final test model evaluation, or subsequent experiment was implemented.
