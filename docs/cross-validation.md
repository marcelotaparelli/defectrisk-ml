# Logistic Regression: group-aware cross-validation

## Scope and method

The existing deterministic M3 group-aware split is unchanged. Only its 8,708
training rows enter cross-validation. The final test outputs are discarded by
the command and are never passed to fold construction, preprocessing, fitting,
prediction, scoring, or parameter selection. No duplicate rows are removed.

`src/defectrisk/cross_validation.py` uses
[`StratifiedGroupKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
with `n_splits=5`, `shuffle=True`, and `random_state=42`. This splitter attempts
to preserve class proportions while keeping groups disjoint. Group identity
comes from the existing `feature_groups` helper: every raw feature column,
excluding the target and row index. Matching missing values compare equal.
Groups with conflicting target labels remain indivisible. Stratification can
only be approximate when groups have different sizes or class mixtures.

Each training row is used for validation exactly once and for fitting in the
other four folds. A fresh pipeline from `create_baseline_pipeline()` is created
for every fold, with the existing configuration unchanged:

1. `SimpleImputer(strategy="median")`
2. `StandardScaler()`
3. `LogisticRegression(random_state=42, max_iter=5000)` with all other defaults

The imputer and scaler learn only from that fold's fitting rows. The validation
rows receive only transformation and prediction. Defective (`"true"`) is the
positive class for precision, recall, and F1. Undefined metrics are reported as
zero, matching the existing baseline's `zero_division=0` setting.

No hyperparameters, class weights, decision thresholds, feature sets, or seeds
were selected using these results. No alternative models or final refit were
added.

## Real JM1 results

Executed against cached OpenML JM1, dataset 1053, using scikit-learn 1.9.1.
M3 training contains 7,023 clean and 1,685 defective rows (19.35% defective).
The complete console report is saved in
[cross-validation-results.txt](cross-validation-results.txt).

| Fold | Accuracy | Defective precision | Defective recall | Defective F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.8020 | 0.4286 | 0.0712 | 0.1221 |
| 2 | 0.8191 | 0.6667 | 0.1306 | 0.2184 |
| 3 | 0.8139 | 0.5942 | 0.1217 | 0.2020 |
| 4 | 0.8084 | 0.5246 | 0.0950 | 0.1608 |
| 5 | 0.8110 | 0.5714 | 0.0950 | 0.1628 |

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Accuracy | 0.8109 | 0.0057 |
| Defective precision | 0.5571 | 0.0789 |
| Defective recall | 0.1027 | 0.0212 |
| Defective F1 | 0.1732 | 0.0339 |

Means weight all five folds equally. Standard deviations use `ddof=0` and are
calculated from unrounded fold metrics. They describe variability across this
fixed set of five folds, not standard errors or confidence intervals; the folds'
fitting sets overlap.

## Fold sizes, stratification, and contamination proof

Before creating each fold's pipeline, the independent `feature_overlap` audit
compares all raw feature columns across its fitting and validation partitions.
Evaluation fails if any shared vector is found. The actual run produced:

| Fold | Fit rows | Validation rows | Fit defective | Validation defective | Validation defective % | Shared vectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,966 | 1,742 | 1,348 | 337 | 19.3456 | **0** |
| 2 | 6,967 | 1,741 | 1,348 | 337 | 19.3567 | **0** |
| 3 | 6,967 | 1,741 | 1,348 | 337 | 19.3567 | **0** |
| 4 | 6,965 | 1,743 | 1,348 | 337 | 19.3345 | **0** |
| 5 | 6,967 | 1,741 | 1,348 | 337 | 19.3567 | **0** |

Fit defective proportions range from 19.3484% to 19.3539%. The complementary
rows in every partition are clean. Both affected-row counts and the matching
cross-boundary row-pair count are also zero in every fold; see the console
report for these independent audit fields. No conflicting-label group crosses
a fold boundary.

## Is the previous 9.5% recall representative?

The previous group-aware single holdout produced recall `32/337 = 0.094955`,
reported as 9.5%. The five-fold mean is 10.27%, with a population standard
deviation of 2.12 percentage points and a range of 7.12%–13.06%.

The single result is 0.77 percentage points below the cross-validation mean,
about 0.36 fold standard deviations, and matches the rounded recall in folds 4
and 5. It appears representative, slightly below average rather than unusually
high or low. This is a descriptive comparison, not a statistical significance
test. Across all folds, the unchanged baseline still detects only a small
fraction of defective modules. These are training-side generalization estimates,
not final test performance.

## Running and verification

From the project directory with its Python environment available:

```sh
python -m defectrisk.cross_validation
python -m pytest -q
```

The complete suite passed: **63 tests**, without exclusions. Added tests verify
five folds, deterministic reproducibility, one validation appearance per row,
complete row coverage, class balance, target exclusion, conflicting-label and
missing-value group integrity, and zero overlap on synthetic data and real JM1.
They also verify fresh pipelines, training-only preprocessing, positive-class
metric calculation, population standard deviation, rejection of a contaminated
fold before fitting, invalid-input handling, and that the command passes only
the M3 training partition. All prior tests remain included.

The workspace virtual environment's Python symlinks remain broken. The commands
were run with the temporary compatible Python 3.14 interpreter under `/tmp`,
using the existing installed packages through `PYTHONPATH` and the existing JM1
cache. No dependencies were installed or replaced. No convergence warning was
observed during the real five-fold evaluation.
