# Logistic Regression baseline

## Purpose and implementation

Logistic Regression is the first baseline because it is a simple, established
binary classifier with a linear decision boundary. It gives the project a clear
starting point before more complex models are considered. This choice does not
claim that it is the best model for JM1.

`src/defectrisk/baseline.py` uses the existing JM1 loader and M3 split utility.
The target `defects` is excluded from predictive features. Labels retain their
OpenML spelling: **`"true"` = defective = positive class**, and `"false"` = clean
= negative class. The report uses `pos_label="true"` explicitly.

## Pipeline and preprocessing

The scikit-learn `Pipeline` contains, in order:

1. `SimpleImputer(strategy="median")` to handle missing numeric feature values.
2. `StandardScaler()` to center and scale numeric metrics.
3. `LogisticRegression(random_state=42, max_iter=5000)` with other settings left
   at their defaults.

Actual missing-value totals remain unverified in this workspace. Median
imputation is included so missing values, if present, do not prevent fitting.
It is a fixed baseline choice rather than a result of comparing strategies.
JM1's predictive columns are numeric; no categorical encoding is added.

Complexity, size, and Halstead metrics can have very different magnitudes.
Scaling helps numerical optimization and makes regularization less sensitive to
feature units. Medians, means, and scales are learned only from the fitting
subset, inside the pipeline. Validation data receives only `transform` and
`predict`; no preprocessing is fitted on validation or final test data.

[Logistic Regression applies regularization by default](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
The default L2 regularization and strength are retained. There is no L1/L2
comparison, strength tuning, class-weight tuning, or decision-threshold tuning.
`max_iter=5000` raises the iteration limit to reduce convergence problems; it
does not prove convergence. Any convergence warning must be reviewed when the
command can run. `random_state=42` is supplied where applicable; the default
`lbfgs` solver does not itself use that seed.

## Training-side validation

The command first recreates the fixed M3 split: approximately 80% training and
20% final test, using whole feature-vector groups and seed 42. Only `X_train` and `y_train` are passed to
`baseline_report`; the final test outputs are discarded by the command.

A single additional group-aware holdout reserves approximately 20% of the M3
training data for validation, using seed 42. The remaining training data fits the complete
pipeline. Approximately 64% of all rows therefore fit the baseline, 16% validate
it, and 20% remain reserved for final testing, subject to whole-group constraints.
There is no cross-validation and no subsequent refit on all training rows in M4.

Validation results describe this one fixed baseline; they are not final test
performance. No alternative models or parameters are selected from the results.
The final test set remains untouched so later final evaluation can estimate
generalization without being influenced by development decisions.

Pipeline boundaries prevent leakage of fitted preprocessing statistics. Shared
group-aware splitting prevents identical raw feature vectors from crossing either
evaluation boundary, including conflicting-label groups. Metric-provenance
concerns remain; no duplicate rows or suspicious features are removed.

## Metrics and product interpretation

The command reports accuracy plus precision, recall, and F1 for defective modules.
Accuracy includes both classes and can hide poor detection of the minority class.
Precision measures the fraction of flagged modules that are defective; recall
measures the fraction of defective modules flagged for review. F1 combines
precision and recall. Undefined precision, recall, or F1 is reported as zero
using `zero_division=0`.

The confusion matrix uses clean then defective for both axes:

| Actual / predicted | Clean | Defective |
| --- | --- | --- |
| Clean | True negative (TN) | False positive (FP) |
| Defective | False negative (FN) | True positive (TP) |

A false positive sends a clean module for additional review or testing, consuming
engineering effort. A false negative leaves a defective module unflagged, which
may allow a defect to reach production. This makes defective recall particularly
relevant to review prioritization. False negatives are provisionally considered
more costly, as stated in the problem definition; that assumption remains open
to evidence rather than dictating all future metric choices.

## Run and verification status

After installation, from the project directory:

```sh
python -m defectrisk.baseline
python -m pytest
```

Implemented: pipeline creation, a single training-side validation split,
training, all requested metrics, a labeled confusion matrix, and tests.

Statically checked: pipeline order, fixed settings, positive-label mapping,
training-only call boundaries, documentation, and excluded later-stage features.

**Runtime verified after the evaluation-integrity fix:** the complete test suite
passes and real JM1 training-side validation was run with the unchanged pipeline.
See [the measured report](evaluation-integrity-fix.md) for results, execution
environment details, and the comparison with the old contaminated holdout.

## Deferred work

Full cross-validation, Random Forest, model selection, GridSearchCV, L1/L2 or
other parameter tuning, threshold selection, final test evaluation, persistence,
APIs, and deployment remain outside M4.
