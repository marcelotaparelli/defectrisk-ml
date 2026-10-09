# JM1 dataset inspection

## Verification status

The initial M2 write-up was prepared before local execution. Later milestones
loaded and audited the real cached JM1 data; its earlier verification blocker is
resolved. Current measured evidence is in the [duplicate audit](duplicate-audit.md),
[feature-quality study](feature-engineering.md) and [final model card](model-card.md).
The original source review below is retained with its provenance caveats.

The full-data audit recorded 10,885 rows, 8,824 unique feature vectors, 1,973
duplicate full rows and 2,061 duplicate feature rows beyond the first. Current
training-only schema/distributions and missingness are documented in the feature
study. No historical final-test data was accessed again during finalization.

## Dataset identity and source-reported schema

The loader selects [OpenML dataset 1053](https://www.openml.org/d/1053), JM1
version 1, with target `defects`. The
[OpenML metadata](https://www.openml.org/api/v1/json/data/1053) identifies
10,885 rows, 21 numeric feature columns, and the nominal target values `false`
and `true`. JM1 concerns module metrics from a C ground system; a module is a
function. The target records whether a module has reported defects.

Feature names in the source description (actual downloaded spelling and pandas
dtypes will be printed by the command):

```text
loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t,
lOCode, lOComment, lOBlank, locCodeAndComment,
uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount
```

## Source-reported class distribution

The [published dataset table](https://pmc.ncbi.nlm.nih.gov/articles/PMC5050670/)
gives the following counts. Percentages use counts divided by 10,885.

| Target | Interpretation | Count | Percentage |
| --- | --- | ---: | ---: |
| `false` | Clean | 8,779 | 80.65% |
| `true` | Defective | 2,106 | 19.35% |

The defective class is the minority, with roughly 4.17 clean modules for each
defective module. This is a substantial imbalance. A constant clean prediction
would have approximately 80.65% accuracy and zero defective-module recall;
accuracy alone therefore obscures the intended review-prioritization outcome.
No classifier was implemented to compute this illustration.

OpenML's free-text description reverses the class counts. It also states that
there are no missing attributes. Neither statement should substitute for
inspection of the loaded frame.

## Missing values, types, and duplicates

Actual training-side dtypes, missing values and distributions were subsequently
measured in the feature study; full-data duplicate counts are in the earlier
duplicate audit. Numeric source attributes do not guarantee a particular pandas
dtype across library versions.

The command reports duplicates in two ways:

- Full-row duplicates: all features and the target match.
- Feature-row duplicates: all features match, irrespective of the target.

Both count repeated rows beyond the first occurrence (`duplicated()`'s default),
rather than all rows belonging to duplicate groups. The raw data is preserved;
no missing values are filled and no duplicates are dropped.

## Leakage and identifier review

Schema review suggests the following concerns; these are not a complete leakage
audit of downloaded data:

- `defects` is the outcome and must be excluded from `X` in later milestones.
- `b` deserves provenance review: it is commonly interpreted as a Halstead defect
  estimate. A code-derived estimate available before release need not leak the
  target; a measured post-release defect count would. Its name alone does not
  settle this, so no feature is removed in M2.
- No obvious module ID, row ID, filename, or index column appears among the
  documented feature names. A pandas row index is not a predictive feature.
- If duplicate feature rows occur, placing related rows across future evaluation
  partitions could overstate generalization. The workflow reports this risk
  without constructing partitions or deciding how to handle duplicates.
- Confirm when metrics and defect labels were collected before treating the data
  as evidence for prospective pre-release predictions.

## Reproduce the inspection

With Python 3.12+ installed, from the project directory:

```sh
python -m pip install -e '.[dev]'
python -m defectrisk.data
python -m pytest
```

Optionally save the actual output separately for review:

```sh
python -m defectrisk.data > docs/jm1-inspection.txt
```

The first data load downloads from OpenML and subsequent runs reuse
`.cache/openml/`. The report includes dependency versions for interpreting dtype
differences. OpenML's source metadata records file ID 53936 and MD5
`0c19aefce52ccb955d0de57e4c0fe8bf` for the ARFF file; scikit-learn handles the
download and cache. Dependency version ranges do not constitute an environment
lockfile.

## Train/Test Split

M3 explicitly separates `X = frame.drop(columns="defects")` from
`y = frame["defects"]`. All other columns remain features; no feature selection,
cleaning, encoding, or imputation occurs. `split_dataset(frame)` returns
`X_train, X_test, y_train, y_test` using
[scikit-learn's train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
with `test_size=0.20`, `random_state=42`, and `stratify=y`.

The test set reserves unseen observations for a final assessment of
generalization. It must remain untouched during model development: using it for
decisions, tuning, preprocessing fits, or model selection would compromise that
assessment. M3 only reports partition sizes and label distributions to verify
the requested split; it does not use test data to make development decisions.

Stratification allocates rows using the target labels so both partitions retain
approximately the full dataset's class proportions. Integer row counts mean
the proportions may differ slightly. It does not rebalance the classes.

The seed `42` fixes the randomized assignment so repeated calls with the same
data, row order, parameters, and compatible library versions reproduce the split.
The number itself has no special statistical significance and was not chosen by
comparing model performance. The seed must not be tuned using the test set.

Original row indices are preserved. The function requires unique index labels;
tests check feature/target alignment, disjoint indices, complete row coverage,
approximately 80/20 sizes, class proportions, repeatability, and unchanged input.
These guarantees concern row identity: identical feature content can still
occur in both partitions. M3 does not resolve the duplicate-content concern
documented above.

Run the split report with:

```sh
python -m defectrisk.split
```

The command uses `load_jm1()` and prints full/train/test sizes and class counts
and percentages. Actual JM1 split sizes and class proportions have **not been
observed here**, because the workspace has no Python runtime. No runtime-dependent
split check is claimed to have passed.

## Scope boundary

M4 adds median imputation, scaling, and a Logistic Regression baseline with one
training-side validation holdout; see
[the baseline report](baseline-logistic-regression.md). Duplicate handling,
feature selection or encoding, model selection, cross-validation, GridSearchCV,
and final test evaluation remain deferred. No API or deployment is included.
