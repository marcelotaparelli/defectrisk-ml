# Final held-out evaluation

The configuration below was recorded before final-test access. No new parameter search was performed. The selected configuration is the most frequent existing inner-CV winner (two of five folds); the prior CV benchmark estimates the nested tuning procedure, rather than this single configuration.

Frozen at: 2026-10-09T18:53:47.810872+00:00

## Every RandomForestClassifier parameter

| Parameter | Frozen value |
| --- | --- |
| bootstrap | True |
| ccp_alpha | 0.0 |
| class_weight | balanced_subsample |
| criterion | gini |
| max_depth | 8 |
| max_features | sqrt |
| max_leaf_nodes | None |
| max_samples | None |
| min_impurity_decrease | 0.0 |
| min_samples_leaf | 3 |
| min_samples_split | 2 |
| min_weight_fraction_leaf | 0.0 |
| monotonic_cst | None |
| n_estimators | 200 |
| n_jobs | None |
| oob_score | False |
| random_state | 42 |
| verbose | 0 |
| warm_start | False |

## Frozen data, preprocessing, and policy

Median SimpleImputer followed by StandardScaler, then weighted Random Forest. Full preprocessing parameters and versions are recorded in [frozen-configuration.json](final-evaluation-results/frozen-configuration.json).

Features (original order, all retained, including b): loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t, lOCode, lOComment, lOBlank, locCodeAndComment, uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount. Target defects is excluded. No engineered or pruned columns.

Fit once on complete M3 training: 8,708 rows, 1,685 defective, 7,023 clean. Unchanged OpenML 1053 version 1, deterministic group-aware 80/20 split with seed 42, grouping all original features without target. Duplicate rows remain. Training fingerprint: ad488942d1d0ee2a4aeb61dc30c811553930f4e6c262b29aadd09bc276a27e05.

Class weighting is balanced_subsample: inverse class frequencies calculated separately within each bootstrap training sample. No final-test labels enter weights.

Review policy: rank test modules by P(true), choose the score cutoff producing the closest attainable number to 30% of rows. Equal scores stay together; equal-distance ties prefer fewer reviews. Flag probability >= cutoff. No labels select this cutoff and no alternative policies are evaluated.



## One held-out evaluation

This held-out test was evaluated once after model selection was frozen.

Frozen configuration timestamp: **2026-10-09T18:53:47.810872+00:00**.
Evaluation timestamp: **2026-10-09T18:55:32.513027+00:00**.
Configuration SHA-256 before and after evaluation:
`aa41cb25142bc599c56083e77db57a2f12521ae504ef19149ca32510cf3e39b4`.

The fitted model used all 8,708 M3 training rows in one pipeline fit. It produced
final-test probabilities in one `predict_proba` call. Median imputation and
StandardScaler were fitted only on M3 training. The forest's balanced-subsample
weights use bootstrap training labels only.

Final test: **2,177 modules**, including **421 defective**
and **1,756 clean**. Final-test prevalence is
19.3385%.

Flagged: **652 modules (29.949472%)**.
The frozen closest-count/tie policy selected a probability cutoff of
**0.29795885923354354** from scores only. This is the cutoff for this
population's ranking policy, not a newly tuned fixed production threshold.
The target count was 653.1; tied scores make 652 the closest attainable count
under the already frozen rule. No test labels selected that cutoff.

| Metric | Held-out value |
| --- | ---: |
| Defective recall | 0.712589 |
| Defective precision | 0.460123 |
| Defective F1 | 0.559180 |
| Average Precision | 0.655320 |
| TP / defects caught | 300 |
| FP / clean modules flagged | 352 |
| TN / clean modules not flagged | 1404 |
| FN / defects missed | 121 |

Confusion matrix: rows actual clean/defective, columns not flagged/flagged:

| | Not flagged | Flagged |
| --- | ---: | ---: |
| Actual clean | 1404 | 352 |
| Actual defective | 121 | 300 |

Average Precision is the step-weighted precision–recall summary used throughout
these studies, not trapezoidal PR integration. No additional PR-AUC calculation,
alternative threshold, or model was evaluated on the held-out data.

## Evaluation integrity and CV comparison

Exact original-vector train/test audit: zero shared feature vectors, zero matched
training rows, zero matched test rows, and zero matching row pairs. Groups exclude
the target and retain conflicting-label duplicates together. No rows are removed.

| Metric | Previous nested group-aware CV | Held-out | Absolute change |
| --- | ---: | ---: | ---: |
| Recall | 0.565579 | 0.712589 | +0.147010 |
| Precision | 0.364855 | 0.460123 | +0.095268 |
| F1 | 0.443565 | 0.559180 | +0.115615 |
| AP | 0.402211 | 0.655320 | +0.253109 |

**Final-test generalization is better than expected on this held-out partition.**
Recall is 14.7010 absolute percentage points higher,
precision 9.5268 points higher, F1
0.115615 higher, and AP 0.253109
higher. Both comparisons use approximately 30% module review capacity.

This is an observed result for one predetermined group-aware held-out partition,
not evidence that this gain will recur in deployment. CV used fold-specific
inner-selected configurations fitted on smaller outer fit partitions; the
held-out model uses the pre-test plurality configuration and complete M3 training.
These differences and differing group composition are plausible contributors,
but their contribution was not investigated with additional test analyses.
The earlier CV reference estimates the nested selection procedure rather than
precisely this frozen configuration, so the comparison is contextual.

No configuration, class weighting, feature, preprocessing setting, seed, or
review policy was changed after seeing the test result. No additional model
fit, tuning, threshold experiment, or held-out re-evaluation was performed.
Model exploration remains finished.

## Single-use execution and complete tests

The freeze command writes its configuration exclusively and refuses overwrite.
Evaluation reserves an exclusive attempt marker before dataset loading/model
fitting. An interrupted attempt cannot be automatically retried. After successful
completion, subsequent calls return saved results without loading data, fitting,
predicting, or scoring. The fitted model is saved for provenance, not retrained.

Commands used for the real evaluation, in this order:

```sh
python -m defectrisk.final_evaluation freeze
python -m defectrisk.final_evaluation evaluate
python -m pytest -q
```

Complete test suite: **155 passed in 20.82 seconds**. New synthetic-data tests verify exclusive configuration freezing, complete parameter preservation, training-data fingerprints, one full-training fit and one test prediction, label-independent review cutoff, contamination rejection, interruption guards, and cached results without re-evaluation. Existing tests remain passing. The real held-out model was not refitted or rescored by these tests.

As in earlier milestones, the workspace interpreter symlink is broken; execution
used the compatible standalone Python and existing project packages, with
`OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`:

```sh
PATH=/tmp/defectrisk-python/python/bin:$PATH \
PYTHONPATH=/workspace/defectrisk-ml/.venv/lib/python3.14/site-packages:/workspace/defectrisk-ml/src \
python -m defectrisk.final_evaluation evaluate
```

The command above has already completed. Its single-use guard returns the saved
result; it does not authorize or perform another held-out evaluation.

Artifacts:

- [Exact frozen configuration](final-evaluation-results/frozen-configuration.json)
- [Evaluation state and timestamps](final-evaluation-results/evaluation-state.json)
- [Saved held-out result](final-evaluation-results/result.json)
- [Saved predictions and review flags](final-evaluation-results/test-predictions.csv)
- [Once-fitted pipeline](final-evaluation-results/fitted-model.joblib)
