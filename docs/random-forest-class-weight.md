# Controlled experiment: Random Forest class weights

## Configurations and controls

Compare exactly:

```python
# A: default forest
RandomForestClassifier(random_state=42)

# B: weighted forest
RandomForestClassifier(random_state=42, class_weight="balanced")

# Unchanged reference
LogisticRegression(random_state=42, max_iter=5000, class_weight="balanced")
```

Both forest classifiers retain all other defaults, including the number of
trees, depth, leaf constraints, and bootstrap setting. Tests compare the entire
forest parameter dictionaries against the requested constructors and verify
that `class_weight` is their sole difference. The Logistic Regression
configuration is unchanged.

All three pipelines retain the existing median imputer and StandardScaler.
Weights and preprocessing are learned from each fold's fitting rows only.
[`class_weight="balanced"`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
weights classes inversely to their frequencies in the fitting data. This is the
requested `balanced` setting, not `balanced_subsample`.

The existing evaluation loop now accepts pipeline factories so the new
controlled variant uses the identical fold construction, integrity checks,
threshold application, and scoring logic. Existing baseline commands retain
their original model sets. There is no external resampling; the forests retain
their default internal bootstrapping. No other parameters, thresholds, or seeds
are changed, searched, or selected, and no default model is replaced.

## Evaluation integrity

Cached OpenML JM1, ID 1053, and the existing group-aware M3 final split are
unchanged. Only the 8,708 training rows enter this experiment: 7,023 clean and
1,685 defective. Final test outputs are immediately discarded and are never
inspected, fitted, predicted, scored, or used to select a candidate.

The existing `training_folds()` is called once, using
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`. All three
models receive the exact same fitting and validation rows in every fold.
Raw feature-vector groups exclude the target and row index, include every
feature, and treat matching missing values as equal. Conflicting-label groups
stay intact; all duplicate rows remain present.

The evaluator requires every training row to appear in validation exactly once
before any fitting. An independent exact-feature audit rejects contamination
before any model is created in that fold. Each of the fifteen pipelines is
fresh. Validation probabilities use the fitted `classes_` to identify the
defective column. The fixed rule is `P(defective) >= 0.50`, including exact ties,
matching the prior threshold and forest experiments.

| Fold | Fit rows | Validation rows | Validation defective | Shared vectors | Fit rows with match | Validation rows with match | Matching pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,966 | 1,742 | 337 | 0 | 0 | 0 | 0 |
| 2 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |
| 3 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |
| 4 | 6,965 | 1,743 | 337 | 0 | 0 | 0 | 0 |
| 5 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |

## Complete per-fold results

LR is balanced Logistic Regression; RF is default Random Forest; WRF is
weighted Random Forest. Precision, recall, and F1 refer to defective (`"true"`).
Flags equal `TP + FP`, and flag percentages use each fold's validation size.

| Model | Fold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Flagged | Flagged % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 1 | 0.7210 | 0.3586 | 0.5608 | 0.4375 | 189 | 338 | 1,067 | 148 | 527 | 30.2526 |
| RF | 1 | 0.8065 | 0.5000 | 0.2077 | 0.2935 | 70 | 70 | 1,335 | 267 | 140 | 8.0367 |
| WRF | 1 | 0.7623 | 0.3871 | 0.3917 | 0.3894 | 132 | 209 | 1,196 | 205 | 341 | 19.5752 |
| LR | 2 | 0.7186 | 0.3596 | 0.5816 | 0.4444 | 196 | 349 | 1,055 | 141 | 545 | 31.3038 |
| RF | 2 | 0.8196 | 0.5920 | 0.2196 | 0.3203 | 74 | 51 | 1,353 | 263 | 125 | 7.1798 |
| WRF | 2 | 0.7904 | 0.4581 | 0.4540 | 0.4560 | 153 | 181 | 1,223 | 184 | 334 | 19.1844 |
| LR | 3 | 0.7105 | 0.3372 | 0.5134 | 0.4071 | 173 | 340 | 1,064 | 164 | 513 | 29.4658 |
| RF | 3 | 0.7990 | 0.4393 | 0.1395 | 0.2117 | 47 | 60 | 1,344 | 290 | 107 | 6.1459 |
| WRF | 3 | 0.7840 | 0.4348 | 0.3858 | 0.4088 | 130 | 169 | 1,235 | 207 | 299 | 17.1740 |
| LR | 4 | 0.7114 | 0.3398 | 0.5223 | 0.4117 | 176 | 342 | 1,064 | 161 | 518 | 29.7189 |
| RF | 4 | 0.8090 | 0.5159 | 0.1929 | 0.2808 | 65 | 61 | 1,345 | 272 | 126 | 7.2289 |
| WRF | 4 | 0.7877 | 0.4444 | 0.3917 | 0.4164 | 132 | 165 | 1,241 | 205 | 297 | 17.0396 |
| LR | 5 | 0.7163 | 0.3421 | 0.5045 | 0.4077 | 170 | 327 | 1,077 | 167 | 497 | 28.5468 |
| RF | 5 | 0.8082 | 0.5133 | 0.1721 | 0.2578 | 58 | 55 | 1,349 | 279 | 113 | 6.4905 |
| WRF | 5 | 0.7714 | 0.3891 | 0.3175 | 0.3497 | 107 | 168 | 1,236 | 230 | 275 | 15.7955 |

## Final comparison: equal-fold mean ± standard deviation

SD is the population standard deviation across the five folds (`ddof=0`),
calculated before rounding. It describes fold variability, not a confidence
interval. Count means refer to a single fold's validation partition.

| Measurement | Balanced LR | Default RF | Weighted RF |
| --- | ---: | ---: | ---: |
| Accuracy | 0.7155 ± 0.0040 | 0.8085 ± 0.0066 | 0.7792 ± 0.0106 |
| Defective precision | 0.3475 ± 0.0097 | 0.5121 ± 0.0487 | 0.4227 ± 0.0292 |
| Defective recall | 0.5365 ± 0.0296 | 0.1864 ± 0.0283 | 0.3881 ± 0.0433 |
| Defective F1 | 0.4217 ± 0.0160 | 0.2728 ± 0.0366 | 0.4041 ± 0.0348 |
| TP | 180.8000 ± 9.9880 | 62.8000 ± 9.5373 | 130.8000 ± 14.5794 |
| FP | 339.2000 ± 7.1386 | 59.4000 ± 6.4062 | 178.4000 ± 16.2432 |
| TN | 1,065.4000 ± 7.0597 | 1,345.2000 ± 6.0133 | 1,226.2000 ± 16.2160 |
| FN | 156.2000 ± 9.9880 | 274.2000 ± 9.5373 | 206.2000 ± 14.5794 |
| Flagged modules | 520.0000 ± 15.8493 | 122.2000 ± 11.4438 | 309.2000 ± 24.6933 |
| Flagged % | 29.8576 ± 0.9098 | 7.0164 ± 0.6553 | 17.7537 ± 1.4174 |

## Pooled out-of-fold totals

Each training row contributes one held-out prediction per model. Pooled metrics
can differ slightly from equal-fold means because denominators are weighted
differently. The Logistic Regression and default forest reruns reproduce their
previous measured results to the reported precision.

| Model | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Flagged | Flagged % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Balanced LR | 0.7155 | 0.3477 | 0.5365 | 0.4219 | 904 | 1,696 | 5,327 | 781 | 2,600 | 29.8576 |
| Default RF | 0.8085 | 0.5139 | 0.1864 | 0.2735 | 314 | 297 | 6,726 | 1,371 | 611 | 7.0165 |
| Weighted RF | 0.7792 | 0.4230 | 0.3881 | 0.4048 | 654 | 892 | 6,131 | 1,031 | 1,546 | 17.7538 |

Aggregated confusion matrices, ordered `[[TN, FP], [FN, TP]]`:

- Balanced LR: `[[5327, 1696], [781, 904]]`
- Default RF: `[[6726, 297], [1371, 314]]`
- Weighted RF: `[[6131, 892], [1031, 654]]`

## Answers and product interpretation

1. **How much does weighting change forest recall?** Mean recall increases
   from **18.64% to 38.81%**, **+20.18 percentage points**, approximately 2.08
   times the default forest's recall. It improves in every fold. The pooled
   forest catches **340 more defects** and reduces false negatives from 1,371
   to 1,031. Mean F1 increases from 0.2728 to 0.4041 (about +13.12 percentage
   points, using unrounded values). This demonstrates sensitivity to imbalance
   handling in this configuration; it does not prove imbalance is the sole
   cause of the original forest's poor recall.
2. **What is sacrificed?** Versus the default forest, mean precision falls by
   **8.94 percentage points**, from 51.21% to 42.27%; accuracy falls by **2.93
   points**, from 80.85% to 77.92%. There are **595 additional false positives**
   and **935 additional review flags**. Workload rises from 611 to 1,546 modules,
   or **7.02% to 17.75%**, about +10.74 percentage points.
3. **Does weighted RF outperform balanced LR for the product objective?**
   **No on recall or mean F1.** Recall remains about **14.84 points lower**
   (38.81% versus 53.65%); it catches **250 fewer defects**, with 1,031 false
   negatives versus 781. Mean F1 is 0.4041 versus 0.4217. It does improve mean
   precision (42.27% versus 34.75%) and saves **804 false positives** and
   **1,054 review flags**. Its workload is 17.75% versus 29.86%. Thus it offers
   a useful workload/precision compromise, but fewer detected defects under
   the stated preference for reducing costly misses. The F1 averages are close;
   this single CV comparison does not establish statistical significance.
4. **Does RF justify its additional complexity?** Weighting makes the forest
   a substantially more useful candidate than its unweighted version, but the
   current evidence does **not justify replacing balanced LR for a recall-first
   objective**. A team constrained to a smaller review queue could value the
   weighted forest's intermediate operating point. No quantified review budget
   or error-cost ratio has been established to justify that choice. The forest
   remains an ensemble of 100 trees versus LR's linear classifier, without a
   higher mean F1 or recall in this comparison. No winner is declared solely
   from accuracy, and no model is automatically promoted.

## Execution and tests

Implementation: `src/defectrisk/random_forest_class_weight.py`.
Complete console output: [random-forest-class-weight-results.txt](random-forest-class-weight-results.txt).

```sh
python -m defectrisk.random_forest_class_weight
python -m pytest -q
```

Real JM1 evaluation ran with scikit-learn 1.9.1. No runtime warning was observed.
The complete test suite passed: **95 tests**, without exclusions. New tests
verify exact forest constructors with class_weight as the sole difference,
unchanged Logistic Regression and preprocessing, all three models' identical
existing folds, fifteen fresh pipelines, fitting-only preprocessing, one
fit/probability call per model/fold, fixed threshold and positive-column
selection, aggregate counts/workload, means/SD, preserved inputs, rejection of
contamination before any model, and that the command never inspects final test
outputs. All earlier tests, including real JM1 integrity checks and the original
two-model forest comparison, remain included.

The workspace's virtual-environment interpreter symlinks remain broken.
Execution used the compatible temporary Python 3.14 interpreter under `/tmp`,
existing installed packages through `PYTHONPATH`, and the existing JM1 cache.
No dependencies were installed or replaced. This milestone stops at the
requested class-weight experiment and reporting; no other tuning is implemented.
