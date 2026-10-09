# Untuned Random Forest baseline

## Controlled comparison

Compare the current balanced Logistic Regression candidate with exactly:

```python
RandomForestClassifier(random_state=42)
```

Every Random Forest parameter other than `random_state` retains its default,
including `class_weight=None`. The requested default estimator is checked in
tests against a newly constructed `RandomForestClassifier(random_state=42)`.
See the [official classifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Both pipelines retain the existing median imputer and StandardScaler. The
forest replaces only the classifier step; Logistic Regression retains
`random_state=42`, `max_iter=5000`, and `class_weight="balanced"`. Keeping the
same preprocessing controls this comparison without introducing a separate
preprocessing experiment. Each pipeline learns preprocessing on fitting rows
only, and each model/fold combination uses a fresh pipeline.

Both models use the same fixed rule from the threshold experiment:
`P(defective) >= 0.50`. Exact probability ties count as defective. The probability
column is selected using the fitted classifier's `classes_`. No threshold is
searched or changed, and no production/default model is replaced.

No external dataset resampling is performed. The forest retains its standard
internal bootstrap behavior as part of the requested default configuration.
There is no class weighting for the forest, GridSearchCV, or other tuning.

## Trusted evaluation boundaries

Cached OpenML JM1 (ID 1053) and the existing group-aware M3 split are unchanged.
Only the 8,708 training rows enter the experiment: 7,023 clean and 1,685
defective. Final test outputs are immediately discarded and never inspected,
fitted, predicted, scored, or used in the comparison.

The existing `training_folds()` is called once. It supplies the same five
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` folds as the
Logistic Regression experiments. Within each fold, both models receive exactly
the same fitting and validation rows. Feature groups include every raw feature,
exclude the target and row index, and treat matching missing values as equal.
Conflicting-label duplicate groups stay together and all rows are retained.

Coverage is checked before fitting: every training row must receive exactly
one held-out prediction. Before either model is created in a fold, an independent
raw-feature overlap audit rejects any contamination. Actual results:

| Fold | Fit rows | Validation rows | Validation defective | Shared vectors | Fit rows with match | Validation rows with match | Matching pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6,966 | 1,742 | 337 | 0 | 0 | 0 | 0 |
| 2 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |
| 3 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |
| 4 | 6,965 | 1,743 | 337 | 0 | 0 | 0 | 0 |
| 5 | 6,967 | 1,741 | 337 | 0 | 0 | 0 | 0 |

## Complete per-fold results

LR denotes balanced Logistic Regression; RF denotes the untuned, unweighted
Random Forest. Precision, recall, and F1 refer to defective (`"true"`). Review
flags are `TP + FP`; their percentage uses the fold's validation size.

| Model | Fold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Flagged | Flagged % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 1 | 0.7210 | 0.3586 | 0.5608 | 0.4375 | 189 | 338 | 1,067 | 148 | 527 | 30.2526 |
| RF | 1 | 0.8065 | 0.5000 | 0.2077 | 0.2935 | 70 | 70 | 1,335 | 267 | 140 | 8.0367 |
| LR | 2 | 0.7186 | 0.3596 | 0.5816 | 0.4444 | 196 | 349 | 1,055 | 141 | 545 | 31.3038 |
| RF | 2 | 0.8196 | 0.5920 | 0.2196 | 0.3203 | 74 | 51 | 1,353 | 263 | 125 | 7.1798 |
| LR | 3 | 0.7105 | 0.3372 | 0.5134 | 0.4071 | 173 | 340 | 1,064 | 164 | 513 | 29.4658 |
| RF | 3 | 0.7990 | 0.4393 | 0.1395 | 0.2117 | 47 | 60 | 1,344 | 290 | 107 | 6.1459 |
| LR | 4 | 0.7114 | 0.3398 | 0.5223 | 0.4117 | 176 | 342 | 1,064 | 161 | 518 | 29.7189 |
| RF | 4 | 0.8090 | 0.5159 | 0.1929 | 0.2808 | 65 | 61 | 1,345 | 272 | 126 | 7.2289 |
| LR | 5 | 0.7163 | 0.3421 | 0.5045 | 0.4077 | 170 | 327 | 1,077 | 167 | 497 | 28.5468 |
| RF | 5 | 0.8082 | 0.5133 | 0.1721 | 0.2578 | 58 | 55 | 1,349 | 279 | 113 | 6.4905 |

## Mean and standard deviation across the five folds

Means weight folds equally; SD is population standard deviation (`ddof=0`).
Count means describe one validation fold, not totals across all folds. SD
describes variability among these folds, not a confidence interval; fitting
sets overlap. Values are calculated before rounding.

| Measurement | LR mean | LR SD | RF mean | RF SD |
| --- | ---: | ---: | ---: | ---: |
| Accuracy | 0.7155 | 0.0040 | 0.8085 | 0.0066 |
| Defective precision | 0.3475 | 0.0097 | 0.5121 | 0.0487 |
| Defective recall | 0.5365 | 0.0296 | 0.1864 | 0.0283 |
| Defective F1 | 0.4217 | 0.0160 | 0.2728 | 0.0366 |
| TP | 180.8000 | 9.9880 | 62.8000 | 9.5373 |
| FP | 339.2000 | 7.1386 | 59.4000 | 6.4062 |
| TN | 1,065.4000 | 7.0597 | 1,345.2000 | 6.0133 |
| FN | 156.2000 | 9.9880 | 274.2000 | 9.5373 |
| Flagged modules | 520.0000 | 15.8493 | 122.2000 | 11.4438 |
| Flagged % | 29.8576 | 0.9098 | 7.0164 | 0.6553 |

## Pooled out-of-fold results

Each of the 8,708 training rows contributes one held-out prediction per model.
These pooled metrics may differ slightly from the equal-fold means above.

| Model | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Flagged | Flagged % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR | 0.7155 | 0.3477 | 0.5365 | 0.4219 | 904 | 1,696 | 5,327 | 781 | 2,600 | 29.8576 |
| RF | 0.8085 | 0.5139 | 0.1864 | 0.2735 | 314 | 297 | 6,726 | 1,371 | 611 | 7.0165 |

Aggregated confusion matrices are `[[TN, FP], [FN, TP]]`:

- LR: `[[5327, 1696], [781, 904]]`
- RF: `[[6726, 297], [1371, 314]]`

LR reproduces the preceding balanced and threshold experiments at 0.50. No
final test predictions contribute to these matrices.

## Does the extra complexity provide product value?

1. **Does RF catch more defects? No.** It detects 314 of 1,685 defective modules,
   versus LR's 904. Mean recall falls from 53.65% to 18.64%, a decrease of
   approximately **35.01 percentage points**. RF recall is lower in every fold.
2. **Does RF reduce false negatives? No.** False negatives increase from 781
   to 1,371: **590 additional missed defects**. At this operating point it misses
   approximately 81% of defective modules.
3. **What happens to false positives and review workload? They decrease.** RF
   produces 297 false positives versus 1,696, saving **1,399 false alarms**.
   Total flags fall from 2,600 to 611, **1,989 fewer reviews**, reducing workload
   from 29.86% to 7.02% of modules. Mean precision rises from 34.75% to 51.21%.
4. **Is that meaningful value for the current product objective? It is a smaller,
   more precise review queue, but not an improvement in finding defects.** The
   stated objective prioritizes reducing missed defects; RF's lower recall and
   lower F1 do not justify replacing balanced LR on that basis. Mean F1 declines
   from 0.4217 to 0.2728 despite accuracy rising from 0.7155 to 0.8085. A strict
   review-capacity objective could value the smaller queue, but no such budget
   or quantified error costs have been established. Higher accuracy alone is
   insufficient to declare a winner.

The default forest is an ensemble of 100 trees rather than LR's single linear
classifier. In this measured configuration at 0.50, that added model complexity
does not buy better recall or F1. This conclusion concerns the specified
untuned, unweighted forest and fixed threshold, not every possible forest model.
No candidate is automatically promoted, tuned, or evaluated on the final test.

## Execution and verification

Implementation: `src/defectrisk/random_forest_baseline.py`.
Complete console output: [random-forest-baseline-results.txt](random-forest-baseline-results.txt).

```sh
python -m defectrisk.random_forest_baseline
python -m pytest -q
```

Real JM1 evaluation ran using scikit-learn 1.9.1. No runtime warning was observed.
The complete suite passed: **90 tests**, without exclusions. New tests verify
exact default forest settings with no class weights, unchanged LR and common
preprocessing, identical existing folds for both models, ten fresh pipelines,
fitting-only preprocessing, one fit/probability call per model per fold, fixed
0.50 threshold including ties, positive-column selection, pooled confusion
counts and workload, means/SD, unchanged inputs, rejection of contamination
before either model runs, and the command's refusal to inspect final test
outputs. Existing real JM1 fold-integrity tests remain included.

The workspace interpreter symlinks remain broken. Execution used the compatible
temporary Python 3.14 interpreter under `/tmp`, existing installed packages via
`PYTHONPATH`, and the existing JM1 cache. No dependencies were installed or
replaced. This experiment stops at measurement and reporting.
