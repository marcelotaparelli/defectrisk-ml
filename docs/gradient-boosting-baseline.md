# Gradient Boosting baseline for JM1

Executed 2026-10-09 using Python 3.14 and scikit-learn 1.9.1. This is a controlled,
training-only comparison; it does not promote a model or deployment threshold.

## Method and evaluation integrity

- Reuse the existing deterministic group-aware M3 split, without modifying it.
  Discard final-test outputs immediately; no final-test inspection, prediction,
  fitting, scoring, or threshold selection occurs.
- Use only the 8,708 M3 training rows: 1,685 defective and 7,023 clean.
- Materialize the same five `StratifiedGroupKFold` folds once:
  `shuffle=True, random_state=42`. Groups use all original feature columns and
  exclude the target; missing feature values are grouped consistently. Conflicting
  target labels within a feature group stay together. No rows are removed.
- Before any fold fitting, audit exact raw feature-vector overlap. Every model
  uses the same fit and validation indices. Each training row receives exactly
  one out-of-fold probability per model; each pipeline is fitted once per fold
  (15 fits total). No fitting is repeated for review budgets.
- Balanced LR is unchanged: median imputation → StandardScaler →
  `LogisticRegression(random_state=42, max_iter=5000, class_weight="balanced", C=1.0)`.
- Weighted RF is unchanged: its existing median imputation → StandardScaler →
  `RandomForestClassifier(random_state=42, class_weight="balanced")` pipeline.
  Keeping its prior preprocessing preserves the trusted comparison.
- New HGB: median imputation →
  `HistGradientBoostingClassifier(random_state=42, class_weight="balanced", early_stopping=False)`.
  No StandardScaler. All other boosting parameters remain sklearn defaults
  (100 boosting iterations, learning rate 0.1, maximum 31 leaves,
  minimum 20 samples per leaf). Disabling internal early stopping avoids a
  separate random validation boundary. There is no parameter search.
- Each classifier computes balanced weights from its own CV training labels only.
  HGB supports this directly; see the
  [official sklearn API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
  There is no resampling or new dependency.
- At the fixed threshold, `P(defective) >= 0.50` flags a module. The positive
  probability column is located by class name `"true"`, not assumed by position.
  Undefined precision/recall/F1 are zero. SD is population SD (`ddof=0`) across
  the five equally weighted folds, consistent with previous experiments.

## Zero-overlap proof

All overlap measures are exactly zero in every fold: shared feature vectors,
matched fit rows, matched validation rows, and matching row pairs.

| fold | fit_rows | validation_rows | fit_defective | validation_defective | shared_feature_vectors | left_rows_with_match | right_rows_with_match | matching_row_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 6966 | 1742 | 1348 | 337 | 0 | 0 | 0 | 0 |
| 2 | 6967 | 1741 | 1348 | 337 | 0 | 0 | 0 | 0 |
| 3 | 6967 | 1741 | 1348 | 337 | 0 | 0 | 0 | 0 |
| 4 | 6965 | 1743 | 1348 | 337 | 0 | 0 | 0 | 0 |
| 5 | 6967 | 1741 | 1348 | 337 | 0 | 0 | 0 | 0 |

## Per-fold results at threshold 0.50

| model | fold | precision | recall | f1 | accuracy | tp | fp | tn | fn | flagged_modules | flagged_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced LR | 1 | 0.358634 | 0.560831 | 0.437500 | 0.721010 | 189 | 338 | 1067 | 148 | 527 | 30.2526% |
| Weighted RF | 1 | 0.387097 | 0.391691 | 0.389381 | 0.762342 | 132 | 209 | 1196 | 205 | 341 | 19.5752% |
| Balanced HGB | 1 | 0.363265 | 0.528190 | 0.430472 | 0.729621 | 178 | 312 | 1093 | 159 | 490 | 28.1286% |
| Balanced LR | 2 | 0.359633 | 0.581602 | 0.444444 | 0.718553 | 196 | 349 | 1055 | 141 | 545 | 31.3038% |
| Weighted RF | 2 | 0.458084 | 0.454006 | 0.456036 | 0.790350 | 153 | 181 | 1223 | 184 | 334 | 19.1844% |
| Balanced HGB | 2 | 0.381356 | 0.534125 | 0.444994 | 0.742102 | 180 | 292 | 1112 | 157 | 472 | 27.1109% |
| Balanced LR | 3 | 0.337232 | 0.513353 | 0.407059 | 0.710511 | 173 | 340 | 1064 | 164 | 513 | 29.4658% |
| Weighted RF | 3 | 0.434783 | 0.385757 | 0.408805 | 0.784032 | 130 | 169 | 1235 | 207 | 299 | 17.1740% |
| Balanced HGB | 3 | 0.378924 | 0.501484 | 0.431673 | 0.744400 | 169 | 277 | 1127 | 168 | 446 | 25.6175% |
| Balanced LR | 4 | 0.339768 | 0.522255 | 0.411696 | 0.711417 | 176 | 342 | 1064 | 161 | 518 | 29.7189% |
| Weighted RF | 4 | 0.444444 | 0.391691 | 0.416404 | 0.787722 | 132 | 165 | 1241 | 205 | 297 | 17.0396% |
| Balanced HGB | 4 | 0.357853 | 0.534125 | 0.428571 | 0.724613 | 180 | 323 | 1083 | 157 | 503 | 28.8583% |
| Balanced LR | 5 | 0.342052 | 0.504451 | 0.407674 | 0.716255 | 170 | 327 | 1077 | 167 | 497 | 28.5468% |
| Weighted RF | 5 | 0.389091 | 0.317507 | 0.349673 | 0.771396 | 107 | 168 | 1236 | 230 | 275 | 15.7955% |
| Balanced HGB | 5 | 0.362791 | 0.462908 | 0.406780 | 0.738656 | 156 | 274 | 1130 | 181 | 430 | 24.6984% |

## Mean ± SD across folds at threshold 0.50

| Model | accuracy | precision | recall | f1 | tp | fp | tn | fn | flagged_modules | flagged_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced LR | 0.715549 ± 0.004045 | 0.347464 ± 0.009655 | 0.536499 ± 0.029638 | 0.421675 ± 0.015988 | 180.8000 ± 9.9880 | 339.2000 ± 7.1386 | 1065.4000 ± 7.0597 | 156.2000 ± 9.9880 | 520.0000 ± 15.8493 | 29.8576 ± 0.9098% |
| Weighted RF | 0.779169 ± 0.010634 | 0.422700 ± 0.029216 | 0.388131 ± 0.043262 | 0.404060 ± 0.034784 | 130.8000 ± 14.5794 | 178.4000 ± 16.2432 | 1226.2000 ± 16.2160 | 206.2000 ± 14.5794 | 309.2000 ± 24.6933 | 17.7537 ± 1.4174% |
| Balanced HGB | 0.735878 ± 0.007552 | 0.368838 ± 0.009452 | 0.512166 ± 0.027428 | 0.428498 ± 0.012311 | 172.6000 ± 9.2434 | 295.6000 ± 19.2104 | 1109.0000 ± 18.4716 | 164.4000 ± 9.2434 | 468.2000 ± 27.0289 | 26.8827 ± 1.5414% |

## Pooled out-of-fold results at threshold 0.50

| model | precision | recall | f1 | accuracy | tp | fp | tn | fn | flagged_modules | flagged_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced LR | 0.347692 | 0.536499 | 0.421937 | 0.715549 | 904 | 1696 | 5327 | 781 | 2600 | 29.8576% |
| Weighted RF | 0.423027 | 0.388131 | 0.404828 | 0.779169 | 654 | 892 | 6131 | 1031 | 1546 | 17.7538% |
| Balanced HGB | 0.368646 | 0.512166 | 0.428713 | 0.735875 | 863 | 1478 | 5545 | 822 | 2341 | 26.8833% |

Pooled confusion counts sum fold counts. Pooled precision/F1 differ slightly
from equal-fold mean precision/F1 because those metrics are nonlinear.

Compared with balanced LR at 0.50, HGB increases mean F1 by 0.006823 and precision
by 2.1374 percentage points, but decreases recall by 2.4332 percentage points.
It catches 41 fewer defects, while generating 218 fewer false positives and
requiring 259 fewer reviews. Its accuracy gain is not the selection criterion.
HGB's F1 exceeds weighted RF's by 0.024438, with 209 more defects caught at this
threshold, but also 586 more false positives and 795 more reviews. These fixed
thresholds produce different workloads, making the capacity comparison essential.

## Precision–recall trade-off at comparable review capacities

For each model and budget, choose the threshold giving the closest attainable
flagged count to `training_rows × budget`, using only its OOF probabilities.
All equal scores are kept together. An equal-distance tie chooses fewer reviews.
The resulting workload can be slightly above or below the requested percentage;
no labels are used to choose thresholds and no arbitrary score ties are broken.
RF's more discrete probabilities cause larger deviations. Report actual workloads
alongside nominal budgets. Thresholds below are rounded to six decimals for
reading; use the full precision in the CSV to reproduce counts.

These are pooled descriptive results, not fold mean/SD for a prespecified
threshold. Choosing a threshold on pooled OOF scores characterizes this training
population's capacity curve; it does not establish a deployment threshold or
provide independent evaluation of a selected threshold. Pooled cross-fold scores
also assume sufficiently comparable score scales across fitted folds.

| model | budget | threshold | flagged_modules | flagged_percent | precision | recall | Defects caught | Defects missed | False positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced LR | 10% | 0.677683 | 871 | 10.0023% | 0.489093 | 0.252819 | 426 | 1259 | 445 |
| Balanced LR | 20% | 0.563412 | 1742 | 20.0046% | 0.405855 | 0.419585 | 707 | 978 | 1035 |
| Balanced LR | 30% | 0.499409 | 2612 | 29.9954% | 0.346861 | 0.537685 | 906 | 779 | 1706 |
| Balanced LR | 40% | 0.456042 | 3483 | 39.9977% | 0.313236 | 0.647478 | 1091 | 594 | 2392 |
| Weighted RF | 10% | 0.592381 | 871 | 10.0023% | 0.473020 | 0.244510 | 412 | 1273 | 459 |
| Weighted RF | 20% | 0.470000 | 1768 | 20.3032% | 0.407805 | 0.427893 | 721 | 964 | 1047 |
| Weighted RF | 30% | 0.380833 | 2602 | 29.8806% | 0.356649 | 0.550742 | 928 | 757 | 1674 |
| Weighted RF | 40% | 0.323333 | 3468 | 39.8254% | 0.314302 | 0.646884 | 1090 | 595 | 2378 |
| Balanced HGB | 10% | 0.684261 | 871 | 10.0023% | 0.483352 | 0.249852 | 421 | 1264 | 450 |
| Balanced HGB | 20% | 0.560689 | 1742 | 20.0046% | 0.396096 | 0.409496 | 690 | 995 | 1052 |
| Balanced HGB | 30% | 0.474457 | 2612 | 29.9954% | 0.353752 | 0.548368 | 924 | 761 | 1688 |
| Balanced HGB | 40% | 0.405062 | 3483 | 39.9977% | 0.314671 | 0.650445 | 1096 | 589 | 2387 |

Full precision–recall curves and every row's OOF probabilities are exported,
not just the four budgets. Average precision summarizes the complete pooled
ranking curve (larger is better; it is not accuracy):

| Model | Average precision |
| --- | --- |
| Balanced LR | 0.392893 |
| Weighted RF | 0.385432 |
| Balanced HGB | 0.387656 |

The ordering changes with capacity:

- **10%:** LR catches 426 defects, HGB 421, RF 412, each with 871 reviews.
  LR has the best tested low-capacity trade-off.
- **20%:** LR beats HGB with exactly 1,742 reviews: 707 versus 690 caught.
  RF catches 721 with 1,768 reviews, so its additional detections involve
  26 extra reviews; this is not an exactly equal-capacity win.
- **30%:** RF catches the most: 928 defects, 757 missed, 1,674 false positives,
  with 2,602 reviews (29.8806%). HGB catches 924 and LR 906, both with 2,612
  reviews (29.9954%). RF wins this measured capacity comparison despite
  requiring ten fewer reviews than HGB and LR.
- **40%:** HGB catches 1,096 versus LR's 1,091 with exactly 3,483 reviews.
  RF catches 1,090 with 3,468 reviews; the differences are small.

Lowering thresholds increases recall and reduces false negatives, while increasing
false positives and review workload. At 30% HGB improves over LR by 18 caught
and 18 fewer missed defects, with 18 fewer false positives at the same workload:
recall rises from 53.7685% to 54.8368%, and precision from 34.6861% to 35.3752%.
At 40%, HGB's advantage over LR is only five defects. LR remains stronger at
10% and 20%, and has higher overall average precision. HGB therefore **outperforms
LR in parts of the measured precision–recall trade-off, but does not dominate it
across the curve**.

## Decision and scope

1. **Best at 30% review capacity: weighted RF**, catching 928 defects, compared
   with HGB's 924 and LR's 906. RF's advantage over HGB is only four defects.
2. **Boosting offers a modest improvement over LR around 30–40% capacity**,
   but loses at 10–20%. The default 0.50 F1 gain is small and is accompanied by
   lower recall. Accuracy alone would obscure these trade-offs.
3. **A limited future tuning experiment is reasonable, not yet justified as a
   replacement.** Its slightly better F1 and equal-budget gains make HGB worth
   investigating if 30–40% review capacity is realistic. The observed gains are
   small relative to fold variation, and no statistical superiority is established.
   There is no evidence here of a large product benefit or universal improvement.

No tuning, model promotion, feature removal, threshold promotion, or final-test
evaluation is implemented in this milestone.

## Reproduction and verification

```sh
python -m pytest -q
python -m defectrisk.gradient_boosting_baseline
```

Complete test suite: **121 passed in 12.07 seconds**. New tests cover exact model
configuration, training-only preprocessing, identical paired folds, single fits
and OOF coverage, contamination rejection before fitting, count arithmetic,
review-budget ties and label independence, invalid inputs, and locked final-test
sentinels. Existing split, CV, and prior experiment tests remain passing.

The current workspace interpreter symlink is broken; execution used the existing
project site-packages with the compatible standalone Python already available
under `/tmp/defectrisk-python/python`, without installing dependencies:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PATH=/tmp/defectrisk-python/python/bin:$PATH \
PYTHONPATH=/workspace/defectrisk-ml/.venv/lib/python3.14/site-packages:/workspace/defectrisk-ml/src \
python -m defectrisk.gradient_boosting_baseline
```

Machine-readable training-only artifacts:

- [Per-fold results](gradient-boosting-results/per-fold.csv)
- [Mean and SD](gradient-boosting-results/summary.csv)
- [Pooled results](gradient-boosting-results/pooled.csv)
- [Fold integrity audits](gradient-boosting-results/audits.csv)
- [Review budget results with full-precision thresholds](gradient-boosting-results/review-budgets.csv)
- [Complete precision–recall curves](gradient-boosting-results/precision-recall.csv)
- [Average precision](gradient-boosting-results/average-precision.csv)
- [Row-aligned OOF probabilities and training labels](gradient-boosting-results/oof-probabilities.csv)
