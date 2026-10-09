# Focused training-only ensemble ranking study

Executed 2026-10-09. Compare a fixed small set of arithmetic probability averages
at approximately 30% review capacity. No models are refitted, no new hyperparameter
or weight search is run, and no new model family is introduced. The previously
selected RF model and its frozen configuration are unchanged.

## Data, models, and exact trusted folds

Use only the unchanged M3 training partition: 8,708 rows, 1,685 defective and
7,023 clean. Discard final-test outputs immediately. The study does not load,
inspect, predict, or score final-test artifacts or the held-out fitted pipeline.

Re-export the three base-model out-of-fold probabilities from the saved trusted
nested experiments. This reuses their already generated held-out predictions,
rather than unnecessarily repeating model training:

- RF: `tuned_weighted_random_forest` from the tree-tuning experiment.
- HGB: `tuned_hist_gradient_boosting` from the same experiment.
- XGBoost: full original-feature `xgboost` from the final model challenge.
  This uses its inner-CV-selected main candidate; the `b` exclusion was a diagnostic
  ablation, not a promoted configuration or another ensemble branch.

Each base model's parameters were selected only inside its outer training fold
using the existing three inner stratified group-aware folds. Outer predictions
come from the exact same five `StratifiedGroupKFold` folds (`shuffle=True`,
`random_state=42`), with complete original feature vectors defining groups and
the target excluded. Preprocessing and imbalance handling remain those of the
original nested experiments. RF and HGB may have fold-specific selected settings;
these predictions estimate their trusted selection procedures, not the
once-fitted held-out RF artifact.

Before combining probabilities, verify complete M3 row/index alignment, training
labels, exact reconstructed outer-fold assignment, finite probabilities within
[0,1], and identical RF probabilities in both source files. Each training row
has exactly one OOF score per model. All five reconstructed original-vector
fit/validation audits have zero shared vectors and matching rows/pairs. The
original nested experiments documented zero overlap at their inner boundaries.
No duplicate row or label is altered.

Source paths and SHA-256 hashes, fixed weights, and zero new-fit count are recorded
in [manifest.json](ensemble-ranking-results/manifest.json). The new
[ensemble-oof.csv](ensemble-ranking-results/ensemble-oof.csv) contains all aligned
base and ensemble probabilities, training labels, and fold IDs.

## Fixed ensemble strategies and review policy

Six unique strategies are specified before looking at results:

| Strategy | RF weight | HGB weight | XGB weight |
| --- | ---: | ---: | ---: |
| Simple three-model average | 1/3 | 1/3 | 1/3 |
| RF/HGB average | 0.5 | 0.5 | 0 |
| RF/XGB average | 0.5 | 0 | 0.5 |
| HGB/XGB average | 0 | 0.5 | 0.5 |
| RF-heavy combination | 0.5 | 0.25 | 0.25 |
| More RF-heavy combination | 0.6 | 0.2 | 0.2 |

`0.5 RF + 0.5 XGB` duplicates the pairwise RF/XGB average and is evaluated only
once. There is no stacking, recalibration, rank normalization, fitted weight
optimization, or additional candidate search.

For each score vector, apply the existing probability-only closest-count rule:
choose the score threshold giving the closest attainable review count to 30% of
training rows. Equal scores stay together; equal-distance ties prefer fewer
reviews. Target labels measure outcomes but do not choose review thresholds.
All candidates below flag exactly **2,612 modules (29.995407%)**.

Meaningful improvement is at least +2 absolute recall percentage points **or**
30 additional defects caught, at comparable workload, without clearly worse AP.
As in the final model challenge, conservatively require AP at least RF AP;
review percentages must be within 0.5 points of 30% and each other. No strategy
qualifies, so the precise AP deterioration tolerance does not affect the decision.
Accuracy plays no role in ranking candidates or making the decision.

## Complete pooled OOF results

TP is defects caught; FN is defects missed. FP is clean modules sent for review,
and TN is clean modules not flagged. AP is Average Precision over the complete
ranking, not trapezoidal PR integration. Thresholds below are rounded; exports
retain full precision. These are pooled training-only results, not final-test
results or a promoted deployment threshold.

| model | flagged_percent | recall | precision | f1 | average_precision | tp | fp | tn | fn | flagged_modules | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tuned RF | 29.995407 | 0.565579 | 0.364855 | 0.443565 | 0.402211 | 953.000000 | 1659.000000 | 5364.000000 | 732.000000 | 2612.000000 | 0.477035 |
| Tuned HGB | 29.995407 | 0.560831 | 0.361792 | 0.439842 | 0.405080 | 945.000000 | 1667.000000 | 5356.000000 | 740.000000 | 2612.000000 | 0.524404 |
| XGBoost | 29.995407 | 0.549555 | 0.354518 | 0.430998 | 0.402409 | 926.000000 | 1686.000000 | 5337.000000 | 759.000000 | 2612.000000 | 0.532520 |
| (RF + HGB + XGB) / 3 | 29.995407 | 0.562611 | 0.362940 | 0.441238 | 0.408080 | 948.000000 | 1664.000000 | 5359.000000 | 737.000000 | 2612.000000 | 0.509755 |
| (RF + HGB) / 2 | 29.995407 | 0.569733 | 0.367534 | 0.446823 | 0.408110 | 960.000000 | 1652.000000 | 5371.000000 | 725.000000 | 2612.000000 | 0.499344 |
| (RF + XGB) / 2 | 29.995407 | 0.560831 | 0.361792 | 0.439842 | 0.406652 | 945.000000 | 1667.000000 | 5356.000000 | 740.000000 | 2612.000000 | 0.501842 |
| (HGB + XGB) / 2 | 29.995407 | 0.558457 | 0.360260 | 0.437980 | 0.406074 | 941.000000 | 1671.000000 | 5352.000000 | 744.000000 | 2612.000000 | 0.528859 |
| 0.50 RF + 0.25 HGB + 0.25 XGB | 29.995407 | 0.566172 | 0.365237 | 0.444031 | 0.407815 | 954.000000 | 1658.000000 | 5365.000000 | 731.000000 | 2612.000000 | 0.500668 |
| 0.60 RF + 0.20 HGB + 0.20 XGB | 29.995407 | 0.564985 | 0.364472 | 0.443100 | 0.407076 | 952.000000 | 1660.000000 | 5363.000000 | 733.000000 | 2612.000000 | 0.494989 |

Meaningfulness versus RF:

| ensemble | additional_defects | recall_gain | ap_change | meaningful |
| --- | --- | --- | --- | --- |
| (RF + HGB + XGB) / 3 | -5 | -0.002967 | 0.005869 | False |
| (RF + HGB) / 2 | 7 | 0.004154 | 0.005899 | False |
| (RF + XGB) / 2 | -8 | -0.004748 | 0.004441 | False |
| (HGB + XGB) / 2 | -12 | -0.007122 | 0.003863 | False |
| 0.50 RF + 0.25 HGB + 0.25 XGB | 1 | 0.000593 | 0.005604 | False |
| 0.60 RF + 0.20 HGB + 0.20 XGB | -1 | -0.000593 | 0.004865 | False |

## Fold variation

Each fold uses its own probability-only approximately-30% cutoff; pooled and
fold-specific counts need not sum to the same totals. Equal-fold mean and population
SD (`ddof=0`) are descriptive, not confidence intervals.

| Model | recall | precision | f1 | average_precision | flagged_percent |
| --- | --- | --- | --- | --- | --- |
| Tuned RF | 0.562611 ± 0.022094 | 0.362946 ± 0.014399 | 0.441242 ± 0.017435 | 0.408067 ± 0.025582 | 29.995401 ± 0.016402 |
| Tuned HGB | 0.560237 ± 0.026700 | 0.361266 ± 0.017129 | 0.439271 ± 0.020869 | 0.410957 ± 0.028352 | 30.006889 ± 0.022509 |
| XGBoost | 0.553709 ± 0.023783 | 0.357073 ± 0.015626 | 0.434165 ± 0.018862 | 0.408846 ± 0.026530 | 30.006889 ± 0.022509 |
| (RF + HGB + XGB) / 3 | 0.561424 ± 0.022552 | 0.362174 ± 0.014534 | 0.440306 ± 0.017675 | 0.413524 ± 0.026028 | 29.995401 ± 0.016402 |
| (RF + HGB) / 2 | 0.567953 ± 0.025705 | 0.366388 ± 0.016643 | 0.445429 ± 0.020203 | 0.413143 ± 0.027615 | 29.995401 ± 0.016402 |
| (RF + XGB) / 2 | 0.559050 ± 0.023857 | 0.360646 ± 0.015453 | 0.438447 ± 0.018756 | 0.412064 ± 0.026222 | 29.995401 ± 0.016402 |
| (HGB + XGB) / 2 | 0.557270 ± 0.025896 | 0.359495 ± 0.016715 | 0.437049 ± 0.020315 | 0.411797 ± 0.026153 | 29.995401 ± 0.016402 |
| 0.50 RF + 0.25 HGB + 0.25 XGB | 0.565579 ± 0.024513 | 0.364856 ± 0.015849 | 0.443566 ± 0.019250 | 0.413202 ± 0.026553 | 29.995401 ± 0.016402 |
| 0.60 RF + 0.20 HGB + 0.20 XGB | 0.566766 ± 0.019047 | 0.365623 ± 0.012387 | 0.444498 ± 0.015010 | 0.412523 ± 0.026931 | 29.995401 ± 0.016402 |

## Ranking overlap and diversity

Unique means caught at that model’s own 30% budget and missed by **both** other
base models. Pairwise complement counts are shown separately.

| model | caught | uniquely_caught_among_three | caught_not_by_rf | rf_caught_not_by_model |
| --- | --- | --- | --- | --- |
| Tuned RF | 953 | 62 | 0 | 0 |
| Tuned HGB | 945 | 30 | 81 | 89 |
| XGBoost | 926 | 27 | 78 | 105 |

All eight defective-row catch patterns:

| rf | hgb | xgb | defects |
| --- | --- | --- | --- |
| False | False | False | 624 |
| False | True | True | 51 |
| True | True | True | 821 |
| True | False | True | 27 |
| False | True | False | 30 |
| True | False | False | 62 |
| False | False | True | 27 |
| True | True | False | 43 |

Pairwise score and review overlap:

| left | right | shared_flagged | flagged_union | flagged_jaccard | shared_caught_defects | probability_spearman |
| --- | --- | --- | --- | --- | --- | --- |
| Tuned RF | Tuned HGB | 2239 | 2985 | 0.750084 | 864 | 0.911033 |
| Tuned RF | XGBoost | 2211 | 3013 | 0.733820 | 848 | 0.901936 |
| Tuned HGB | XGBoost | 2354 | 2870 | 0.820209 | 872 | 0.957312 |

The three models jointly catch 1,061 distinct defects at their separate 30%
budgets; 821 defects are caught by all three, and 624 are missed by all three.
This union is not a deployable 30%-capacity result: reviewing the union increases
workload. The highly correlated rankings (Spearman 0.902–0.957) also show that
these models are far from independent.

HGB catches 81 defective modules RF misses, but misses 89 that RF catches.
XGBoost catches 78 RF misses, but misses 105 RF catches. Diversity is real at
the level of different held-out detections, yet that alone does not ensure a
better ranking at a fixed review capacity.

For every ensemble, compare newly caught defective rows with previously
RF-caught rows lost:

| ensemble | new_defects_vs_rf | rf_defects_lost | new_caught_by_other_base |
| --- | --- | --- | --- |
| (RF + HGB + XGB) / 3 | 54 | 59 | 54 |
| (RF + HGB) / 2 | 50 | 43 | 50 |
| (RF + XGB) / 2 | 43 | 51 | 43 |
| (HGB + XGB) / 2 | 75 | 87 | 75 |
| 0.50 RF + 0.25 HGB + 0.25 XGB | 45 | 44 | 45 |
| 0.60 RF + 0.20 HGB + 0.20 XGB | 35 | 36 | 35 |

## Answers and selected model

1. **Best ensemble: the equal RF/HGB average.** It catches 960 defects, misses
   725, and produces 1,652 false positives at 2,612 reviews. Recall is 0.569733,
   precision 0.367534, F1 0.446823, and AP 0.408110.
2. **It does not materially beat tuned RF.** The gain is seven detections
   (+0.4154 absolute recall percentage points), below both the +30-detection
   and +2-point requirements. Its AP increase is 0.005899; accuracy is not the
   reason for selecting it as the best ensemble.
3. **Diversity helps slightly, not materially.** RF/HGB averaging newly catches
   50 defects missed by RF; all 50 were caught by at least one other base model
   at its own budget. It simultaneously loses 43 RF-caught defects, leaving
   +7 net detections. This supports actual complementary detections, but does
   not prove a causal mechanism or robust gain under another dataset/split.
4. **Do not replace tuned RF with the ensemble.** No fixed combination meets
   the predeclared meaningful-improvement rule. Retain the selected RF and its
   existing frozen configuration. No retraining, deployment, or model promotion
   is performed, and no further ensemble search is proposed or implemented.

## Evidence limits after the held-out milestone

This experiment was requested after the held-out test had already been used.
It is a post-held-out exploratory study using training evidence only. No held-out
metrics, labels, predictions, or artifacts enter the ensemble calculations.
The previous held-out evaluation therefore remains evidence for the previously
frozen RF, not validation of a newly chosen ensemble. A replacement would need
new untouched evaluation data; the existing test cannot be reused to validate it.

Each fixed ensemble uses valid base OOF predictions, but choosing the best of
six strategies on those same outcomes introduces winner-selection optimism.
Weights were not separately nested-selected. The small observed improvement
is not an independently estimated gain from a tuned ensemble-selection procedure.
Five folds share training data; fold SD is not a significance test. Arithmetic
averaging also assumes sufficiently comparable probability scales across models
and folds. None of these limitations changes the decision to retain RF here.

## Reproduction and complete tests

```sh
python -m defectrisk.ensemble_ranking
python -m pytest -q
```

Complete suite: **168 passed in 22.60 seconds**. Added tests cover fixed average
arithmetic and duplicate-strategy exclusion; invalid weights; exact source
row/label/fold alignment and RF agreement; corrupted/missing probabilities;
OOF coverage and metrics/count arithmetic; exclusive catch patterns, overlaps,
new/lost detections, population SD; and locked final-test sentinels. Existing
tests remain passing; no real held-out model is refitted or rescored by them.

The workspace interpreter symlink remains broken. Execution used the compatible
standalone Python and existing packages, with no new dependency:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PATH=/tmp/defectrisk-python/python/bin:$PATH \
PYTHONPATH=/workspace/defectrisk-ml/.venv/lib/python3.14/site-packages:/workspace/defectrisk-ml/src \
python -m defectrisk.ensemble_ranking
```

Training-only exports:

- [Pooled results](ensemble-ranking-results/pooled.csv)
- [Per-fold results](ensemble-ranking-results/per-fold.csv)
- [Fold mean/SD](ensemble-ranking-results/summary.csv)
- [Fold integrity audits](ensemble-ranking-results/audits.csv)
- [Unique defective catches](ensemble-ranking-results/unique-defects.csv)
- [Defective catch patterns](ensemble-ranking-results/defect-patterns.csv)
- [Ranking overlap](ensemble-ranking-results/ranking-overlap.csv)
- [New and lost catches versus RF](ensemble-ranking-results/ensemble-gains.csv)
- [Meaningful-improvement checks](ensemble-ranking-results/meaningfulness.csv)
- [All training OOF scores](ensemble-ranking-results/ensemble-oof.csv)
- [Source hashes and fixed strategy definitions](ensemble-ranking-results/manifest.json)
