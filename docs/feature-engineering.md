# Feature quality and interpretable engineering for JM1

Executed 2026-10-09 using Python 3.14 and scikit-learn 1.9.1. This is a
training-only exploratory representation study, without a model or threshold
promotion. The final test partition is never inspected, fitted, predicted, or
scored by the study. No rows are removed or relabeled.

## Scope and independent outer evaluation

Use the existing M3 training partition only: 8,708 rows, 1,685 defective and
7,023 clean, with all 21 original metrics. The final split is unchanged; its test
outputs are discarded by the command.

Reuse the exact five outer stratified group-aware folds and three inner folds
per outer fit partition from the previous tuning experiment. Groups always use
complete **original** feature vectors, excluding the target. Conflicting-label
groups stay intact. The five outer fit/validation sizes are unchanged.

Repeat the previous deterministic limited parameter selection on original
features: eight coverage-first seed-42 sampled configurations plus each unchanged
reference, scored using pooled inner OOF recall at approximately 30% review
capacity, with AP as a tie-breaker. No outer validation labels select parameters.
Then **hold those selected parameters fixed** across all five representations
within each outer fold. This isolates feature changes from additional model
tuning, preserves the original tuned reference, and avoids selecting an engineered
configuration on the evidence used to estimate its performance.

Skew-based log eligibility and correlation-based pruning are fitted using outer
fit features only. Ratio formulas are predetermined and row-local. The full
training audit is descriptive and is not used for label-based feature selection.
There are 270 original-feature inner fits plus 50 outer fits, or 320 model fits.
No inner feature-variant search or additional model family is introduced.

## Feature-quality audit

### Distributions and skew

All 21 metrics have positive skew. Means, medians, upper quantiles, extremes,
missingness, and zero values are shown below; the CSV also includes SD, lower
quartiles, 90th percentile, negative counts, and noninteger counts. Heavy tails
make means and linear correlations sensitive to unusually large modules.
Five rows have missing values in each of the five operator/operand/branch fields;
median imputation remains fitted within each model's training fold.

| index | count | mean | std | min | 50% | 99% | max | skew | missing | zero_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loc | 8708.000000 | 48.177894 | 81.259499 | 1.000000 | 27.000000 | 324.000000 | 3442.000000 | 14.739267 | 0 | 0 |
| v(g) | 8708.000000 | 7.239136 | 13.993622 | 1.000000 | 4.000000 | 50.000000 | 470.000000 | 14.710479 | 0 | 0 |
| ev(g) | 8708.000000 | 3.814469 | 7.305363 | 1.000000 | 1.000000 | 33.000000 | 165.000000 | 7.771473 | 0 | 0 |
| iv(g) | 8708.000000 | 4.495797 | 9.813280 | 1.000000 | 2.000000 | 32.000000 | 402.000000 | 20.708902 | 0 | 0 |
| n | 8708.000000 | 133.248312 | 265.066041 | 0.000000 | 64.000000 | 1178.600000 | 8441.000000 | 10.189081 | 0 | 914 |
| v | 8708.000000 | 788.205472 | 2078.337885 | 0.000000 | 299.560000 | 7934.262400 | 80843.080000 | 15.129067 | 0 | 928 |
| l | 8708.000000 | 0.097410 | 0.102091 | 0.000000 | 0.070000 | 0.500000 | 1.300000 | 2.460325 | 0 | 943 |
| d | 8708.000000 | 16.293693 | 19.368187 | 0.000000 | 11.395000 | 85.494400 | 418.200000 | 6.057496 | 0 | 934 |
| i | 8708.000000 | 33.174453 | 36.144533 | 0.000000 | 25.280000 | 177.225300 | 569.780000 | 5.084571 | 0 | 934 |
| e | 8708.000000 | 42992.281012 | 474884.129577 | 0.000000 | 3483.735000 | 529739.833200 | 31079782.270000 | 43.413652 | 0 | 934 |
| b | 8708.000000 | 0.262966 | 0.692910 | 0.000000 | 0.100000 | 2.648600 | 26.950000 | 15.122583 | 0 | 960 |
| t | 8708.000000 | 2388.460211 | 26382.451511 | 0.000000 | 193.545000 | 29429.990100 | 1726654.570000 | 43.413652 | 0 | 934 |
| lOCode | 8708.000000 | 30.603583 | 63.502612 | 0.000000 | 16.000000 | 234.000000 | 2824.000000 | 17.296995 | 0 | 919 |
| lOComment | 8708.000000 | 3.225425 | 9.738752 | 0.000000 | 0.000000 | 42.000000 | 344.000000 | 10.877908 | 0 | 5308 |
| lOBlank | 8708.000000 | 5.432706 | 10.646720 | 0.000000 | 3.000000 | 43.000000 | 447.000000 | 13.345457 | 0 | 1754 |
| locCodeAndComment | 8708.000000 | 0.437414 | 2.099317 | 0.000000 | 0.000000 | 8.000000 | 108.000000 | 20.759348 | 0 | 7486 |
| uniq_Op | 8703.000000 | 12.484109 | 10.407559 | 0.000000 | 12.000000 | 33.000000 | 411.000000 | 15.978088 | 5 | 913 |
| uniq_Opnd | 8703.000000 | 19.288544 | 28.437679 | 0.000000 | 13.000000 | 109.980000 | 1026.000000 | 13.633823 | 5 | 933 |
| total_Op | 8703.000000 | 79.203056 | 160.934716 | 0.000000 | 38.000000 | 675.940000 | 5420.000000 | 10.998325 | 5 | 913 |
| total_Opnd | 8703.000000 | 54.186970 | 106.695891 | 0.000000 | 25.000000 | 465.980000 | 3021.000000 | 9.170626 | 5 | 933 |
| branchCount | 8703.000000 | 12.996944 | 24.147667 | 1.000000 | 7.000000 | 97.000000 | 826.000000 | 11.295640 | 5 | 0 |

### Pairwise redundancy

Both complete Pearson and Spearman matrices are exported. These pairs have
absolute Pearson or Spearman correlation at least 0.95. Spearman captures
monotonic redundancy despite skew. High correlation does not prove that a
feature contains no incremental information, nor that it is safe to discard.

| left | right | pearson | spearman |
| --- | --- | --- | --- |
| v(g) | branchCount | 0.970031 | 0.995943 |
| n | v | 0.984342 | 0.998911 |
| n | e | 0.720863 | 0.986301 |
| n | b | 0.983966 | 0.997431 |
| n | t | 0.720863 | 0.986300 |
| n | lOCode | 0.944019 | 0.960021 |
| n | total_Op | 0.995754 | 0.997258 |
| n | total_Opnd | 0.989582 | 0.993842 |
| v | e | 0.802894 | 0.984814 |
| v | b | 0.999671 | 0.998512 |
| v | t | 0.802894 | 0.984814 |
| v | lOCode | 0.962664 | 0.961690 |
| v | total_Op | 0.982002 | 0.995679 |
| v | total_Opnd | 0.971659 | 0.993682 |
| d | e | 0.584003 | 0.966738 |
| d | t | 0.584003 | 0.966738 |
| e | b | 0.802764 | 0.983265 |
| e | t | 1.000000 | 1.000000 |
| e | lOCode | 0.814774 | 0.950621 |
| e | total_Op | 0.740519 | 0.987622 |
| e | total_Opnd | 0.677084 | 0.974902 |
| b | t | 0.802764 | 0.983271 |
| b | lOCode | 0.962329 | 0.960171 |
| b | total_Op | 0.981639 | 0.994166 |
| b | total_Opnd | 0.971276 | 0.992291 |
| t | lOCode | 0.814774 | 0.950621 |
| t | total_Op | 0.740519 | 0.987622 |
| t | total_Opnd | 0.677084 | 0.974901 |
| lOCode | total_Op | 0.943731 | 0.958257 |
| lOCode | total_Opnd | 0.928764 | 0.954550 |
| total_Op | total_Opnd | 0.973100 | 0.983326 |

### Univariate association with defects

Report rank-based effect sizes rather than treating highly skewed metrics as
normally distributed. Spearman uses the binary target and feature ranks.
Directional ROC AUC uses larger feature values as higher risk; values below 0.5
indicate the opposite direction. Rank-biserial effect size is `2 × AUC - 1`.
Observed-row class medians provide scale context. Missing feature values are
excluded for that feature's descriptive association, not removed from modeling.

These are row-weighted descriptions, not independent significance tests;
duplicates and correlated predictors preclude interpreting them as causal
importance or proof of incremental multivariate value.

| feature | observed_rows | spearman_target | auc_increasing | rank_biserial | clean_median | defective_median |
| --- | --- | --- | --- | --- | --- | --- |
| loc | 8708 | 0.280214 | 0.704737 | 0.409474 | 24.000000 | 50.000000 |
| v(g) | 8708 | 0.212881 | 0.654625 | 0.309250 | 4.000000 | 6.000000 |
| ev(g) | 8708 | 0.149018 | 0.594108 | 0.188216 | 1.000000 | 1.000000 |
| iv(g) | 8708 | 0.218326 | 0.656608 | 0.313216 | 2.000000 | 4.000000 |
| n | 8708 | 0.165398 | 0.620792 | 0.241585 | 58.000000 | 111.000000 |
| v | 8708 | 0.167995 | 0.622687 | 0.245375 | 266.890000 | 571.590000 |
| l | 8708 | -0.180836 | 0.368112 | -0.263775 | 0.070000 | 0.040000 |
| d | 8708 | 0.128788 | 0.594053 | 0.188105 | 10.770000 | 16.000000 |
| i | 8708 | 0.152701 | 0.611517 | 0.223034 | 23.900000 | 34.280000 |
| e | 8708 | 0.157372 | 0.614928 | 0.229856 | 2942.100000 | 9425.910000 |
| b | 8708 | 0.168848 | 0.623251 | 0.246502 | 0.090000 | 0.190000 |
| t | 8708 | 0.157374 | 0.614930 | 0.229859 | 163.450000 | 523.650000 |
| lOCode | 8708 | 0.174318 | 0.627284 | 0.254567 | 15.000000 | 28.000000 |
| lOComment | 8708 | 0.146820 | 0.594306 | 0.188612 | 0.000000 | 1.000000 |
| lOBlank | 8708 | 0.190968 | 0.638495 | 0.276989 | 3.000000 | 5.000000 |
| locCodeAndComment | 8708 | 0.134320 | 0.559250 | 0.118500 | 0.000000 | 0.000000 |
| uniq_Op | 8703 | 0.139416 | 0.601768 | 0.203537 | 12.000000 | 15.000000 |
| uniq_Opnd | 8703 | 0.170130 | 0.624256 | 0.248512 | 12.000000 | 20.000000 |
| total_Op | 8703 | 0.165552 | 0.620958 | 0.241915 | 35.000000 | 66.000000 |
| total_Opnd | 8703 | 0.161963 | 0.618327 | 0.236653 | 23.000000 | 44.000000 |
| branchCount | 8703 | 0.212648 | 0.654541 | 0.309083 | 7.000000 | 11.000000 |

LOC has the largest absolute Spearman association (0.280214) and increasing-value
AUC 0.704737: clean median 24, defective median 50. No individual association
is particularly strong. Low `l` values are associated with more defects; its
increasing-value AUC below 0.5 reflects this direction.

### `b` provenance and prediction-time safety

The [OpenML 1053 source metadata](https://www.openml.org/api/v1/json/data/1053)
identifies source-code-extracted McCabe/Halstead metrics and a separate observed
defect label. It names `b` only as a Halstead metric, without its extraction
formula or snapshot timing. The [Halstead tool documentation](https://www.verifysoft.com/en_halstead_metrics.html)
describes `B` as a calculated estimate of delivered bugs from code complexity,
not a count of observed bugs; it documents an effort-based formula. These facts
support the general interpretation of `b`, but do not prove which extractor
produced this particular column.

Training-only numerical checks:

| formula | observed_rows | within_rounding_rows | rounding_tolerance | max_absolute_error |
| --- | --- | --- | --- | --- |
| b vs v/3000 | 8708 | 8700 | 0.005100 | 1.299567 |
| b vs e^(2/3)/3000 | 8708 | 1921 | 0.005100 | 8.165539 |
| t vs e/18 | 8708 | 8627 | 0.005100 | 1.227778 |
| n vs total_Op + total_Opnd | 8703 | 8569 | 0.000100 | 482.000000 |

**Inference from the training values:** 8,700 of 8,708 `b` values (99.9081%) match
`v/3000` within 0.0051, consistent with rounding to two decimals. Only 1,921 match
the effort-based alternative within that tolerance. Pearson correlation with
volume is 0.999671. Thus this column appears to be mostly a volume-derived
Halstead bug estimate, rather than a recorded outcome. Eight exceptions mean
it cannot be treated as an exact identity.

**Prediction-time conclusion:** a static estimate recomputed from the available
pre-review code snapshot is appropriate at prediction time. The archived `b`
column is **conditionally safe, not provenance-certified**: there is no evidence
here of direct label leakage, but source metadata does not establish whether
metrics came from code before or after defect correction. Production use should
verify extractor and snapshot provenance or recompute the static estimate.
Original and engineered comparisons retain `b`; the pruning ablation removes it
for redundancy, alongside `t` and `n` when their fit-only correlations qualify.

Other formula deviations are quality flags, not grounds for automatic correction:
81 `t` values exceed rounding tolerance for `e/18`; 134 observed `n` values differ
from `total_Op + total_Opnd`. Existing values and labels are left unchanged.

### Conflicting-label duplicate groups

| training_rows | conflicting_groups_in_training | rows_in_conflicting_training_groups | clean_rows_in_conflicting_groups | defective_rows_in_conflicting_groups | minimum_row_errors_for_identical_vector_predictions |
| --- | --- | --- | --- | --- | --- |
| 8708 | 11 | 22 | 11 | 11 | 11 |

Of the **88 groups previously reported across the whole dataset**, only **11 are
present in M3 training**, containing **22 rows**: 11 clean and 11 defective. The
other groups' row counts are deliberately not recomputed, because that would
require inspecting the locked partition. This report quantifies the training
portion, not the total membership of all 88 groups.

A deterministic classifier using these vectors must give each identical vector
the same prediction. The sum of minority labels within these groups is 11,
which lower-bounds unavoidable row-level classification errors for this training
population at any fixed deterministic decision rule. It does **not** upper-bound
recall at 30% capacity, identify which label is wrong, or establish an information
ceiling. It is only 0.1263% of training rows, so these exact conflicts alone cannot
explain the hundreds of missed defects. No rows are altered or relabeled.

## Small interpretable feature set

| Derived feature | Formula |
| --- | --- |
| branch_density | branchCount / loc |
| complexity_density | v(g) / loc |
| comment_ratio | lOComment / loc |
| blank_ratio | lOBlank / loc |
| operator_density | total_Op / loc |
| operand_density | total_Opnd / loc |
| unique_operator_ratio | uniq_Op / total_Op |
| unique_operand_ratio | uniq_Opnd / total_Opnd |

Safe division produces a missing value for zero/nonpositive denominators,
negative count numerators, missing inputs, or nonfinite inputs. It does not add
an arbitrary epsilon or produce infinity. The existing fold-trained median
imputer handles these missing derived values. Ratios are derived from the raw
counts **before** any log transform. They are intensity/variety proxies; ratios
to `loc` should not be treated as exact fractions of a common line-count total
because the metrics come from different counting conventions.

Representations:

- **Original:** unchanged 21 features.
- **Ratios:** original features plus the eight ratios, 29 total.
- **Logs:** replace eligible count-like columns by `log1p`, retaining 21 columns.
  Eligibility requires nonnegative observed fit values and fit skew at least 2.
  Candidates are LOC, the three McCabe complexities, `n`, four code/comment/blank
  line metrics, four operator/operand counts, and branch count (14 columns).
  Do not log the ratio/difficulty/effort/volume/bug-estimate features. Negative
  validation values in a fitted log column become missing rather than being
  silently clipped to zero.
- **Engineered:** eight ratios plus the same log replacements, 29 total.
- **Pruned:** start from original features and drop `b` versus retained `v`,
  `t` versus retained `e`, and `n` versus retained `total_Op` only when each
  absolute fit-only Pearson correlation is at least 0.99. There is no
  target-driven deletion or blanket removal of weak univariate predictors.

Log transforms introduce no new underlying information. They are monotonic;
axis-aligned trees are often nearly invariant to them, although split thresholds,
rounding, and histogram implementation can change predictions. Ratios represent
interactions more directly, which may help or hinder finite fitted trees.

The same RF parameters are retained for each representation, including
`max_features`; increasing the number of columns changes the resulting candidate
split-feature count naturally. This is part of the representation comparison,
not an additional tuned hyperparameter.

Descriptive ratio associations:

| feature | observed_rows | spearman_target | auc_increasing | rank_biserial | clean_median | defective_median |
| --- | --- | --- | --- | --- | --- | --- |
| branch_density | 8703 | -0.018852 | 0.486218 | -0.027563 | 0.250000 | 0.243243 |
| complexity_density | 8708 | -0.076487 | 0.444115 | -0.111769 | 0.153846 | 0.138889 |
| comment_ratio | 8708 | 0.103729 | 0.566665 | 0.133331 | 0.000000 | 0.005917 |
| blank_ratio | 8708 | 0.045237 | 0.532920 | 0.065839 | 0.111111 | 0.123288 |
| operator_density | 8703 | -0.003186 | 0.497672 | -0.004656 | 1.583333 | 1.594667 |
| operand_density | 8703 | -0.005507 | 0.495977 | -0.008047 | 1.071429 | 1.088889 |
| unique_operator_ratio | 7790 | -0.219985 | 0.336887 | -0.326226 | 0.311111 | 0.198347 |
| unique_operand_ratio | 7770 | -0.136360 | 0.398951 | -0.202098 | 0.514019 | 0.430380 |

## Integrity proof

All 20 original-vector boundaries have zero shared vectors and matched rows.
All 100 additional representation checks (five representations × 20 boundaries)
also have zero overlaps, including the reduced vectors after pruning. Original
feature grouping is never replaced by engineered/reduced grouping. OOF coverage
is exactly once per training row and candidate.

| level | variant | checks | shared_vectors |
| --- | --- | --- | --- |
| inner | engineered | 15 | 0 |
| inner | logs | 15 | 0 |
| inner | original | 15 | 0 |
| inner | pruned | 15 | 0 |
| inner | ratios | 15 | 0 |
| outer | engineered | 5 | 0 |
| outer | logs | 5 | 0 |
| outer | original | 5 | 0 |
| outer | pruned | 5 | 0 |
| outer | ratios | 5 | 0 |

## Controlled results at approximately 30% capacity

| model | recall | precision | f1 | average_precision | tp | fn | fp | flagged_modules | flagged_percent | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weighted_random_forest__original | 0.565579 | 0.364855 | 0.443565 | 0.402211 | 953.000000 | 732.000000 | 1659.000000 | 2612.000000 | 29.995407 | 0.477035 |
| weighted_random_forest__ratios | 0.560831 | 0.361792 | 0.439842 | 0.410143 | 945.000000 | 740.000000 | 1667.000000 | 2612.000000 | 29.995407 | 0.471479 |
| weighted_random_forest__logs | 0.561424 | 0.362175 | 0.440307 | 0.403666 | 946.000000 | 739.000000 | 1666.000000 | 2612.000000 | 29.995407 | 0.478160 |
| weighted_random_forest__engineered | 0.563798 | 0.363706 | 0.442169 | 0.409943 | 950.000000 | 735.000000 | 1662.000000 | 2612.000000 | 29.995407 | 0.471352 |
| weighted_random_forest__pruned | 0.566172 | 0.365237 | 0.444031 | 0.403187 | 954.000000 | 731.000000 | 1658.000000 | 2612.000000 | 29.995407 | 0.480202 |
| hist_gradient_boosting__original | 0.560831 | 0.361792 | 0.439842 | 0.405080 | 945.000000 | 740.000000 | 1667.000000 | 2612.000000 | 29.995407 | 0.524404 |
| hist_gradient_boosting__ratios | 0.562018 | 0.362557 | 0.440773 | 0.409487 | 947.000000 | 738.000000 | 1665.000000 | 2612.000000 | 29.995407 | 0.525300 |
| hist_gradient_boosting__logs | 0.560831 | 0.361792 | 0.439842 | 0.405096 | 945.000000 | 740.000000 | 1667.000000 | 2612.000000 | 29.995407 | 0.524404 |
| hist_gradient_boosting__engineered | 0.562018 | 0.362557 | 0.440773 | 0.409517 | 947.000000 | 738.000000 | 1665.000000 | 2612.000000 | 29.995407 | 0.525300 |
| hist_gradient_boosting__pruned | 0.560831 | 0.361792 | 0.439842 | 0.405074 | 945.000000 | 740.000000 | 1667.000000 | 2612.000000 | 29.995407 | 0.524187 |

`tp` is defects caught, `fn` defects missed, and `fp` clean modules flagged.
Thresholds use the same probability-only closest-count rule as earlier studies,
keeping equal scores together and favoring fewer reviews for equal-distance ties.
Actual workloads are reported. Thresholds here are rounded; raw CSVs retain full
precision. Pooled OOF thresholds describe this population's capacity, without
establishing a deployment threshold. AP means Average Precision over the complete
ranking, not trapezoidal PR integration. No accuracy optimization is performed.

The tuned original-feature results reproduce the previous 953 RF and 945 HGB
caught counts. Selected parameters are identical across feature variants within
each outer fold. Their exact values, selected log columns, and pruned columns
are exported in `feature-choices.csv`.

## Which representations and individual ratios help?

- **weighted_random_forest, ratios:** -8 caught defects, -0.4748 recall points, +0.007932 AP versus original; actual workload 29.9954%.
- **weighted_random_forest, logs:** -7 caught defects, -0.4154 recall points, +0.001455 AP versus original; actual workload 29.9954%.
- **weighted_random_forest, engineered:** -3 caught defects, -0.1780 recall points, +0.007732 AP versus original; actual workload 29.9954%.
- **weighted_random_forest, pruned:** +1 caught defects, +0.0593 recall points, +0.000976 AP versus original; actual workload 29.9954%.
- **hist_gradient_boosting, ratios:** +2 caught defects, +0.1187 recall points, +0.004407 AP versus original; actual workload 29.9954%.
- **hist_gradient_boosting, logs:** +0 caught defects, +0.0000 recall points, +0.000016 AP versus original; actual workload 29.9954%.
- **hist_gradient_boosting, engineered:** +2 caught defects, +0.1187 recall points, +0.004437 AP versus original; actual workload 29.9954%.
- **hist_gradient_boosting, pruned:** +0 caught defects, +0.0000 recall points, -0.000007 AP versus original; actual workload 29.9954%.


Held-out permutation sensitivity for the combined representation is also
reported. Each derived ratio is shuffled independently three times within each
outer validation fold, without refitting. Report the mean across folds of the
original-minus-shuffled recall at approximately 30% and AP, with population SD
of fold means. Positive drops indicate reliance on that ratio in the fitted
model, not a causal effect or proof that adding it improves overall performance.
Correlated ratios can hide or redistribute importance; independently shuffling
a ratio creates combinations inconsistent with its original source columns.
These measurements never select a feature set or model.

| model | feature | recall_drop_mean | recall_drop_std | ap_drop_mean | ap_drop_std |
| --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting | blank_ratio | 0.006133 | 0.007543 | 0.004196 | 0.003549 |
| hist_gradient_boosting | branch_density | -0.001583 | 0.005293 | 0.003804 | 0.003565 |
| hist_gradient_boosting | comment_ratio | -0.001583 | 0.005181 | 0.001727 | 0.001844 |
| hist_gradient_boosting | complexity_density | 0.000791 | 0.007723 | 0.001594 | 0.000899 |
| hist_gradient_boosting | operand_density | 0.010880 | 0.009404 | 0.009250 | 0.002605 |
| hist_gradient_boosting | operator_density | 0.000198 | 0.005360 | 0.000855 | 0.001362 |
| hist_gradient_boosting | unique_operand_ratio | 0.011078 | 0.007144 | 0.003305 | 0.003970 |
| hist_gradient_boosting | unique_operator_ratio | 0.001780 | 0.006142 | 0.000922 | 0.000923 |
| weighted_random_forest | blank_ratio | 0.002572 | 0.004083 | 0.004805 | 0.003332 |
| weighted_random_forest | branch_density | -0.001583 | 0.006983 | 0.003418 | 0.002241 |
| weighted_random_forest | comment_ratio | 0.001187 | 0.005286 | 0.000572 | 0.001628 |
| weighted_random_forest | complexity_density | -0.001978 | 0.005968 | 0.002378 | 0.003050 |
| weighted_random_forest | operand_density | 0.008506 | 0.008147 | 0.006246 | 0.004291 |
| weighted_random_forest | operator_density | -0.001187 | 0.004073 | 0.000100 | 0.002820 |
| weighted_random_forest | unique_operand_ratio | 0.008902 | 0.010835 | 0.001499 | 0.003606 |
| weighted_random_forest | unique_operator_ratio | 0.000989 | 0.003805 | 0.000872 | 0.003540 |

## Answers and next direction

1. **Did better representation materially improve performance? No.** Combined
   features change tuned RF from 953 to 950 caught defects (−3, −0.1780 recall
   points), and tuned HGB from 945 to 947 (+2, +0.1187 recall points), at exactly
   the same 2,612 reviews. AP improves by 0.007732 for RF and 0.004437 for HGB,
   suggesting a modest ranking change across the complete curve, but not a
   useful improvement at the specified capacity. Ratio-only RF loses eight
   detections; log-only RF loses seven. HGB log replacement leaves its capacity
   counts unchanged. These gains are smaller than the prior tuning gains of
   +25 RF and +21 HGB detections over their untuned/current references.
2. **Which engineered features appear useful?** Operand density has the largest
   mean held-out AP permutation drop in both models (0.006246 RF, 0.009250 HGB).
   Unique operand ratio has a larger recall-at-capacity sensitivity than most
   other ratios (0.008902 RF, 0.011078 HGB). Blank-line ratio has some additional
   sensitivity. Unique operator ratio has the strongest univariate rank
   association among ratios, but small conditional permutation drops. Thus
   apparent association and incremental fitted-model reliance differ. None is
   independently proven to improve nested performance when added alone; the
   combined representation does not materially improve the product objective.
   Logs do not help here, consistent with the near invariance of trees to
   monotonic transformations.
3. **Does removing redundant/noisy features help? Only negligibly.** Fit-only
   pruning drops `b`, `t`, and `n` in every outer fold. RF catches 954 (+1 over
   original), with AP 0.403187 (+0.000976). HGB catches the same 945, with almost
   unchanged AP. The three columns were dropped together, so these results do
   not isolate the effect of `b`. Weak target association alone is not used as
   a reason to discard any feature.
4. **Are we reaching a data-information ceiling? Possibly a practical plateau
   for these static metrics, but not a proven ceiling.** Ratios and monotonic
   transforms are deterministic functions of the same source metrics; they do
   not supply new information. Small gains across tuned nonlinear models and
   representation changes suggest limited additional value from rearranging
   these columns. Even the best observed result, pruned RF, still misses 731
   of 1,685 defects (43.38%). The 11 unavoidable exact-vector classification
   errors are too few to explain this gap. Broader label noise, snapshot timing,
   missing contextual predictors, and generalization error remain possible.
5. **Prefer better data to another model-family escalation next.** Investigate
   source-code snapshot and label alignment, extractor/formula inconsistencies,
   and label provenance before spending more effort on XGBoost/CatBoost or
   additional tuning. Seek genuinely new predictive context, such as code-change
   history, churn, ownership, dependencies, and testing context available before
   review. Do not infer that XGBoost/CatBoost could never help; this evidence
   simply does not make another tree-family switch the strongest next priority.

The single extra detection from pruning does not justify promoting that
representation. No columns are permanently removed, no rows are relabeled, no
model or threshold is promoted, and no stronger model family is implemented.

## Per-fold evidence and limitations

Fold-specific probability-only 30% budgets are reported separately; summed fold
counts need not match pooled counts because their thresholds differ. Detailed
per-fold metrics, fold mean/SD, nested search scores, and every outer OOF probability
are exported. Comparing several representations on outer evidence remains
exploratory; choosing the best after seeing these results adds selection optimism.
No statistical superiority, global feature-importance ranking, causality, or
provable data-information ceiling is claimed. Nested fit partitions overlap and
five folds are a limited basis for uncertainty estimates.

## Reproduction and tests

```sh
python -m pytest -q
python -m defectrisk.feature_engineering
```

Complete suite: **136 passed in 20.11 seconds**. New tests cover safe division,
raw-ratio derivation, fit-only skew/pruning rules, label-independent transforms,
schema/target exclusion, descriptive rank associations and conflicting counts,
paired nested parameter reuse, fold-trained imputation, new reduced-vector
collision rejection, OOF count arithmetic, input preservation, and final-test
sentinels. All existing behavior remains tested.

As in previous runs, the workspace interpreter symlink is broken; execution uses
the compatible standalone Python with the existing environment's packages:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PATH=/tmp/defectrisk-python/python/bin:$PATH \
PYTHONPATH=/workspace/defectrisk-ml/.venv/lib/python3.14/site-packages:/workspace/defectrisk-ml/src \
python -m defectrisk.feature_engineering
```

Training-only artifacts:

- [Feature distributions](feature-engineering-results/distributions.csv)
- [Pearson matrix](feature-engineering-results/pearson.csv) and [Spearman matrix](feature-engineering-results/spearman.csv)
- [Highly correlated pairs](feature-engineering-results/redundant-pairs.csv)
- [Original-feature target associations](feature-engineering-results/target-associations.csv)
- [Ratio target associations](feature-engineering-results/ratio-associations.csv)
- [Static formula checks](feature-engineering-results/formula-checks.csv)
- [Conflict summary](feature-engineering-results/conflict-summary.json) and [training conflict-group label counts](feature-engineering-results/conflicting-groups.csv)
- [Original-vector audits](feature-engineering-results/audits.csv) and [representation audits](feature-engineering-results/representation-audits.csv)
- [Pooled 30% results](feature-engineering-results/pooled.csv)
- [Per-fold results](feature-engineering-results/per-fold.csv) and [mean/SD](feature-engineering-results/summary.csv)
- [Inner search evidence](feature-engineering-results/inner-search.csv)
- [Parameters and feature choices](feature-engineering-results/feature-choices.csv)
- [Held-out ratio permutation sensitivity](feature-engineering-results/ratio-permutation.csv)
- [Outer OOF probabilities, training labels, and fold assignment](feature-engineering-results/oof-probabilities.csv)
