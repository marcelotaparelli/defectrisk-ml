# Targeted nested tree-model tuning at 30% review capacity

Executed 2026-10-09, using Python 3.14 and scikit-learn 1.9.1. This experiment
estimates how many defects can be caught when approximately 30% of modules can
be reviewed. Accuracy is not used to select hyperparameters or the preferred
model. No model or threshold is promoted.

## Data and untouched final test

Only the existing M3 training partition is passed to the experiment: 8,708 rows,
1,685 defective and 7,023 clean. The existing group-aware 80/20 split is unchanged.
The command discards final-test outputs immediately, without inspection, fitting,
prediction, scoring, or threshold selection. All 21 features and all duplicate
rows remain. Groups use complete original feature vectors only, excluding the
target; groups with conflicting labels are kept together.

## Nested evaluation and limited search

1. Reuse the exact five outer `StratifiedGroupKFold` folds from previous
   experiments (`shuffle=True, random_state=42`). These folds estimate performance.
2. Within each outer fit partition, construct three new stratified group-aware
   inner folds (`shuffle=True, random_state=42`), explicitly passing raw feature
   groups into the splitter. Inner folds never see outer validation rows.
3. For each tree family, fit nine candidates on the inner folds: eight sampled
   configurations plus the unchanged current reference as a guardrail.
   Generate pooled inner OOF probabilities, then select by **recall at approximately
   30% review capacity**. Average precision breaks recall ties; stable candidate
   order breaks any remaining tie. Accuracy is never part of selection.
4. Refit the selected configuration on that outer fit partition only, then
   predict its untouched outer validation rows. Repeat for all five outer folds.
   Each tuned candidate therefore represents a **selection procedure**, with
   potentially different selected parameters per fold; it is not a single
   configuration chosen using outer results.
5. Fit the three unchanged references on the same outer folds. Collect exactly
   one held-out defective probability per training row for all five candidates.
6. Pool outer OOF probabilities and apply the same review-budget rule as before:
   choose the score threshold whose flagged count is closest to 30% of rows,
   using probabilities alone. Keep equal scores together; equal-distance ties
   prefer fewer reviews. Report the actual review percentage.

RF's requested space has 72 combinations; HGB's has 72. The deterministic
limited strategy builds a seed-42 shuffled order of each grid, selects candidates
covering the most as-yet-uncovered parameter values first, then fills the remaining
slots in seeded order. **Every requested value is represented**, without evaluating
all combinations. The sample is fixed before seeing any outcomes and reused in
every outer fold. The unchanged RF reference has 100 trees and is an additional
guardrail, not a new searched value.

The run requires 270 inner fits (5 outer × 2 families × 9 candidates × 3 inner),
plus 25 outer fits, for **295 total fits**. Two candidate workers limit concurrent
computation; OpenMP and BLAS threads are capped at one during execution.

All imputation/scaling is fitted inside the relevant training fold. Current
balanced LR remains median imputer → StandardScaler →
`LogisticRegression(random_state=42, max_iter=5000, class_weight="balanced", C=1.0)`.
RF retains its existing median imputation and scaling pipeline; only the requested
RF parameters vary, with `random_state=42` unchanged. HGB retains median imputation,
no scaler, `class_weight="balanced"`, `random_state=42`, and
`early_stopping=False`, avoiding an internal random validation boundary. Only the
requested HGB parameters vary. Imbalance weights are computed from each fit
partition, including per-bootstrap training data for `balanced_subsample`.
No resampling, feature removal, new dependency, or final full-training refit is done.

## Fixed sampled configurations

Candidate 0 is `{}`: use the unchanged current reference configuration. The
following eight non-reference candidates are identical across outer folds.

### Random Forest

| candidate | class_weight | max_depth | max_features | min_samples_leaf | n_estimators |
| --- | --- | --- | --- | --- | --- |
| 1 | balanced | None | sqrt | 5 | 200 |
| 2 | balanced_subsample | 16.0 | 0.7 | 3 | 500 |
| 3 | balanced | 8.0 | 0.7 | 1 | 200 |
| 4 | balanced_subsample | 16.0 | sqrt | 3 | 200 |
| 5 | balanced | None | sqrt | 1 | 200 |
| 6 | balanced | 16.0 | sqrt | 5 | 200 |
| 7 | balanced_subsample | 8.0 | sqrt | 3 | 200 |
| 8 | balanced | None | 0.7 | 5 | 200 |

### HistGradientBoosting

| candidate | l2_regularization | learning_rate | max_iter | max_leaf_nodes | min_samples_leaf |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0 | 0.03 | 100 | 31 | 20 |
| 2 | 1.0 | 0.1 | 300 | 63 | 10 |
| 3 | 1.0 | 0.03 | 300 | 15 | 40 |
| 4 | 1.0 | 0.1 | 100 | 63 | 40 |
| 5 | 0.0 | 0.1 | 100 | 15 | 10 |
| 6 | 0.0 | 0.03 | 100 | 15 | 10 |
| 7 | 0.0 | 0.1 | 300 | 15 | 20 |
| 8 | 1.0 | 0.03 | 300 | 31 | 40 |

## Evaluation-boundary audits

All five outer and all fifteen inner boundaries have zero shared raw feature
vectors, zero matched rows on either side, and zero matching row pairs. Coverage
is checked before fitting: every row receives exactly one validation prediction
within each CV level.

| level | outer_fold | fold | fit_rows | validation_rows | shared_feature_vectors | left_rows_with_match | right_rows_with_match | matching_row_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outer | 0 | 1 | 6966 | 1742 | 0 | 0 | 0 | 0 |
| outer | 0 | 2 | 6967 | 1741 | 0 | 0 | 0 | 0 |
| outer | 0 | 3 | 6967 | 1741 | 0 | 0 | 0 | 0 |
| outer | 0 | 4 | 6965 | 1743 | 0 | 0 | 0 | 0 |
| outer | 0 | 5 | 6967 | 1741 | 0 | 0 | 0 | 0 |
| inner | 1 | 1 | 4644 | 2322 | 0 | 0 | 0 | 0 |
| inner | 1 | 2 | 4644 | 2322 | 0 | 0 | 0 | 0 |
| inner | 1 | 3 | 4644 | 2322 | 0 | 0 | 0 | 0 |
| inner | 2 | 1 | 4644 | 2323 | 0 | 0 | 0 | 0 |
| inner | 2 | 2 | 4645 | 2322 | 0 | 0 | 0 | 0 |
| inner | 2 | 3 | 4645 | 2322 | 0 | 0 | 0 | 0 |
| inner | 3 | 1 | 4644 | 2323 | 0 | 0 | 0 | 0 |
| inner | 3 | 2 | 4645 | 2322 | 0 | 0 | 0 | 0 |
| inner | 3 | 3 | 4645 | 2322 | 0 | 0 | 0 | 0 |
| inner | 4 | 1 | 4644 | 2321 | 0 | 0 | 0 | 0 |
| inner | 4 | 2 | 4643 | 2322 | 0 | 0 | 0 | 0 |
| inner | 4 | 3 | 4643 | 2322 | 0 | 0 | 0 | 0 |
| inner | 5 | 1 | 4644 | 2323 | 0 | 0 | 0 | 0 |
| inner | 5 | 2 | 4645 | 2322 | 0 | 0 | 0 | 0 |
| inner | 5 | 3 | 4645 | 2322 | 0 | 0 | 0 | 0 |

## Selected parameters per outer fold

| outer_fold | model | parameters |
| --- | --- | --- |
| 1 | Tuned weighted RF | {'class_weight': 'balanced', 'max_depth': 8, 'max_features': 0.7, 'min_samples_leaf': 1, 'n_estimators': 200} |
| 1 | Tuned balanced HGB | {'l2_regularization': 0.0, 'learning_rate': 0.03, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 10} |
| 2 | Tuned weighted RF | {'class_weight': 'balanced_subsample', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'n_estimators': 200} |
| 2 | Tuned balanced HGB | {'l2_regularization': 0.0, 'learning_rate': 0.03, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 10} |
| 3 | Tuned weighted RF | {'class_weight': 'balanced', 'max_depth': 16, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 200} |
| 3 | Tuned balanced HGB | {'l2_regularization': 0.0, 'learning_rate': 0.03, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 10} |
| 4 | Tuned weighted RF | {'class_weight': 'balanced', 'max_depth': None, 'max_features': 0.7, 'min_samples_leaf': 5, 'n_estimators': 200} |
| 4 | Tuned balanced HGB | {'l2_regularization': 0.0, 'learning_rate': 0.03, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 10} |
| 5 | Tuned weighted RF | {'class_weight': 'balanced_subsample', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'n_estimators': 200} |
| 5 | Tuned balanced HGB | {'l2_regularization': 0.0, 'learning_rate': 0.03, 'max_iter': 100, 'max_leaf_nodes': 15, 'min_samples_leaf': 10} |

Selections above use inner OOF evidence only. They are not permanently adopted.

## Pooled outer OOF results at approximately 30% capacity

| model | threshold | recall | precision | f1 | tp | fn | fp | flagged_modules | flagged_percent | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced LR | 0.499409 | 0.537685 | 0.346861 | 0.421690 | 906 | 779 | 1706 | 2612 | 29.9954% | 0.392893 |
| Current weighted RF | 0.380833 | 0.550742 | 0.356649 | 0.432937 | 928 | 757 | 1674 | 2602 | 29.8806% | 0.385432 |
| Tuned weighted RF | 0.477035 | 0.565579 | 0.364855 | 0.443565 | 953 | 732 | 1659 | 2612 | 29.9954% | 0.402211 |
| Current balanced HGB | 0.474457 | 0.548368 | 0.353752 | 0.430067 | 924 | 761 | 1688 | 2612 | 29.9954% | 0.387656 |
| Tuned balanced HGB | 0.524404 | 0.560831 | 0.361792 | 0.439842 | 945 | 740 | 1667 | 2612 | 29.9954% | 0.405080 |

Thresholds in this table are rounded; CSVs retain full precision. `tp` is defects
caught, `fn` defects missed, and `fp` clean modules sent for review. Average
Precision (AP) summarizes the complete precision–recall ranking curve without
selecting a threshold; it is not trapezoidal PR integration. Accuracy is included
in raw exports for completeness but plays no role in selection or conclusions.

These pooled thresholds describe the capacity of this OOF population, not a
validated production threshold. Pooling assumes reasonably comparable probability
scales across folds. Outer validation labels do not select hyperparameters or
capacity thresholds, but are used to measure outcomes. This provides separation
between hyperparameter selection and performance estimation; historical model
experiments and comparison among model families still mean this is exploratory
training-partition evidence, not an independent final confirmation.

## Per-outer-fold capacity results

Each row uses a probability-only threshold for approximately 30% of that fold,
so summed per-fold counts need not equal counts from the pooled threshold above.

| model | outer_fold | threshold | recall | precision | f1 | tp | fn | fp | flagged_modules | flagged_percent | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Balanced LR | 1 | 0.502288 | 0.554896 | 0.357553 | 0.434884 | 187 | 150 | 336 | 523 | 30.0230% | 0.369288 |
| Current weighted RF | 1 | 0.400000 | 0.566766 | 0.368726 | 0.446784 | 191 | 146 | 327 | 518 | 29.7359% | 0.383743 |
| Current balanced HGB | 1 | 0.485919 | 0.543027 | 0.349904 | 0.425581 | 183 | 154 | 340 | 523 | 30.0230% | 0.358230 |
| Tuned weighted RF | 1 | 0.522945 | 0.551929 | 0.355641 | 0.432558 | 186 | 151 | 337 | 523 | 30.0230% | 0.382624 |
| Tuned balanced HGB | 1 | 0.524181 | 0.560831 | 0.361377 | 0.439535 | 189 | 148 | 334 | 523 | 30.0230% | 0.374929 |
| Balanced LR | 2 | 0.507006 | 0.569733 | 0.367816 | 0.447031 | 192 | 145 | 330 | 522 | 29.9828% | 0.438786 |
| Current weighted RF | 2 | 0.400000 | 0.575668 | 0.374517 | 0.453801 | 194 | 143 | 324 | 518 | 29.7530% | 0.423985 |
| Current balanced HGB | 2 | 0.477211 | 0.575668 | 0.371648 | 0.451688 | 194 | 143 | 328 | 522 | 29.9828% | 0.438097 |
| Tuned weighted RF | 2 | 0.484753 | 0.596439 | 0.385057 | 0.467986 | 201 | 136 | 321 | 522 | 29.9828% | 0.450564 |
| Tuned balanced HGB | 2 | 0.536772 | 0.590504 | 0.380497 | 0.462791 | 199 | 138 | 324 | 523 | 30.0402% | 0.459678 |
| Balanced LR | 3 | 0.498301 | 0.522255 | 0.337165 | 0.409779 | 176 | 161 | 346 | 522 | 29.9828% | 0.411630 |
| Current weighted RF | 3 | 0.390000 | 0.563798 | 0.368217 | 0.445487 | 190 | 147 | 326 | 516 | 29.6381% | 0.383613 |
| Current balanced HGB | 3 | 0.464251 | 0.543027 | 0.350575 | 0.426077 | 183 | 154 | 339 | 522 | 29.9828% | 0.392160 |
| Tuned weighted RF | 3 | 0.469787 | 0.578635 | 0.373563 | 0.454016 | 195 | 142 | 327 | 522 | 29.9828% | 0.422306 |
| Tuned balanced HGB | 3 | 0.519381 | 0.584570 | 0.377395 | 0.458673 | 197 | 140 | 325 | 522 | 29.9828% | 0.420213 |
| Balanced LR | 4 | 0.499409 | 0.528190 | 0.340344 | 0.413953 | 178 | 159 | 345 | 523 | 30.0057% | 0.380200 |
| Current weighted RF | 4 | 0.390000 | 0.519288 | 0.339806 | 0.410798 | 175 | 162 | 340 | 515 | 29.5468% | 0.387209 |
| Current balanced HGB | 4 | 0.488771 | 0.545994 | 0.351816 | 0.427907 | 184 | 153 | 339 | 523 | 30.0057% | 0.386280 |
| Tuned weighted RF | 4 | 0.460240 | 0.551929 | 0.355641 | 0.432558 | 186 | 151 | 337 | 523 | 30.0057% | 0.400372 |
| Tuned balanced HGB | 4 | 0.531539 | 0.548961 | 0.353728 | 0.430233 | 185 | 152 | 338 | 523 | 30.0057% | 0.403210 |
| Balanced LR | 5 | 0.492779 | 0.519288 | 0.335249 | 0.407451 | 175 | 162 | 347 | 522 | 29.9828% | 0.369913 |
| Current weighted RF | 5 | 0.376667 | 0.528190 | 0.340996 | 0.414435 | 178 | 159 | 344 | 522 | 29.9828% | 0.358312 |
| Current balanced HGB | 5 | 0.463307 | 0.531157 | 0.342912 | 0.416764 | 179 | 158 | 343 | 522 | 29.9828% | 0.386320 |
| Tuned weighted RF | 5 | 0.468130 | 0.534125 | 0.344828 | 0.419092 | 180 | 157 | 342 | 522 | 29.9828% | 0.384469 |
| Tuned balanced HGB | 5 | 0.505379 | 0.516320 | 0.333333 | 0.405122 | 174 | 163 | 348 | 522 | 29.9828% | 0.396755 |

Equal-fold mean and population SD (`ddof=0`), using these fold-specific budgets:

| Model | recall | precision | f1 | average_precision | flagged_percent |
| --- | --- | --- | --- | --- | --- |
| Balanced LR | 0.538872 ± 0.019914 | 0.347625 ± 0.012821 | 0.422620 ± 0.015598 | 0.393963 ± 0.027195 | 29.995401 ± 0.016402 |
| Current weighted RF | 0.550742 ± 0.022568 | 0.358452 ± 0.014909 | 0.434261 ± 0.017934 | 0.387372 ± 0.021035 | 29.731323 ± 0.145921 |
| Tuned weighted RF | 0.562611 ± 0.022094 | 0.362946 ± 0.014399 | 0.441242 ± 0.017435 | 0.408067 ± 0.025582 | 29.995401 ± 0.016402 |
| Current balanced HGB | 0.547774 ± 0.014849 | 0.353371 ± 0.009651 | 0.429603 ± 0.011697 | 0.392217 ± 0.025807 | 29.995401 ± 0.016402 |
| Tuned balanced HGB | 0.560237 ± 0.026700 | 0.361266 ± 0.017129 | 0.439271 ± 0.020869 | 0.410957 ± 0.028352 | 30.006889 ± 0.022509 |

## Product interpretation

- **Tuned weighted RF versus its current version:** +25 defects caught, +1.4837 percentage points recall, +0.8206 points precision, +0.010628 F1, +0.016779 AP; actual review percentage changes from 29.8806% to 29.9954%.
- **Tuned balanced HGB versus its current version:** +21 defects caught, +1.2463 percentage points recall, +0.8040 points precision, +0.009774 F1, +0.017425 AP; actual review percentage changes from 29.9954% to 29.9954%.

**Most defects caught at approximately 30%: Tuned weighted RF**, with 953 of 1,685 defects caught (56.5579% recall), 732 missed, and 1,659 false positives across 2,612 reviews (29.9954%). It catches **+47 defects versus balanced LR**, a +2.7893-point recall change.

**Tuning improves performance, but the gain is modest rather than a clear material
breakthrough.** Tuned RF catches 25 additional defects over current RF, a 1.4837
percentage-point recall increase, while requiring ten additional reviews. It also
produces 15 fewer false positives. Tuned HGB catches 21 additional defects over
current HGB at exactly the same pooled review count, a 1.2463-point recall increase,
with 21 fewer false positives. Both AP scores improve, supporting some ranking
improvement rather than merely a favorable threshold change.

The best procedure, tuned RF, catches 47 more defects than LR at exactly the same
2,612 reviews: a 2.7893-point recall increase, 5.19% more defects caught relative
to LR, and 47 fewer false positives. It still misses 732 defects, or 43.44% of
all defects. Tuned HGB has the highest AP (0.405080), but tuned RF catches eight
more defects at the actual 30% product capacity; AP alone therefore does not
choose the preferred candidate for this objective.

Using fold-specific 30% budgets, tuned RF beats its current reference in four of
five folds and LR in four of five; tuned HGB beats its current reference in four
of five. Improvements are not uniform. The mean fold recall gain is 1.1869 points
for RF and 1.2463 points for HGB; fold-level recall SDs remain about 2.2–2.7 points
for tuned procedures. These descriptive comparisons do not establish statistical
superiority: five folds are few and their training sets overlap. No significance
claim is made.

**The results do not yet make a compelling case for replacing LR solely to gain
this additional complexity.** The best tree procedure may be worthwhile if 47
additional detections per 8,708 modules have high enough product value, but that
requires review-cost and missed-defect-cost evidence not supplied here. Nested
selection costs 295 fits, and RF selects different configurations in different
folds. HGB consistently selects the smaller, slower-learning configuration
(`learning_rate=0.03, max_iter=100, max_leaf_nodes=15, min_samples_leaf=10,
l2_regularization=0.0`), but this observation is not used to refit or promote it.
The limited search cannot rule out improvements elsewhere in parameter space.

**Model choice is probably no longer the main bottleneck.** Given the modest
capacity gains and the large number of remaining missed defects, the next useful
investigation is feature quality, label noise (including conflicting labels for
identical vectors), and feature engineering. Check feature measurement consistency,
label provenance, and whether additional software context could distinguish defects
that the present 21 metrics cannot separate. These are recommendations only; no
such investigation or next milestone is implemented here.

## Reproduction and complete tests

```sh
python -m pytest -q
python -m defectrisk.tree_model_tuning
```

Complete suite: **129 passed in 13.75 seconds**. Added tests verify deterministic
limited search coverage, capacity-recall selection rather than accuracy, nested
selection isolation from outer validation, single held-out probability coverage,
raw-feature contamination rejection at both levels, count arithmetic, and locked
final-test sentinels. All prior tests remain passing.

The workspace interpreter symlink is broken. Execution uses the existing project
packages with the compatible standalone Python already available in this environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PATH=/tmp/defectrisk-python/python/bin:$PATH \
PYTHONPATH=/workspace/defectrisk-ml/.venv/lib/python3.14/site-packages:/workspace/defectrisk-ml/src \
python -m defectrisk.tree_model_tuning
```

Training-only raw exports:

- [Pooled capacity results](tree-model-tuning-results/pooled.csv)
- [Per-outer-fold results](tree-model-tuning-results/per-fold.csv)
- [Equal-fold means and SD](tree-model-tuning-results/summary.csv)
- [Outer and inner integrity audits](tree-model-tuning-results/audits.csv)
- [All inner search scores](tree-model-tuning-results/inner-search.csv)
- [Selected parameters per fold](tree-model-tuning-results/selected-parameters.csv)
- [Outer OOF probabilities, fold membership, and training labels](tree-model-tuning-results/oof-probabilities.csv)
