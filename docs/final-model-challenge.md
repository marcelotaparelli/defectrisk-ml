# Final model challenge: XGBoost versus tuned weighted Random Forest

Executed 2026-10-09, using Python 3.14, scikit-learn 1.9.1, and XGBoost
**3.4.1**. This is the last model-family experiment. The final test
remains untouched, with no inspection, fitting, prediction, scoring, or threshold
selection. This decision does not constitute deployment or final-test validation.

## Predeclared decision rule

At approximately 30% review capacity, select XGBoost if it catches at least 30
additional defects **or** improves defective recall by at least 2 absolute
percentage points over tuned weighted RF, without clearly worse Average Precision.
Otherwise select tuned weighted RF.

The implementation takes a conservative interpretation of the AP requirement:
XGBoost AP must be at least RF AP (zero deterioration tolerance). Workloads must
be within 0.5 percentage points of 30% and within 0.5 points of each other.
These guards are specified before examining outcomes. The `b` ablation is a
single sensitivity analysis, not an additional selection or tuning branch.
Accuracy is not used for search or the final decision.

## Data and nested evaluation

Use only the same M3 training partition: 8,708 rows, 1,685 defective and 7,023
clean, with all 21 original features for the main comparison. The deterministic
80/20 group-aware split and final-test lock are unchanged. Test outputs are
discarded immediately by the command.

- Use exactly the existing five outer `StratifiedGroupKFold` folds
  (`shuffle=True, random_state=42`), with groups from complete original feature
  vectors only. Target values never enter group construction or X.
- Use three inner stratified group-aware folds within each outer fit partition,
  also with seed 42 and groups passed explicitly into the splitter.
- Repeat RF's trusted nine-candidate selection procedure on original features,
  using its same eight limited search configurations plus unchanged reference.
  Select by pooled inner OOF recall at approximately 30% capacity, with AP as
  the tie-breaker. Do not select RF parameters using outer evidence.
- Select XGBoost by the same criterion on the same inner indices, using just
  eight deterministic configurations from the requested 32-combination space.
  A seed-42 shuffled order and coverage-first sampling ensure every requested
  value appears, without evaluating all combinations or using labels to sample.
- Refit each selected candidate using its outer fit partition, then predict only
  that outer validation partition. Pool exactly one held-out probability for
  every training row per candidate.
- For the single ablation, refit XGBoost without `b` using the exact same selected
  parameters and outer fit/validation indices. No ablation-specific inner search.
  Original vectors still define groups; reduced vectors are audited separately.

This is 135 RF inner fits plus 120 XGBoost inner fits and 15 outer fits,
**270 fits total**. Two search workers limit concurrency. No additional model
families, resampling, feature engineering, or threshold search is introduced.

## Pipelines and training-only class weights

RF retains its existing median imputer → StandardScaler → weighted RF pipeline
and prior parameter grid. The selected configuration may vary by outer fold;
this estimates the RF tuning procedure, not one fixed configuration chosen
retrospectively using outer scores.

XGBoost uses median imputation → `XGBClassifier` without StandardScaler, with
`objective="binary:logistic"`, `tree_method="hist"`, `device="cpu"`,
`random_state=42`, and `n_jobs=1`. Only the five requested parameters are searched.
All other model parameters remain default, with no early stopping, internal
validation split, or `eval_set`. Map `"true"` to 1 and `"false"` to 0 explicitly.

**For every individual fit**, compute
`scale_pos_weight = number_of_clean_fit_rows / number_of_defective_fit_rows`.
Inner fits use only their own inner fit labels; outer fits use only their outer
fit labels. Validation labels and final-test labels never set weights. This
follows the [official XGBoost parameter guidance](https://xgboost.readthedocs.io/en/stable/parameter.html).

Weights for all fit partitions (all candidates within a partition use its same
computed weight):

| level | outer_fold | inner_fold | fit_positive | fit_negative | scale_pos_weight |
| --- | --- | --- | --- | --- | --- |
| outer | 1 | 0 | 1348 | 5618 | 4.167656 |
| inner | 1 | 1 | 899 | 3745 | 4.165740 |
| inner | 1 | 2 | 899 | 3745 | 4.165740 |
| inner | 1 | 3 | 898 | 3746 | 4.171492 |
| outer | 2 | 0 | 1348 | 5619 | 4.168398 |
| inner | 2 | 1 | 898 | 3746 | 4.171492 |
| inner | 2 | 2 | 899 | 3746 | 4.166852 |
| inner | 2 | 3 | 899 | 3746 | 4.166852 |
| outer | 3 | 0 | 1348 | 5619 | 4.168398 |
| inner | 3 | 1 | 898 | 3746 | 4.171492 |
| inner | 3 | 2 | 899 | 3746 | 4.166852 |
| inner | 3 | 3 | 899 | 3746 | 4.166852 |
| outer | 4 | 0 | 1348 | 5617 | 4.166914 |
| inner | 4 | 1 | 899 | 3745 | 4.165740 |
| inner | 4 | 2 | 898 | 3745 | 4.170379 |
| inner | 4 | 3 | 899 | 3744 | 4.164627 |
| outer | 5 | 0 | 1348 | 5619 | 4.168398 |
| inner | 5 | 1 | 898 | 3746 | 4.171492 |
| inner | 5 | 2 | 899 | 3746 | 4.166852 |
| inner | 5 | 3 | 899 | 3746 | 4.166852 |

## Eight fixed XGBoost search configurations

| candidate | colsample_bytree | learning_rate | max_depth | n_estimators | subsample |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.000000 | 0.100000 | 6 | 200 | 1.000000 |
| 1 | 0.800000 | 0.030000 | 3 | 500 | 0.800000 |
| 2 | 0.800000 | 0.100000 | 6 | 500 | 1.000000 |
| 3 | 1.000000 | 0.100000 | 3 | 200 | 0.800000 |
| 4 | 1.000000 | 0.030000 | 3 | 200 | 1.000000 |
| 5 | 0.800000 | 0.100000 | 3 | 200 | 0.800000 |
| 6 | 0.800000 | 0.100000 | 3 | 200 | 1.000000 |
| 7 | 1.000000 | 0.100000 | 6 | 500 | 0.800000 |

## Selected parameters per outer fold

| model | outer_fold | parameters |
| --- | --- | --- |
| Tuned weighted RF | 1 | {'class_weight': 'balanced', 'max_depth': 8, 'max_features': 0.7, 'min_samples_leaf': 1, 'n_estimators': 200} |
| XGBoost | 1 | {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8} |
| XGBoost without b | 1 | {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8} |
| Tuned weighted RF | 2 | {'class_weight': 'balanced_subsample', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'n_estimators': 200} |
| XGBoost | 2 | {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |
| XGBoost without b | 2 | {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |
| Tuned weighted RF | 3 | {'class_weight': 'balanced', 'max_depth': 16, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 200} |
| XGBoost | 3 | {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |
| XGBoost without b | 3 | {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |
| Tuned weighted RF | 4 | {'class_weight': 'balanced', 'max_depth': None, 'max_features': 0.7, 'min_samples_leaf': 5, 'n_estimators': 200} |
| XGBoost | 4 | {'colsample_bytree': 0.8, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8} |
| XGBoost without b | 4 | {'colsample_bytree': 0.8, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.8} |
| Tuned weighted RF | 5 | {'class_weight': 'balanced_subsample', 'max_depth': 8, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'n_estimators': 200} |
| XGBoost | 5 | {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |
| XGBoost without b | 5 | {'colsample_bytree': 1.0, 'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |

Every XGBoost-without-b row reuses its corresponding full-feature XGBoost
parameters. No global hyperparameter configuration is chosen using outer scores,
and no final full-training fit is performed in this milestone.

## Zero-overlap proof

All five outer and fifteen inner original-vector boundaries have zero shared
feature vectors, matched fit rows, matched validation rows, and matching row
pairs. All 20 reduced-vector checks after dropping `b` also have zero overlap.
No duplicate rows are removed, and conflicting-label groups remain indivisible.

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

Reduced-vector audit totals:

| level | checks | shared_vectors |
| --- | --- | --- |
| inner | 15 | 0 |
| outer | 5 | 0 |

## Pooled outer OOF results at approximately 30% capacity

| model | threshold | recall | precision | f1 | average_precision | tp | fp | tn | fn | flagged_modules | flagged_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tuned weighted RF | 0.477035 | 0.565579 | 0.364855 | 0.443565 | 0.402211 | 953.000000 | 1659.000000 | 5364.000000 | 732.000000 | 2612.000000 | 29.995407 |
| XGBoost | 0.532520 | 0.549555 | 0.354518 | 0.430998 | 0.402409 | 926.000000 | 1686.000000 | 5337.000000 | 759.000000 | 2612.000000 | 29.995407 |
| XGBoost without b | 0.530213 | 0.555490 | 0.358346 | 0.435653 | 0.404088 | 936.000000 | 1676.000000 | 5347.000000 | 749.000000 | 2612.000000 | 29.995407 |

TP is defects caught; FN is defects missed. FP is clean modules unnecessarily
flagged. The flagged count is TP + FP. The positive-class probability column is
located explicitly, not assumed by position.

Apply the unchanged closest-count budget rule using OOF probabilities only:
choose a threshold giving the closest attainable flagged count to 30% of rows;
keep equal scores together, and prefer fewer reviews for equal-distance ties.
No labels select the capacity threshold. Thresholds above are rounded; CSVs
retain full precision. This describes the OOF population's review capacity,
not an independently validated production threshold.

Average Precision (AP) summarizes the complete ranking curve; it is not
trapezoidal PR integration. Undefined precision/recall/F1 are zero. Pooled
nonlinear metrics differ slightly from equal-fold means.

## Per-fold capacity results

Each fold uses its own probability-only approximately-30% threshold. The pooled
threshold above can differ, so fold counts need not sum to its pooled counts.

| model | threshold | recall | precision | f1 | average_precision | tp | fp | tn | fn | flagged_modules | flagged_percent | outer_fold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tuned weighted RF | 0.522945 | 0.551929 | 0.355641 | 0.432558 | 0.382624 | 186.000000 | 337.000000 | 1068.000000 | 151.000000 | 523.000000 | 30.022962 | 1 |
| XGBoost | 0.537150 | 0.543027 | 0.349904 | 0.425581 | 0.380936 | 183.000000 | 340.000000 | 1065.000000 | 154.000000 | 523.000000 | 30.022962 | 1 |
| XGBoost without b | 0.536260 | 0.560831 | 0.361377 | 0.439535 | 0.385045 | 189.000000 | 334.000000 | 1071.000000 | 148.000000 | 523.000000 | 30.022962 | 1 |
| Tuned weighted RF | 0.484753 | 0.596439 | 0.385057 | 0.467986 | 0.450564 | 201.000000 | 321.000000 | 1083.000000 | 136.000000 | 522.000000 | 29.982769 | 2 |
| XGBoost | 0.543645 | 0.584570 | 0.377395 | 0.458673 | 0.454548 | 197.000000 | 325.000000 | 1079.000000 | 140.000000 | 522.000000 | 29.982769 | 2 |
| XGBoost without b | 0.542133 | 0.584570 | 0.377395 | 0.458673 | 0.456063 | 197.000000 | 325.000000 | 1079.000000 | 140.000000 | 522.000000 | 29.982769 | 2 |
| Tuned weighted RF | 0.469787 | 0.578635 | 0.373563 | 0.454016 | 0.422306 | 195.000000 | 327.000000 | 1077.000000 | 142.000000 | 522.000000 | 29.982769 | 3 |
| XGBoost | 0.522895 | 0.572700 | 0.369732 | 0.449360 | 0.420423 | 193.000000 | 329.000000 | 1075.000000 | 144.000000 | 522.000000 | 29.982769 | 3 |
| XGBoost without b | 0.524926 | 0.563798 | 0.363985 | 0.442375 | 0.419003 | 190.000000 | 332.000000 | 1072.000000 | 147.000000 | 522.000000 | 29.982769 | 3 |
| Tuned weighted RF | 0.460240 | 0.551929 | 0.355641 | 0.432558 | 0.400372 | 186.000000 | 337.000000 | 1069.000000 | 151.000000 | 523.000000 | 30.005737 | 4 |
| XGBoost | 0.535060 | 0.551929 | 0.355641 | 0.432558 | 0.400728 | 186.000000 | 337.000000 | 1069.000000 | 151.000000 | 523.000000 | 30.005737 | 4 |
| XGBoost without b | 0.531248 | 0.557864 | 0.359465 | 0.437209 | 0.402638 | 188.000000 | 335.000000 | 1071.000000 | 149.000000 | 523.000000 | 30.005737 | 4 |
| Tuned weighted RF | 0.468130 | 0.534125 | 0.344828 | 0.419092 | 0.384469 | 180.000000 | 342.000000 | 1062.000000 | 157.000000 | 522.000000 | 29.982769 | 5 |
| XGBoost | 0.523412 | 0.516320 | 0.332696 | 0.404651 | 0.387594 | 174.000000 | 349.000000 | 1055.000000 | 163.000000 | 523.000000 | 30.040207 | 5 |
| XGBoost without b | 0.521846 | 0.519288 | 0.335249 | 0.407451 | 0.388928 | 175.000000 | 347.000000 | 1057.000000 | 162.000000 | 522.000000 | 29.982769 | 5 |

Equal-fold mean ± population SD (`ddof=0`):

| Model | recall | precision | f1 | average_precision | tp | fp | tn | fn | flagged_modules | flagged_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tuned weighted RF | 0.562611 ± 0.022094 | 0.362946 ± 0.014399 | 0.441242 ± 0.017435 | 0.408067 ± 0.025582 | 189.600000 ± 7.445804 | 332.800000 ± 7.652451 | 1071.800000 ± 7.359348 | 147.400000 ± 7.445804 | 522.400000 ± 0.489898 | 29.995401 ± 0.016402 |
| XGBoost | 0.553709 ± 0.023783 | 0.357073 ± 0.015626 | 0.434165 ± 0.018862 | 0.408846 ± 0.026530 | 186.600000 ± 8.014986 | 336.000000 ± 8.438009 | 1068.600000 ± 8.333067 | 150.400000 ± 8.014986 | 522.600000 ± 0.489898 | 30.006889 ± 0.022509 |
| XGBoost without b | 0.557270 ± 0.021183 | 0.359494 ± 0.013653 | 0.437048 ± 0.016603 | 0.410335 ± 0.025781 | 187.800000 ± 7.138627 | 334.600000 ± 7.116179 | 1070.000000 ± 7.155418 | 149.200000 ± 7.138627 | 522.400000 ± 0.489898 | 29.995401 ± 0.016402 |

## Materiality and final model decision

| selected_model | recall_gain | additional_defects_caught | ap_change | comparable_workload | ap_not_worse | material_improvement | model_exploration_finished |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tuned_weighted_random_forest | -0.016024 | -27 | 0.000198 | True | True | False | True |

Full-feature XGBoost versus RF: **-27 defects caught**, **-1.6024 absolute recall points**, -1.0337 precision points, -0.012567 F1, and +0.000198 AP. Actual review percentages: RF 29.995407%, XGBoost 29.995407%.

Excluding `b` versus full-feature XGBoost: **+10 defects caught**, +0.5935 recall points, +0.001679 AP, at 29.995407% review. This is a frozen-parameter sensitivity check, not evidence about the best possible retuned b-free model.

**FINAL MODEL DECISION: select the current tuned weighted Random Forest.**

XGBoost catches 926 defects versus RF's 953 at exactly the same 2,612 reviews:
27 fewer detections, 27 more false positives, and 27 more missed defects. Its
recall is 1.6024 absolute percentage points lower. The AP advantage is only
0.000198 and does not compensate for worse detection at the specified capacity.
XGBoost meets neither the +2-point recall nor the +30-detection requirement.
The final decision is unchanged under any reasonable interpretation of
"without clearly worse AP", because XGBoost fails the recall/count condition.

Dropping `b` adds ten XGBoost detections (+0.5935 recall points) and increases AP
by 0.001679 at the same workload. This is **not a material change under the given
rule**. The b-free ablation still catches 17 fewer defects than RF, so it would
not change the final model-family decision either. This sensitivity analysis
does not certify snapshot provenance or imply that `b` is a leaked label.

**Model exploration is finished.** No further model families or parameter-search
experiments are proposed or implemented. Retain the trusted original-feature RF
tuning procedure as the selected candidate; do not promote the one-detection
pruning change or replace it with XGBoost. The selected model family is recorded
in `decision.json`. No deployed/default configuration is changed, no full-training
model is fitted, and no final-test evaluation is performed.

The outcome reinforces a practical limitation of the current static feature
set rather than evidence that greater model complexity is the main missing
ingredient. This is an inference from repeated small gains, not proof of an
information ceiling. Label ambiguity and snapshot/measurement quality remain
plausible contributors that deserve direct investigation rather than attribution
from model scores alone.

## Evidence limits and genuinely new future data

Nested evaluation separates parameter selection from performance estimation;
however, earlier repeated exploration of this same training population means
this remains exploratory evidence. Five folds have overlapping training sets;
fold SD is descriptive, not a formal significance test. Scores pooled across
folds assume reasonably comparable scales. The limited search cannot prove
another untested configuration would fail, but model exploration ends here.

Static-feature limits appear the most plausible broad limitation: these metrics
summarize code size and complexity but omit change/process/context information.
Exact conflicting labels exist, but only 22 training rows in 11 such groups were
identified in the previous study, with a minimum 11 deterministic classification
errors; those conflicts alone cannot explain hundreds of missed defects. Broader
label ambiguity or noise remains possible. Source-metric formula inconsistencies
and unverified pre/post-fix snapshot timing also warrant data-quality investigation;
this experiment cannot measure their contribution to error or certify provenance.
No numerical information ceiling or dominant error cause is proved.

Future effort should collect genuinely new, prediction-time-available data:
code churn, commit history, previous defects, ownership, change frequency, test
coverage, and review history. Define prediction time and label horizon explicitly,
using only history available at that time. Align metrics with the correct code
snapshot and investigate label provenance. These are recommendations, not
additional model experiments or changes implemented here.

## Reproduction, dependency, and complete tests

The official CPU-only package provides the same `xgboost.XGBClassifier` without
GPU dependencies; see the [official installation guide](https://xgboost.readthedocs.io/en/stable/install.html).
It is pinned as `xgboost-cpu==3.4.1` in project dependencies. The package was
installed into the current Python environment; no other new dependency is needed.

```sh
python -m pytest -q
python -m defectrisk.final_model_challenge
```

Complete test suite: **149 passed in 20.98 seconds**. New tests verify small
reproducible search coverage, exact allowed parameters, CPU settings and
prediction reproducibility, training-only class weights and binary target mapping,
capacity-recall selection rather than accuracy, nested selection isolation,
single frozen-parameter b ablation, fold-trained imputation, complete OOF
coverage/count arithmetic, ablation-induced duplicate rejection, materiality
boundary/AP/workload rules, and final-test sentinels. All previous tests pass.

As in earlier milestones, the workspace interpreter symlink is broken. Execution
uses the compatible standalone Python with the project's existing site-packages:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PATH=/tmp/defectrisk-python/python/bin:$PATH \
PYTHONPATH=/workspace/defectrisk-ml/.venv/lib/python3.14/site-packages:/workspace/defectrisk-ml/src \
python -m defectrisk.final_model_challenge
```

Training-only artifacts:

- [Pooled results](final-model-challenge-results/pooled.csv)
- [Per-fold results](final-model-challenge-results/per-fold.csv)
- [Mean and SD](final-model-challenge-results/summary.csv)
- [Original-vector audits](final-model-challenge-results/audits.csv)
- [b-exclusion audits](final-model-challenge-results/ablation-audits.csv)
- [Training-only class weights](final-model-challenge-results/fit-weights.csv)
- [All inner search evidence](final-model-challenge-results/inner-search.csv)
- [Selected parameters](final-model-challenge-results/selected-parameters.csv)
- [OOF probabilities, training targets, and fold assignments](final-model-challenge-results/oof-probabilities.csv)
- [Final model decision](final-model-challenge-results/decision.json)
