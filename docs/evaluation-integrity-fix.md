# Evaluation-integrity fix

Both evaluation boundaries now use `group_holdout` from `defectrisk.split`.
Groups use all 21 raw feature columns only, with matching missing values treated
as equal. Targets and row indices do not enter the grouping key. All duplicate
rows are retained, including the 88 conflicting-label groups. No feature group
can cross an evaluation boundary; the helper checks group disjointness before
returning the partitions.

Allocation is deterministic for the same data, row order, and seed (42 by
default). It considers largest groups first, with seeded ties, targeting 20% of
the input rows and of each class. It then moves or exchanges whole groups while
the normalized squared row/class-count deviation improves. This uses label counts
to balance one holdout; no models, predictions, validation scores, or final test
scores influence allocation. Indivisible groups may prevent exact ratios; inputs
with fewer than two groups are rejected rather than splitting a group.

## Real JM1 results

| Partition | Rows | Clean | Clean % | Defective | Defective % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 10,885 | 8,779 | 80.65 | 2,106 | 19.35 |
| Train | 8,708 | 7,023 | 80.65 | 1,685 | 19.35 |
| Final test | 2,177 | 1,756 | 80.66 | 421 | 19.34 |
| Fit | 6,966 | 5,618 | 80.65 | 1,348 | 19.35 |
| Validation | 1,742 | 1,405 | 80.65 | 337 | 19.35 |

The independent exact-feature audit reports zero shared vectors, zero affected
rows on either side, and zero matching pairs across **both** train/test and
fit/validation. Previously, shared vectors were 299 and 227 respectively.
Raw data still has 8,824 unique feature vectors, 1,973 duplicate full rows beyond
first, and 2,061 duplicate feature rows beyond first.

## Unchanged Logistic Regression pipeline: validation comparison

The old baseline was rerun before changing the split code to obtain an observed
comparison in the same environment. Both runs fit only their fitting subset and
predict only their training-side validation subset. The final test set is neither
fitted nor scored. Pipeline settings are unchanged.

| Validation metric | Old row-based splits | New group-aware splits |
| --- | ---: | ---: |
| Accuracy | 0.8134 | 0.8203 |
| Defective precision | 0.5811 | 0.8000 |
| Defective recall | 0.1276 | 0.0950 |
| Defective F1 | 0.2092 | 0.1698 |

Old confusion matrix: `[[1374, 31], [294, 43]]`.
New confusion matrix: `[[1397, 8], [305, 32]]`.
Rows are actual clean/defective; columns are predicted clean/defective.

These are different holdout memberships. The differences do not isolate the
effect of contamination. The new validation results measure performance on
feature vectors absent from the fitting subset; defective recall remains low.

## Execution and evidence

Executed `python -m defectrisk.split`, `python -m defectrisk.baseline`, and
`python -m defectrisk.audit` on cached OpenML JM1, ID 1053. Reports are retained:

- [New split report](split-after-group-fix.txt)
- [New baseline report](baseline-after-group-fix.txt)
- [Old baseline report](baseline-before-group-fix.txt)
- [New independent audit](duplicate-audit-after-group-fix.json)
- [Historical independent audit](duplicate-audit-results.json)

The workspace's virtual-environment interpreter symlinks are broken. Commands
used a temporary compatible Python 3.14 interpreter under `/tmp` and the
existing virtual environment's packages through `PYTHONPATH`. No dependencies
were installed or replaced. No convergence warning occurred in either real
baseline run.

The complete suite passes: **52 tests**, without exclusions. Tests cover both
boundaries on synthetic and real JM1 data, deterministic reproducibility,
alignment, complete row preservation, target exclusion, approximate sizes and
class balance, conflicting labels, missing values, preprocessing fit boundaries,
and existing classifier/report behavior. Tests allow small percentage deviations
for whole-group allocation rather than requiring the old exact rounded counts.

The earlier report's 32 tests were deliberately a subset: 36 original tests plus
5 new audit tests minus 9 baseline tests equals 32. Baseline tests were omitted
then to honor the request not to train models during the audit. This fix adds
11 test cases, making the complete suite 52.

No cross-validation, new models, hyperparameter search, row removal, or final
test model evaluation is implemented.
