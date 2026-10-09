# Evaluation integrity: exact feature duplicates

This records the historical audit before the split fix. Current audit code uses
group-aware holdouts; see [the fix report](evaluation-integrity-fix.md) and
[current measured output](duplicate-audit-after-group-fix.json). The original
JSON below is retained as historical evidence, not current split results.

Run `python -m defectrisk.audit` from the project directory. The reusable
`duplicate_audit(frame)` returns counts, conflicting examples, partition sizes,
and overlaps. `feature_overlap(left, right)` measures two feature partitions.
Both compare all raw feature columns exactly, ignoring targets and row indices;
missing values at the same positions compare equal. No rounding, imputation,
row removal, model fitting, or tuning is performed.

The audit uses the existing M3 split and reproduces the baseline's training-side
validation split: both have `test_size=0.20`, `random_state=42`, and stratification
by their input labels. Production split and baseline code are unchanged.

## Measured JM1 results

The actual cached OpenML dataset 1053 was loaded and audited. Full output,
including all 21 feature values and original row indices for three conflicting
groups, is saved in [duplicate-audit-results.json](duplicate-audit-results.json).

| Measurement | Count |
| --- | ---: |
| Rows | 10,885 |
| Unique feature vectors | 8,824 |
| Duplicate full rows beyond first | 1,973 |
| Duplicate feature rows beyond first | 2,061 |
| Duplicate feature groups (at least two rows) | 669 |
| Duplicate groups with one consistent target label | 581 |
| Duplicate groups with conflicting target labels | 88 |

| Split | Left rows | Right rows | Shared unique vectors | Left rows with a match | Right rows with a match | Matching row pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Final train / test | 8,708 | 2,177 | 299 | 1,317 | 501 | 11,467 |
| Fit / validation | 6,966 | 1,742 | 227 | 969 | 381 | 7,837 |

A shared vector counts once regardless of multiplicity. Affected rows count
every row whose vector occurs on the other side. Matching row pairs count all
cross-partition matches: m rows on the left and n on the right contribute m*n.
These counts include matches with either consistent or conflicting targets.
Thus 501 test rows and 381 validation rows have feature vectors already present
on their respective training sides. This establishes duplicate contamination;
the audit does not measure how much it affects model metrics.

Examples (indices are original zero-based dataset indices):

| Row indices sharing all 21 features | Clean (`false`) | Defective (`true`) |
| --- | ---: | ---: |
| 8, 4999, 5002 | 2 | 1 |
| 60, 3172 | 1 | 1 |
| 74, 1086, 1311, 1657, 4240, 8398 | 2 | 4 |

## Recommendation, not implemented

Use a group-aware holdout at both boundaries, defining each group by the entire
feature vector, excluding the target. Assign every row in a group to one side,
including groups with conflicting labels. Preserve approximately 80/20 row
sizes and class balance where group constraints allow, and audit zero shared
vectors before evaluating models. Exact row ratios or class proportions may
not be attainable when groups must remain intact.

This follows scikit-learn's [guidance on grouped splits](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
Conflicting labels mean feature grouping must not include the target; a full-row
group would separate opposite labels for the same vector and leave leakage.
Conflicts alone do not establish whether labels are wrong or whether different
modules share the same recorded metrics. No labels or rows were altered.

## Execution and validation

The workspace virtual environment's Python symlinks were broken because the
referenced `/usr/bin/python` interpreter was absent. The audit ran with a
temporary compatible Python 3.14 interpreter under `/tmp`, using the existing
virtual environment's installed packages and cached JM1 data via `PYTHONPATH`.
No project dependencies were installed or replaced.

Audit tests cover missing-value equality, conflicting labels, group/row/pair
count distinctions, input preservation, disjoint features with matching indices,
schema checking, and agreement with the actual baseline validation split while
stopping before pipeline creation. Data and split tests are also run; baseline
tests that fit models are excluded from this milestone's test run.
