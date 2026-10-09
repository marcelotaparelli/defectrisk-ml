"""Measure exact feature duplicates and existing holdout contamination; no fitting."""

import json

import pandas as pd

from defectrisk.data import TARGET, load_jm1
from defectrisk.split import group_holdout, split_dataset


def feature_overlap(left: pd.DataFrame, right: pd.DataFrame) -> dict[str, int]:
    """Count shared vectors, affected rows on each side, and matching row pairs.

    Compare all raw feature columns exactly, ignoring row indices and targets.
    Missing values in the same positions compare equal. No hashing or rounding
    is used. A group with m left rows and n right rows contributes m*n pairs.
    """
    if not left.columns.equals(right.columns) or left.shape[1] == 0:
        raise ValueError("Partitions must have the same nonempty feature schema.")
    combined = pd.concat([left, right], ignore_index=True)
    groups = combined.groupby(
        list(combined.columns), dropna=False, sort=False, observed=True
    ).ngroup()
    left_counts = groups.iloc[:len(left)].value_counts()
    right_counts = groups.iloc[len(left):].value_counts()
    shared = left_counts.index.intersection(right_counts.index)
    return {
        "shared_feature_vectors": len(shared),
        "left_rows_with_match": int(left_counts.loc[shared].sum()),
        "right_rows_with_match": int(right_counts.loc[shared].sum()),
        "matching_row_pairs": int((left_counts.loc[shared] * right_counts.loc[shared]).sum()),
    }


def duplicate_audit(frame: pd.DataFrame, *, example_limit: int = 3) -> dict:
    """Audit raw duplicates and reproduce M3/M4 splits without training a model.

    Duplicate groups contain at least two rows. Label consistency considers
    every row in a group, including its first occurrence. Examples retain the
    original row indices and all feature values. Input data is not modified.
    """
    if example_limit < 0:
        raise ValueError("example_limit must be nonnegative.")
    # Reuse M3, including its target/index checks and unchanged default settings.
    X_train, X_test, y_train, _ = split_dataset(frame)
    features = frame.drop(columns=TARGET)
    groups = features.groupby(
        list(features.columns), dropna=False, sort=False, observed=True
    ).ngroup()
    sizes = groups.value_counts()
    label_counts = frame[TARGET].groupby(groups).nunique(dropna=False)
    duplicates = sizes.index[sizes > 1]
    conflicts = label_counts.index[(label_counts > 1) & label_counts.index.isin(duplicates)]
    examples = []
    for group in conflicts[:example_limit]:
        rows = frame.loc[groups == group]
        examples.append({
            "features": rows.drop(columns=TARGET).iloc[0].to_dict(),
            "row_indices": rows.index.tolist(),
            "target_counts": {
                str(label): int(count)
                for label, count in rows[TARGET].value_counts(dropna=False).items()
                if count > 0
            },
        })
    # Exactly the validation split currently used in baseline_report; no pipeline.
    X_fit, X_validation, _, _ = group_holdout(
        X_train, y_train, test_size=0.20, random_state=42
    )
    return {
        "rows": len(frame),
        "unique_feature_vectors": len(sizes),
        "duplicate_full_rows_beyond_first": int(frame.duplicated().sum()),
        "duplicate_feature_rows_beyond_first": int(features.duplicated().sum()),
        "duplicate_feature_groups": len(duplicates),
        "same_label_duplicate_groups": len(duplicates) - len(conflicts),
        "conflicting_label_duplicate_groups": len(conflicts),
        "conflicting_examples": examples,
        "partition_rows": {
            "train": len(X_train), "test": len(X_test),
            "fit": len(X_fit), "validation": len(X_validation),
        },
        "train_test_overlap": feature_overlap(X_train, X_test),
        "fit_validation_overlap": feature_overlap(X_fit, X_validation),
    }


def main() -> None:
    print(json.dumps(duplicate_audit(load_jm1()), indent=2))


if __name__ == "__main__":
    main()
