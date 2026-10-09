"""Load and inspect the unchanged JM1 dataset; no preprocessing or modeling."""

from pathlib import Path

import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml


OPENML_DATA_ID = 1053
TARGET = "defects"


def load_jm1(data_home: str | Path = ".cache/openml") -> pd.DataFrame:
    """Return all JM1 features and its target, preserving missing/duplicate rows.

    The fixed OpenML ID selects JM1 version 1. Downloads are cached locally;
    the first call needs internet access. The target is explicitly requested so
    it cannot accidentally become a predictive feature through metadata changes.
    """
    dataset = fetch_openml(
        data_id=OPENML_DATA_ID,
        target_column=TARGET,
        as_frame=True,
        parser="pandas",
        data_home=str(data_home),
        cache=True,
    )
    return dataset.frame


def dataset_report(frame: pd.DataFrame) -> str:
    """Summarize the raw frame without changing it."""
    features = frame.drop(columns=TARGET)
    target = frame[TARGET]
    counts = target.value_counts(dropna=False).sort_index()
    distribution = pd.DataFrame(
        {"count": counts, "percent": counts / len(frame) * 100}
    )
    columns = pd.DataFrame(
        {"dtype": frame.dtypes.astype(str), "missing": frame.isna().sum()}
    )
    class_counts = target.value_counts()
    minority = class_counts.idxmin()
    ratio = class_counts.max() / class_counts.min()
    return "\n".join(
        [
            f"JM1 — OpenML dataset {OPENML_DATA_ID}, version 1",
            f"pandas {pd.__version__}; scikit-learn {sklearn.__version__}",
            f"Rows: {len(frame):,}; feature columns: {features.shape[1]}",
            f"Target: {TARGET}; values: {sorted(target.dropna().unique())}",
            "Features: " + ", ".join(features.columns),
            "\nColumn types and missing values:",
            columns.to_string(),
            f"\nRows with any missing value: {frame.isna().any(axis=1).sum():,}",
            f"Duplicate full rows (beyond first): {frame.duplicated().sum():,}",
            f"Duplicate feature rows (beyond first): {features.duplicated().sum():,}",
            "\nClass distribution (% of all rows):",
            distribution.to_string(float_format=lambda value: f"{value:.2f}"),
            f"Minority class: {minority}",
            f"Majority/minority ratio: {ratio:.2f}:1",
            "\nSchema review: defects is the outcome and must be excluded from X.",
            "b needs provenance review as a Halstead defect estimate; its name alone",
            "does not establish leakage. No obvious identifier column in the JM1 schema.",
            "Duplicates could contaminate future evaluation if shared across partitions.",
            "No rows or columns have been cleaned, removed, or split.",
        ]
    )


def main() -> None:
    print(dataset_report(load_jm1()))


if __name__ == "__main__":
    main()
