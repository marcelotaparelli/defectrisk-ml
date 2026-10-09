"""Logistic Regression baseline with validation confined to training data."""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from defectrisk.data import TARGET, load_jm1
from defectrisk.split import feature_groups, group_holdout, split_dataset


# Preserve OpenML's string labels: true = defective, false = clean.
POSITIVE_CLASS = "true"
NEGATIVE_CLASS = "false"


def create_baseline_pipeline() -> Pipeline:
    """Create an unfitted pipeline with fixed, untuned baseline settings."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=5000)),
        ]
    )


def baseline_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """Fit on a training subset and report one training-side validation holdout.

    Only pass the M3 training partition. The final test partition is neither
    accepted nor evaluated here. Do not interpret this as final test performance.
    """
    if TARGET in X_train.columns:
        raise ValueError(f"Target {TARGET!r} must not be included in features.")
    if len(X_train) != len(y_train) or not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels must have matching lengths and indices.")
    if y_train.isna().any() or set(y_train) != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError("Target must contain exactly the non-missing labels 'false' and 'true'.")

    X_fit, X_validation, y_fit, y_validation = group_holdout(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
    )
    pipeline = create_baseline_pipeline()
    pipeline.fit(X_fit, y_fit)
    predictions = pipeline.predict(X_validation)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_validation,
        predictions,
        average="binary",
        pos_label=POSITIVE_CLASS,
        zero_division=0,
    )
    matrix = pd.DataFrame(
        confusion_matrix(y_validation, predictions, labels=[NEGATIVE_CLASS, POSITIVE_CLASS]),
        index=["actual clean", "actual defective"],
        columns=["predicted clean", "predicted defective"],
    )
    groups = feature_groups(X_train)
    overlap = len(set(groups.loc[X_fit.index]) & set(groups.loc[X_validation.index]))
    distributions = []
    for name, labels in [("Fit", y_fit), ("Validation", y_validation)]:
        counts = labels.value_counts().sort_index()
        distribution = pd.DataFrame({"count": counts, "percent": counts / len(labels) * 100})
        distributions.append(f"{name} class distribution:\n" + distribution.to_string(
            float_format=lambda value: f"{value:.2f}"
        ))
    return "\n".join(
        [
            "Logistic Regression baseline — training-side validation only",
            "Positive class: 'true' (defective); negative class: 'false' (clean)",
            "Validation split: group-aware, test_size=0.20, random_state=42",
            f"M3 training rows: {len(X_train):,}",
            f"Fit rows: {len(X_fit):,}; validation rows: {len(X_validation):,}",
            f"Shared feature vectors (fit/validation): {overlap}",
            *distributions,
            f"Accuracy: {accuracy_score(y_validation, predictions):.4f}",
            f"Precision (defective): {precision:.4f}",
            f"Recall (defective): {recall:.4f}",
            f"F1 (defective): {f1:.4f}",
            "Confusion matrix (rows=actual, columns=predicted; [[TN, FP], [FN, TP]]):",
            matrix.to_string(),
            "Undefined precision/recall/F1 are reported as 0 (zero_division=0).",
            "The final test set was not fitted, scored, or used for decisions.",
        ]
    )


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(baseline_report(X_train, y_train))


if __name__ == "__main__":
    main()
