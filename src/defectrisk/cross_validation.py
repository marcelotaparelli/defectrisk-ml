"""Five-fold evaluation of the unchanged baseline on M3 training data only."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from defectrisk.audit import feature_overlap
from defectrisk.baseline import NEGATIVE_CLASS, POSITIVE_CLASS, create_baseline_pipeline
from defectrisk.data import load_jm1
from defectrisk.split import feature_groups, split_dataset


METRICS = ["accuracy", "precision", "recall", "f1"]


def training_folds(X_train: pd.DataFrame, y_train: pd.Series):
    """Yield five reproducible positional fit/validation index pairs.

    Accept only the M3 training partition. Groups use all raw features, with
    matching missing values treated as equal; labels never define a group.
    Stratification is approximate because groups must remain indivisible.
    """
    if not X_train.index.is_unique:
        raise ValueError("Training row indices must be unique.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels must have matching indices.")
    if y_train.isna().any() or set(y_train) != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError("Target must contain exactly the non-missing labels 'false' and 'true'.")
    groups = feature_groups(X_train)
    if groups.nunique() < 5:
        raise ValueError("Five-fold cross-validation requires at least five feature groups.")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yield from splitter.split(X_train, y_train, groups=groups)


def cross_validate_baseline(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-fold results and unweighted mean/population SD (ddof=0).

    A fresh existing pipeline is fitted in each fold. Neither the final test
    partition nor fitted estimators are accepted or evaluated here.
    """
    rows = []
    for fold, (fit_indices, validation_indices) in enumerate(training_folds(X_train, y_train), 1):
        X_fit, X_validation = X_train.iloc[fit_indices], X_train.iloc[validation_indices]
        y_fit, y_validation = y_train.iloc[fit_indices], y_train.iloc[validation_indices]
        overlap = feature_overlap(X_fit, X_validation)
        if overlap["shared_feature_vectors"]:
            raise RuntimeError(f"Fold {fold} has duplicate feature-vector contamination.")
        pipeline = create_baseline_pipeline()
        pipeline.fit(X_fit, y_fit)
        predictions = pipeline.predict(X_validation)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_validation, predictions, average="binary", pos_label=POSITIVE_CLASS, zero_division=0
        )
        rows.append({
            "fold": fold,
            "fit_rows": len(X_fit), "validation_rows": len(X_validation),
            "fit_defective": int((y_fit == POSITIVE_CLASS).sum()),
            "validation_defective": int((y_validation == POSITIVE_CLASS).sum()),
            "fit_defective_percent": float((y_fit == POSITIVE_CLASS).mean() * 100),
            "validation_defective_percent": float((y_validation == POSITIVE_CLASS).mean() * 100),
            **overlap,
            "accuracy": accuracy_score(y_validation, predictions),
            "precision": precision, "recall": recall, "f1": f1,
        })
    folds = pd.DataFrame(rows).set_index("fold")
    summary = pd.DataFrame({
        "mean": folds[METRICS].mean(),
        "std": folds[METRICS].std(ddof=0),
    })
    return folds, summary


def cross_validation_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    folds, summary = cross_validate_baseline(X_train, y_train)
    return "\n".join([
        "Logistic Regression — 5-fold StratifiedGroupKFold on M3 training only",
        "shuffle=True, random_state=42; groups exclude target; positive class='true'",
        f"M3 training rows: {len(X_train):,}",
        "Fold metrics and exact raw-feature overlap audit:",
        folds.to_string(float_format=lambda value: f"{value:.4f}"),
        "Unweighted fold means and population standard deviations (ddof=0):",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "Undefined precision/recall/F1 are reported as 0 (zero_division=0).",
        "The final test partition was not passed to cross-validation, fitted, or scored.",
    ])


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(cross_validation_report(X_train, y_train))


if __name__ == "__main__":
    main()
