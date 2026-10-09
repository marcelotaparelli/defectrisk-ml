"""Inspect the fixed balanced C=1 candidate fitted on all M3 training rows."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from defectrisk.baseline import NEGATIVE_CLASS, POSITIVE_CLASS
from defectrisk.class_weight_experiment import create_experiment_pipeline
from defectrisk.data import TARGET, load_jm1
from defectrisk.split import split_dataset


def coefficient_table(pipeline: Pipeline) -> tuple[pd.DataFrame, float]:
    """Map fitted standardized coefficients to original names, largest |coef| first."""
    classifier = pipeline.named_steps["classifier"]
    if list(classifier.classes_) != [NEGATIVE_CLASS, POSITIVE_CLASS]:
        raise ValueError("Coefficient direction requires binary classes ['false', 'true'].")
    names = pipeline[:-1].get_feature_names_out()
    if not np.array_equal(names, pipeline.feature_names_in_):
        raise ValueError("Preprocessing must preserve every original feature for coefficient mapping.")
    coefficients = classifier.coef_[0]
    if len(names) != len(coefficients):
        raise ValueError("Feature names and coefficients must have matching lengths.")
    table = pd.DataFrame({
        "feature": names,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
        "direction": np.select(
            [coefficients > 0, coefficients < 0],
            ["toward defective", "toward clean"], default="neutral",
        ),
    }).sort_values("absolute_coefficient", ascending=False, kind="stable").reset_index(drop=True)
    return table, float(classifier.intercept_[0])


def fit_training_coefficients(X_train: pd.DataFrame, y_train: pd.Series):
    """Fit once on the full supplied M3 training partition; no prediction/scoring."""
    if TARGET in X_train.columns:
        raise ValueError(f"Target {TARGET!r} must not be included in features.")
    if not X_train.index.is_unique or not X_train.index.equals(y_train.index):
        raise ValueError("Training features and labels must have matching unique indices.")
    if y_train.isna().any() or set(y_train) != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError("Target must contain exactly the non-missing labels 'false' and 'true'.")
    pipeline = create_experiment_pipeline(balanced=True).set_params(classifier__C=1.0)
    pipeline.fit(X_train, y_train)
    table, intercept = coefficient_table(pipeline)
    return table, intercept, pipeline


def logistic_coefficient_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    table, intercept, _ = fit_training_coefficients(X_train, y_train)
    return "\n".join([
        "Balanced Logistic Regression parameters — full M3 training fit only",
        "random_state=42, max_iter=5000, class_weight='balanced', C=1.0",
        "Pipeline: median imputer -> StandardScaler -> LogisticRegression",
        f"M3 training rows: {len(X_train):,}; coefficients: {len(table)}",
        "Coefficients act on standardized imputed features; positive class='true' (defective)",
        f"Learned intercept (standardized feature space): {intercept:.9f}",
        "Features sorted by absolute coefficient magnitude:",
        table.to_string(index=False, float_format=lambda value: f"{value:.9f}"),
        "Positive means increasing that standardized feature pushes toward defective, holding others fixed.",
        "Negative means increasing that standardized feature pushes toward clean, holding others fixed.",
        "StandardScaler makes magnitudes more comparable across features than raw-scale coefficients.",
        "Coefficient importance is NOT causality; correlated features can make individual coefficients unstable.",
        "No predictions, scoring, tuning, or model promotion were performed in this analysis.",
        "Final test outputs discarded; not inspected, fitted, predicted, or scored.",
    ])


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(logistic_coefficient_report(X_train, y_train))


if __name__ == "__main__":
    main()
