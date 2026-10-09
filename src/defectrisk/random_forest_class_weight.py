"""Controlled forest class-weight comparison with balanced LR as reference."""

import pandas as pd

from defectrisk.data import load_jm1
from defectrisk.random_forest_baseline import compare_random_forest, create_comparison_pipeline
from defectrisk.split import split_dataset


MODELS = ("balanced_logistic_regression", "random_forest", "weighted_random_forest")


def create_weight_comparison_pipeline(model: str):
    """Change only class_weight for the weighted forest; retain other candidates."""
    if model == "weighted_random_forest":
        return create_comparison_pipeline("random_forest").set_params(
            classifier__class_weight="balanced"
        )
    return create_comparison_pipeline(model)


def compare_forest_class_weights(X_train: pd.DataFrame, y_train: pd.Series):
    factories = {
        model: (lambda model=model: create_weight_comparison_pipeline(model)) for model in MODELS
    }
    return compare_random_forest(X_train, y_train, pipeline_factories=factories)


def forest_class_weight_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    folds, summary, pooled, audits = compare_forest_class_weights(X_train, y_train)
    return "\n".join([
        "Controlled Random Forest class-weight experiment, with balanced LR reference",
        "Identical five StratifiedGroupKFold folds: shuffle=True, random_state=42",
        "All models: P(defective) >= 0.50; positive class='true'",
        f"M3 training rows: {len(X_train):,}",
        "Fold integrity audits:", audits.to_string(),
        "Per-fold metrics, confusion counts, and review workload:",
        folds.to_string(float_format=lambda value: f"{value:.4f}"),
        "Unweighted means and population standard deviations (ddof=0):",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "Mean changes (weighted forest minus default forest):",
        (summary.loc["weighted_random_forest", "mean"] - summary.loc["random_forest", "mean"]).to_string(
            float_format=lambda value: f"{value:+.4f}"
        ),
        "Pooled out-of-fold metrics, confusion counts, and review workload:",
        pooled.to_string(float_format=lambda value: f"{value:.4f}"),
        "Undefined precision/recall/F1 are reported as 0 (zero_division=0).",
        "Only forest class_weight changes; LR, preprocessing, thresholds, and folds are unchanged.",
        "No external resampling, other parameter tuning, or model promotion.",
        "Final test outputs discarded; not inspected, fitted, predicted, or scored.",
    ])


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(forest_class_weight_report(X_train, y_train))


if __name__ == "__main__":
    main()
