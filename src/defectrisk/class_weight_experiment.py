"""Paired baseline/balanced Logistic Regression comparison on training-only CV."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from defectrisk.audit import feature_overlap
from defectrisk.baseline import NEGATIVE_CLASS, POSITIVE_CLASS, create_baseline_pipeline
from defectrisk.cross_validation import METRICS, training_folds
from defectrisk.data import load_jm1
from defectrisk.split import split_dataset


def create_experiment_pipeline(*, balanced: bool = False):
    """Create the existing pipeline, optionally changing only class_weight."""
    pipeline = create_baseline_pipeline()
    if balanced:
        pipeline.set_params(classifier__class_weight="balanced")
    return pipeline


def compare_class_weights(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate both configurations on the same five existing training folds.

    Use predict's unchanged classification rule. Balanced weights are learned
    from each fold's fitting labels by LogisticRegression itself. Return fold
    metrics, equal-fold mean/population SD, and summed validation matrices.
    Matrix axes are actual/predicted clean then defective: [[TN, FP], [FN, TP]].
    """
    rows = []
    matrices = {name: np.zeros((2, 2), dtype=int) for name in ("baseline", "balanced")}
    for fold, (fit_indices, validation_indices) in enumerate(training_folds(X_train, y_train), 1):
        X_fit, X_validation = X_train.iloc[fit_indices], X_train.iloc[validation_indices]
        y_fit, y_validation = y_train.iloc[fit_indices], y_train.iloc[validation_indices]
        overlap = feature_overlap(X_fit, X_validation)
        if overlap["shared_feature_vectors"]:
            raise RuntimeError(f"Fold {fold} has duplicate feature-vector contamination.")
        for name in matrices:
            pipeline = create_experiment_pipeline(balanced=name == "balanced")
            pipeline.fit(X_fit, y_fit)
            predictions = pipeline.predict(X_validation)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_validation, predictions, average="binary",
                pos_label=POSITIVE_CLASS, zero_division=0,
            )
            matrix = confusion_matrix(y_validation, predictions, labels=[NEGATIVE_CLASS, POSITIVE_CLASS])
            matrices[name] += matrix
            tn, fp, fn, tp = matrix.ravel()
            rows.append({
                "model": name, "fold": fold,
                "fit_rows": len(X_fit), "validation_rows": len(X_validation),
                "fit_defective": int((y_fit == POSITIVE_CLASS).sum()),
                "validation_defective": int((y_validation == POSITIVE_CLASS).sum()),
                **overlap,
                "accuracy": accuracy_score(y_validation, predictions),
                "precision": precision, "recall": recall, "f1": f1,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            })
    folds = pd.DataFrame(rows).set_index(["model", "fold"])
    summary = pd.concat({
        name: pd.DataFrame({
            "mean": folds.loc[name, METRICS].mean(),
            "std": folds.loc[name, METRICS].std(ddof=0),
        }) for name in matrices
    }, names=["model", "metric"])
    return folds, summary, matrices


def class_weight_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    folds, summary, matrices = compare_class_weights(X_train, y_train)
    lines = [
        "Controlled Logistic Regression experiment: baseline vs class_weight='balanced'",
        "Same 5-fold StratifiedGroupKFold: shuffle=True, random_state=42",
        f"M3 training rows: {len(X_train):,}; positive class='true' (defective)",
        "Per-fold metrics, raw-feature overlap audit, and confusion counts:",
        folds.to_string(float_format=lambda value: f"{value:.4f}"),
        "Unweighted means and population standard deviations (ddof=0):",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "Mean changes (balanced minus baseline):",
        (summary.loc["balanced", "mean"] - summary.loc["baseline", "mean"]).to_string(
            float_format=lambda value: f"{value:+.4f}"
        ),
    ]
    for name, matrix in matrices.items():
        lines.extend([
            f"{name} aggregated cross-validation confusion matrix ([[TN, FP], [FN, TP]]):",
            pd.DataFrame(matrix, index=["actual clean", "actual defective"],
                         columns=["predicted clean", "predicted defective"]).to_string(),
        ])
    lines.extend([
        "Each M3 training row contributes one held-out prediction per model.",
        "Undefined precision/recall/F1 are reported as 0 (zero_division=0).",
        "Only class_weight differs. No resampling or classification-threshold change.",
        "The final test partition was not passed to the experiment, fitted, or scored.",
    ])
    return "\n".join(lines)


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(class_weight_report(X_train, y_train))


if __name__ == "__main__":
    main()
