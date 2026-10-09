"""Measure seven decision thresholds using one balanced fit per existing fold."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from defectrisk.audit import feature_overlap
from defectrisk.baseline import NEGATIVE_CLASS, POSITIVE_CLASS
from defectrisk.class_weight_experiment import create_experiment_pipeline
from defectrisk.cross_validation import training_folds
from defectrisk.data import load_jm1
from defectrisk.split import split_dataset


THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)


def apply_threshold(probabilities, threshold: float) -> np.ndarray:
    """Flag defective when P(defective) >= threshold, including exact ties."""
    values = np.asarray(probabilities, dtype=float)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be a finite one-dimensional array between 0 and 1.")
    return np.where(values >= threshold, POSITIVE_CLASS, NEGATIVE_CLASS)


def balanced_oof_probabilities(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    """Fit five identical balanced pipelines and predict only validation rows.

    Return one held-out defective probability per input row, in original row
    order, plus exact-feature overlap audits. No threshold is used during fit.
    """
    folds = list(training_folds(X_train, y_train))
    coverage = np.zeros(len(X_train), dtype=int)
    for _, validation_indices in folds:
        np.add.at(coverage, validation_indices, 1)
    if len(folds) != 5 or not np.all(coverage == 1):
        raise RuntimeError("Five folds must validate every training row exactly once.")
    probabilities = np.full(len(X_train), np.nan)
    audits = []
    for fold, (fit_indices, validation_indices) in enumerate(folds, 1):
        X_fit, X_validation = X_train.iloc[fit_indices], X_train.iloc[validation_indices]
        y_fit, y_validation = y_train.iloc[fit_indices], y_train.iloc[validation_indices]
        overlap = feature_overlap(X_fit, X_validation)
        if overlap["shared_feature_vectors"]:
            raise RuntimeError(f"Fold {fold} has duplicate feature-vector contamination.")
        pipeline = create_experiment_pipeline(balanced=True)
        pipeline.fit(X_fit, y_fit)
        positive_column = list(pipeline.classes_).index(POSITIVE_CLASS)
        probabilities[validation_indices] = pipeline.predict_proba(X_validation)[:, positive_column]
        audits.append({
            "fold": fold, "fit_rows": len(X_fit), "validation_rows": len(X_validation),
            "fit_defective": int((y_fit == POSITIVE_CLASS).sum()),
            "validation_defective": int((y_validation == POSITIVE_CLASS).sum()),
            **overlap,
        })
    result = pd.Series(probabilities, index=X_train.index, name="oof_defective_probability")
    # Reject missing, non-finite, or invalid outputs before threshold evaluation.
    apply_threshold(result, 0.50)
    return result, pd.DataFrame(audits).set_index("fold")


def evaluate_thresholds(
    y: pd.Series, probabilities: pd.Series, *, thresholds=THRESHOLDS
) -> pd.DataFrame:
    """Measure pooled OOF metrics/counts from the same probabilities each time.

    These are pooled row-level results, not means of fold metrics. No model
    fitting or threshold selection happens here. Preserve input data.
    """
    if y.empty or not y.index.is_unique or not y.index.equals(probabilities.index):
        raise ValueError("Labels and probabilities must have matching unique nonempty indices.")
    if y.isna().any() or not set(y) <= {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError("Labels must be non-missing 'false' or 'true'.")
    rows = []
    for threshold in thresholds:
        predictions = apply_threshold(probabilities, threshold)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, predictions, average="binary", pos_label=POSITIVE_CLASS, zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[NEGATIVE_CLASS, POSITIVE_CLASS]).ravel()
        flagged = int(tp + fp)
        rows.append({
            "threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy_score(y, predictions),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "flagged_modules": flagged, "flagged_percent": flagged / len(y) * 100,
        })
    return pd.DataFrame(rows).set_index("threshold")


def threshold_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    probabilities, audits = balanced_oof_probabilities(X_train, y_train)
    results = evaluate_thresholds(y_train, probabilities)
    return "\n".join([
        "Threshold measurement — balanced Logistic Regression, M3 training only",
        "Same 5-fold StratifiedGroupKFold: shuffle=True, random_state=42",
        "Exactly five pipeline fits and five validation predict_proba calls; no per-threshold retraining",
        f"M3 training rows: {len(X_train):,}; positive class='true' (defective)",
        "Fold integrity audits:", audits.to_string(),
        "Pooled out-of-fold results; predict defective when P(defective) >= threshold:",
        results.to_string(float_format=lambda value: f"{value:.4f}"),
        "Undefined precision/recall/F1 are reported as 0 (zero_division=0).",
        "No default threshold is changed or automatically selected.",
        "The final test outputs were discarded; not inspected, fitted, predicted, or scored.",
    ])


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(threshold_report(X_train, y_train))


if __name__ == "__main__":
    main()
