"""Paired training-only boosting experiment and descriptive OOF review budgets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from defectrisk.baseline import POSITIVE_CLASS
from defectrisk.data import load_jm1
from defectrisk.random_forest_baseline import compare_random_forest
from defectrisk.random_forest_class_weight import create_weight_comparison_pipeline
from defectrisk.split import split_dataset
from defectrisk.threshold_experiment import apply_threshold, evaluate_thresholds

MODELS = ("balanced_logistic_regression", "weighted_random_forest", "hist_gradient_boosting")
BUDGETS = (0.10, 0.20, 0.30, 0.40)


def create_boosting_comparison_pipeline(model):
    if model == "hist_gradient_boosting":
        # No internal random holdout: only the audited outer folds validate.
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(
                random_state=42, class_weight="balanced", early_stopping=False,
            )),
        ])
    if model in MODELS:
        return create_weight_comparison_pipeline(model)
    raise ValueError(f"Unknown comparison model: {model!r}")


def review_threshold(probabilities, budget):
    """Closest attainable review count; ties favor fewer reviews, never use labels.

    All equal scores are kept together under the existing >= threshold rule.
    Budgets describe this OOF population, not a deployment threshold estimate.
    """
    values = np.asarray(probabilities, dtype=float)
    apply_threshold(values, 0.5)  # Reuse finite/range/dimensionality validation.
    if not len(values) or not np.isfinite(budget) or not 0 < budget < 1:
        raise ValueError("Nonempty scores and a budget strictly between 0 and 1 required.")
    scores, counts = np.unique(values, return_counts=True)
    flagged = np.cumsum(counts[::-1])[::-1]
    candidates = list(zip(scores, flagged))
    if scores[-1] < 1:
        candidates.append((np.nextafter(scores[-1], 1.0), 0))
    threshold, _ = min(candidates, key=lambda item: (abs(item[1] - len(values) * budget), item[1]))
    return float(threshold)


def evaluate_review_budgets(y, oof, budgets=BUDGETS):
    rows = []
    for model in oof.columns:
        for budget in budgets:
            threshold = review_threshold(oof[model], budget)
            metrics = evaluate_thresholds(y, oof[model], thresholds=[threshold]).iloc[0].to_dict()
            rows.append({"model": model, "budget": budget, "threshold": threshold, **metrics})
    return pd.DataFrame(rows).set_index(["model", "budget"])


def compare_gradient_boosting(X_train, y_train):
    factories = {model: (lambda model=model: create_boosting_comparison_pipeline(model)) for model in MODELS}
    folds, summary, pooled, audits, oof = compare_random_forest(
        X_train, y_train, pipeline_factories=factories, return_oof=True,
    )
    budgets = evaluate_review_budgets(y_train, oof)
    curve_rows, ap = [], {}
    for model in MODELS:
        precision, recall, thresholds = precision_recall_curve(y_train == POSITIVE_CLASS, oof[model])
        ap[model] = average_precision_score(y_train == POSITIVE_CLASS, oof[model])
        curve_rows.append(pd.DataFrame({"model": model, "precision": precision, "recall": recall,
                                       "threshold": np.append(thresholds, np.nan)}))
    return folds, summary, pooled, audits, oof, budgets, pd.concat(curve_rows, ignore_index=True), pd.Series(ap, name="average_precision")


def main():
    X_train, _, y_train, _ = split_dataset(load_jm1())
    folds, summary, pooled, audits, oof, budgets, curves, ap = compare_gradient_boosting(X_train, y_train)
    destination = Path("docs/gradient-boosting-results")
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in [("per-fold", folds), ("summary", summary), ("pooled", pooled),
                        ("audits", audits), ("review-budgets", budgets), ("precision-recall", curves),
                        ("average-precision", ap), ("oof-probabilities", oof.assign(target=y_train))]:
        frame.to_csv(destination / f"{name}.csv")
    for title, frame in [("Fold integrity", audits), ("Per fold at 0.50", folds),
                         ("Mean and population SD", summary), ("Pooled at 0.50", pooled),
                         ("Review budgets (closest attainable workload)", budgets), ("Average precision", ap)]:
        print(title)
        print(frame.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
