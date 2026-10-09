"""Untuned forest versus balanced Logistic Regression on identical training folds."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from defectrisk.audit import feature_overlap
from defectrisk.baseline import POSITIVE_CLASS, create_baseline_pipeline
from defectrisk.class_weight_experiment import create_experiment_pipeline
from defectrisk.cross_validation import training_folds
from defectrisk.data import load_jm1
from defectrisk.split import split_dataset
from defectrisk.threshold_experiment import evaluate_thresholds


MODELS = ("balanced_logistic_regression", "random_forest")


def create_comparison_pipeline(model: str):
    """Keep median imputation/scaling; use exactly the requested classifiers."""
    if model == "balanced_logistic_regression":
        return create_experiment_pipeline(balanced=True)
    if model == "random_forest":
        return create_baseline_pipeline().set_params(
            classifier=RandomForestClassifier(random_state=42)
        )
    raise ValueError(f"Unknown comparison model: {model!r}")


def compare_random_forest(X_train: pd.DataFrame, y_train: pd.Series, *, pipeline_factories=None, return_oof=False):
    """Return fold results, equal-fold mean/population SD, pooled results, audits.

    Both models use P(defective) >= 0.50, the threshold experiment's unchanged
    rule. Fit one fresh pipeline per model per fold; no final test input.
    Optional factories allow controlled variants to reuse this exact evaluator.
    With return_oof=True, append a row-aligned dataframe of defective probabilities;
    the original four-result interface remains unchanged by default.
    """
    if pipeline_factories is None:
        pipeline_factories = {
            model: (lambda model=model: create_comparison_pipeline(model)) for model in MODELS
        }
    folds = list(training_folds(X_train, y_train))
    coverage = np.zeros(len(X_train), dtype=int)
    for _, validation in folds:
        np.add.at(coverage, validation, 1)
    if len(folds) != 5 or not np.all(coverage == 1):
        raise RuntimeError("Five folds must validate every training row exactly once.")
    oof = {model: np.full(len(X_train), np.nan) for model in pipeline_factories}
    rows, audits = [], []
    for fold, (fit, validation) in enumerate(folds, 1):
        X_fit, X_validation = X_train.iloc[fit], X_train.iloc[validation]
        y_fit, y_validation = y_train.iloc[fit], y_train.iloc[validation]
        overlap = feature_overlap(X_fit, X_validation)
        if overlap["shared_feature_vectors"]:
            raise RuntimeError(f"Fold {fold} has duplicate feature-vector contamination.")
        audits.append({
            "fold": fold, "fit_rows": len(fit), "validation_rows": len(validation),
            "fit_defective": int((y_fit == POSITIVE_CLASS).sum()),
            "validation_defective": int((y_validation == POSITIVE_CLASS).sum()),
            **overlap,
        })
        for model, factory in pipeline_factories.items():
            pipeline = factory()
            pipeline.fit(X_fit, y_fit)
            positive_column = list(pipeline.classes_).index(POSITIVE_CLASS)
            probabilities = pipeline.predict_proba(X_validation)[:, positive_column]
            oof[model][validation] = probabilities
            scores = evaluate_thresholds(
                y_validation, pd.Series(probabilities, index=y_validation.index), thresholds=[0.50]
            ).iloc[0].to_dict()
            rows.append({"model": model, "fold": fold, **scores})
    per_fold = pd.DataFrame(rows).set_index(["model", "fold"])
    summary = pd.concat({
        model: pd.DataFrame({
            "mean": per_fold.loc[model].mean(), "std": per_fold.loc[model].std(ddof=0)
        }) for model in pipeline_factories
    }, names=["model", "metric"])
    pooled = pd.concat({
        model: evaluate_thresholds(
            y_train, pd.Series(values, index=y_train.index), thresholds=[0.50]
        ) for model, values in oof.items()
    }, names=["model", "threshold"]).droplevel("threshold")
    result = (per_fold, summary, pooled, pd.DataFrame(audits).set_index("fold"))
    if return_oof:
        return (*result, pd.DataFrame(oof, index=y_train.index))
    return result


def random_forest_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    folds, summary, pooled, audits = compare_random_forest(X_train, y_train)
    return "\n".join([
        "Untuned Random Forest versus balanced Logistic Regression — M3 training only",
        "Identical five StratifiedGroupKFold folds: shuffle=True, random_state=42",
        "Both models: P(defective) >= 0.50; positive class='true'",
        f"M3 training rows: {len(X_train):,}",
        "Fold integrity audits:", audits.to_string(),
        "Per-fold metrics, confusion counts, and review workload:",
        folds.to_string(float_format=lambda value: f"{value:.4f}"),
        "Unweighted means and population standard deviations (ddof=0):",
        summary.to_string(float_format=lambda value: f"{value:.4f}"),
        "Pooled out-of-fold metrics, confusion counts, and review workload:",
        pooled.to_string(float_format=lambda value: f"{value:.4f}"),
        "Undefined precision/recall/F1 are reported as 0 (zero_division=0).",
        "No hyperparameter/threshold tuning, external resampling, or model promotion.",
        "Final test outputs discarded; not inspected, fitted, predicted, or scored.",
    ])


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(random_forest_report(X_train, y_train))


if __name__ == "__main__":
    main()
