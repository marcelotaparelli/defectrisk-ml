"""Training-only, group-aware C search for the fixed balanced LR pipeline."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from defectrisk.audit import feature_overlap
from defectrisk.baseline import POSITIVE_CLASS
from defectrisk.class_weight_experiment import create_experiment_pipeline
from defectrisk.cross_validation import METRICS, training_folds
from defectrisk.data import load_jm1
from defectrisk.split import feature_groups, split_dataset
from defectrisk.threshold_experiment import apply_threshold


C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]


def threshold_metric(y_true, probabilities, *, metric: str, pos_label=POSITIVE_CLASS):
    """Score defective probabilities at the fixed existing threshold of 0.50."""
    predictions = apply_threshold(probabilities, 0.50)
    if metric == "accuracy":
        return accuracy_score(y_true, predictions)
    functions = {"precision": precision_score, "recall": recall_score, "f1": f1_score}
    return functions[metric](y_true, predictions, pos_label=pos_label, zero_division=0)


def create_grid_search() -> GridSearchCV:
    """Tune only classifier C; report F1 selection without refitting/promoting."""
    scoring = {
        metric: make_scorer(threshold_metric, response_method="predict_proba",
                            metric=metric, pos_label=POSITIVE_CLASS)
        for metric in METRICS
    }
    return GridSearchCV(
        create_experiment_pipeline(balanced=True),
        param_grid={"classifier__C": C_VALUES.copy()},
        scoring=scoring,
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
        refit=False,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
    )


def best_c_by_f1(results: pd.DataFrame) -> float:
    """Select the highest mean defective F1; grid order resolves exact ties."""
    return float(results["mean_f1"].idxmax())


def run_logistic_grid_search(X_train: pd.DataFrame, y_train: pd.Series):
    """Audit existing folds and explicitly pass feature groups into search.fit.

    Accept only M3 training features/labels. Return CV metric summaries, fold
    audits, and the search object (without a refitted estimator). Defaults and
    the final test partition are never changed or evaluated.
    """
    expected_folds = list(training_folds(X_train, y_train))
    groups = feature_groups(X_train)
    search = create_grid_search()
    actual_folds = list(search.cv.split(X_train, y_train, groups=groups))
    if len(actual_folds) != 5 or len(expected_folds) != 5:
        raise RuntimeError("Grid search must use the existing five folds.")
    audits = []
    for fold, ((fit, validation), (expected_fit, expected_validation)) in enumerate(
        zip(actual_folds, expected_folds), 1
    ):
        overlap = feature_overlap(X_train.iloc[fit], X_train.iloc[validation])
        if overlap["shared_feature_vectors"]:
            raise RuntimeError(f"Fold {fold} has duplicate feature-vector contamination.")
        if not (np.array_equal(fit, expected_fit) and np.array_equal(validation, expected_validation)):
            raise RuntimeError("Grid search folds differ from the existing trusted folds.")
        audits.append({
            "fold": fold, "fit_rows": len(fit), "validation_rows": len(validation),
            "fit_defective": int((y_train.iloc[fit] == POSITIVE_CLASS).sum()),
            "validation_defective": int((y_train.iloc[validation] == POSITIVE_CLASS).sum()),
            **overlap,
        })
    # Integer seed makes this same splitter reproduce its audited indices here.
    search.fit(X_train, y_train, groups=groups)
    raw = search.cv_results_
    results = pd.DataFrame({"C": [params["classifier__C"] for params in raw["params"]]})
    for metric in METRICS:
        results[f"mean_{metric}"] = raw[f"mean_test_{metric}"]
        results[f"std_{metric}"] = raw[f"std_test_{metric}"]
    results["rank_f1"] = raw["rank_test_f1"]
    return results.set_index("C"), pd.DataFrame(audits).set_index("fold"), search


def logistic_grid_report(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    results, audits, _ = run_logistic_grid_search(X_train, y_train)
    best_c = best_c_by_f1(results)
    mean_columns = [f"mean_{metric}" for metric in METRICS]
    return "\n".join([
        "Balanced Logistic Regression — group-aware GridSearchCV, C only",
        "5-fold StratifiedGroupKFold: shuffle=True, random_state=42; groups passed explicitly",
        "Fixed class_weight='balanced', threshold=0.50, median imputer and StandardScaler",
        f"M3 training rows: {len(X_train):,}; positive class='true' (defective)",
        "Audited folds (verified identical to existing trusted CV folds):", audits.to_string(),
        "Equal-fold means and population standard deviations (ddof=0):",
        results.to_string(float_format=lambda value: f"{value:.6f}"),
        f"Best C by mean defective F1: {best_c:g}",
        "Changes versus C=1 (best minus current):",
        (results.loc[best_c, mean_columns] - results.loc[1.0, mean_columns]).to_string(
            float_format=lambda value: f"{value:+.6f}"
        ),
        "Selection metric is defective F1; recall is reported separately for product review.",
        "refit=False: 25 CV fits; no selected model refit, promotion, or default change.",
        "Final test outputs discarded; not inspected, fitted, predicted, or scored.",
    ])


def main() -> None:
    X_train, _, y_train, _ = split_dataset(load_jm1())
    print(logistic_grid_report(X_train, y_train))


if __name__ == "__main__":
    main()
