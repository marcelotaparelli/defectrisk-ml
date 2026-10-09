"""Limited nested group-aware tree search, selected by recall at 30% capacity."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from defectrisk.audit import feature_overlap
from defectrisk.baseline import POSITIVE_CLASS
from defectrisk.cross_validation import training_folds
from defectrisk.data import load_jm1
from defectrisk.gradient_boosting_baseline import (
    MODELS as REFERENCES, create_boosting_comparison_pipeline, evaluate_review_budgets,
)
from defectrisk.split import feature_groups, split_dataset

RF_SPACE = {
    "n_estimators": [200, 500], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", 0.7], "class_weight": ["balanced", "balanced_subsample"],
}
HGB_SPACE = {
    "learning_rate": [0.03, 0.1], "max_iter": [100, 300], "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40], "l2_regularization": [0.0, 1.0],
}
FAMILIES = {"tuned_weighted_random_forest": "weighted_random_forest",
            "tuned_hist_gradient_boosting": "hist_gradient_boosting"}


def search_candidates(model):
    """Eight coverage-first deterministic draws plus the unchanged reference."""
    space = RF_SPACE if model == "weighted_random_forest" else HGB_SPACE
    if model not in ("weighted_random_forest", "hist_gradient_boosting"):
        raise ValueError(f"Unknown tree family: {model}")
    # {} denotes the trusted configuration; HGB's exact reference lies in its grid.
    reference = create_boosting_comparison_pipeline(model).named_steps["classifier"].get_params()
    grid = list(ParameterGrid(space))
    draws = [grid[index] for index in np.random.RandomState(42).permutation(len(grid))
             if any(reference[key] != value for key, value in grid[index].items())]
    uncovered = {(key, repr(value)) for key, values in space.items() for value in values}
    chosen = []
    for _ in range(8):
        # Seeded order breaks ties. Cover every requested value before filling
        # remaining slots randomly; no outcomes or target labels enter sampling.
        best = max(range(len(draws)), key=lambda i: sum(
            (key, repr(value)) in uncovered for key, value in draws[i].items()
        ))
        params = draws.pop(best)
        chosen.append(params)
        uncovered.difference_update((key, repr(value)) for key, value in params.items())
    if uncovered:
        raise RuntimeError("Limited search failed to cover all requested parameter values.")
    return [{}] + chosen


def audited_folds(X, y, folds, level, outer_fold):
    """Validate OOF coverage and audit raw feature boundaries before fitting."""
    folds = list(folds)
    coverage = np.zeros(len(X), dtype=int)
    rows = []
    for number, (fit, validation) in enumerate(folds, 1):
        np.add.at(coverage, validation, 1)
        if np.intersect1d(fit, validation).size or len(fit) + len(validation) != len(X):
            raise RuntimeError("Fold indices must partition all rows.")
        overlap = feature_overlap(X.iloc[fit], X.iloc[validation])
        if overlap["shared_feature_vectors"]:
            raise RuntimeError(f"{level} fold {number} has duplicate feature-vector contamination.")
        rows.append({"level": level, "outer_fold": outer_fold, "fold": number,
                     "fit_rows": len(fit), "validation_rows": len(validation), **overlap})
    if not np.all(coverage == 1):
        raise RuntimeError("Every row must receive one held-out prediction.")
    return folds, rows


def fitted_probabilities(model, params, X_fit, y_fit, X_validation):
    pipeline = create_boosting_comparison_pipeline(model)
    pipeline.set_params(**{f"classifier__{key}": value for key, value in params.items()})
    pipeline.fit(X_fit, y_fit)
    column = list(pipeline.classes_).index(POSITIVE_CLASS)
    return pipeline.predict_proba(X_validation)[:, column]


def candidate_score(model, params, X, y, folds):
    probabilities = np.full(len(X), np.nan)
    for fit, validation in folds:
        probabilities[validation] = fitted_probabilities(model, params, X.iloc[fit], y.iloc[fit], X.iloc[validation])
    scores = evaluate_review_budgets(y, pd.DataFrame({model: probabilities}, index=y.index), budgets=[.3]).iloc[0]
    return {**scores.to_dict(), "average_precision": average_precision_score(y == POSITIVE_CLASS, probabilities)}


def select_candidate(model, candidates, X, y, folds, *, workers=1):
    """Selection sees inner training/OOF labels only, never outer validation."""
    scores = Parallel(n_jobs=workers, prefer="threads")(
        delayed(candidate_score)(model, params, X, y, folds) for params in candidates
    )
    results = pd.DataFrame(scores)
    results["candidate"] = range(len(candidates))
    results["parameters"] = [repr(params) for params in candidates]
    # Recall at capacity is primary, AP only breaks ties; accuracy is never used.
    best = int(results.sort_values(["recall", "average_precision", "candidate"],
                                  ascending=[False, False, True]).iloc[0]["candidate"])
    results["selected"] = results.candidate.eq(best)
    return candidates[best], results


def nested_tree_comparison(X_train, y_train, *, workers=1, candidates=None):
    outer, audits = audited_folds(X_train, y_train, training_folds(X_train, y_train), "outer", 0)
    if len(outer) != 5:
        raise RuntimeError("Exactly five outer folds are required.")
    candidates = candidates if candidates is not None else {
        model: search_candidates(model) for model in FAMILIES.values()
    }
    model_names = (*REFERENCES, *FAMILIES)
    oof = pd.DataFrame(np.nan, index=y_train.index, columns=model_names)
    assignments = pd.Series(0, index=y_train.index, name="outer_fold")
    searches, selections, fold_scores = [], [], []
    for number, (fit, validation) in enumerate(outer, 1):
        X_fit, y_fit = X_train.iloc[fit], y_train.iloc[fit]
        X_validation, y_validation = X_train.iloc[validation], y_train.iloc[validation]
        groups = feature_groups(X_fit)
        inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        inner, inner_audits = audited_folds(
            X_fit, y_fit, inner_splitter.split(X_fit, y_fit, groups=groups), "inner", number,
        )
        audits.extend(inner_audits)
        for model in REFERENCES:
            oof.loc[X_validation.index, model] = fitted_probabilities(model, {}, X_fit, y_fit, X_validation)
        for tuned, model in FAMILIES.items():
            best, search = select_candidate(model, candidates[model], X_fit, y_fit, inner, workers=workers)
            search["outer_fold"], search["model"] = number, tuned
            searches.append(search)
            selections.append({"outer_fold": number, "model": tuned, "parameters": repr(best)})
            oof.loc[X_validation.index, tuned] = fitted_probabilities(model, best, X_fit, y_fit, X_validation)
        assignments.iloc[validation] = number
        fold_result = evaluate_review_budgets(y_validation, oof.loc[X_validation.index], budgets=[.3]).reset_index()
        fold_result["outer_fold"] = number
        fold_result["average_precision"] = [average_precision_score(y_validation == POSITIVE_CLASS,
                                             oof.loc[X_validation.index, model]) for model in fold_result.model]
        fold_scores.append(fold_result)
        print(f"Completed outer fold {number}/5; inner searches used only its fit partition.", flush=True)
    results = evaluate_review_budgets(y_train, oof, budgets=[.3]).droplevel("budget")
    results["average_precision"] = [average_precision_score(y_train == POSITIVE_CLASS, oof[model]) for model in results.index]
    per_fold = pd.concat(fold_scores, ignore_index=True)
    metrics = ["recall", "precision", "f1", "average_precision", "flagged_percent"]
    summary = per_fold.groupby("model")[metrics].agg(["mean", lambda values: values.std(ddof=0)])
    summary.columns = [f"{metric}_{'std' if stat != 'mean' else 'mean'}" for metric, stat in summary.columns]
    return {"pooled": results, "per-fold": per_fold, "summary": summary,
            "audits": pd.DataFrame(audits), "inner-search": pd.concat(searches, ignore_index=True),
            "selected-parameters": pd.DataFrame(selections),
            "oof-probabilities": oof.assign(target=y_train, outer_fold=assignments)}


def main():
    X_train, _, y_train, _ = split_dataset(load_jm1())
    results = nested_tree_comparison(X_train, y_train, workers=2)
    destination = Path("docs/tree-model-tuning-results")
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(destination / f"{name}.csv")
    print(results["pooled"].to_string(float_format=lambda value: f"{value:.6f}"))
    print(results["selected-parameters"].to_string(index=False))


if __name__ == "__main__":
    main()
