"""Final nested XGBoost challenge; one frozen-parameter b exclusion ablation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from defectrisk.audit import feature_overlap
from defectrisk.baseline import NEGATIVE_CLASS, POSITIVE_CLASS
from defectrisk.cross_validation import training_folds
from defectrisk.data import TARGET, load_jm1
from defectrisk.gradient_boosting_baseline import evaluate_review_budgets
from defectrisk.split import feature_groups, split_dataset
from defectrisk.tree_model_tuning import audited_folds, fitted_probabilities, search_candidates, select_candidate

SPACE = {'n_estimators': [200, 500], 'max_depth': [3, 6], 'learning_rate': [.03, .1],
         'subsample': [.8, 1.], 'colsample_bytree': [.8, 1.]}
MODELS = ('tuned_weighted_random_forest', 'xgboost', 'xgboost_without_b')


def xgboost_candidates():
    grid = list(ParameterGrid(SPACE))
    remaining = [grid[i] for i in np.random.RandomState(42).permutation(len(grid))]
    uncovered = {(name, value) for name, values in SPACE.items() for value in values}
    result = []
    for _ in range(8):
        index = max(range(len(remaining)), key=lambda i: sum(
            (name, value) in uncovered for name, value in remaining[i].items()))
        params = remaining.pop(index)
        result.append(params)
        uncovered.difference_update(params.items())
    if uncovered:
        raise RuntimeError('Limited search did not cover requested values.')
    return result


def binary_labels(labels):
    if labels.isna().any() or set(labels) != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError('Each fit partition requires both nonmissing target classes.')
    return (labels == POSITIVE_CLASS).astype(int)


def create_xgboost_pipeline(params, y_fit):
    labels = binary_labels(y_fit)
    if set(params) != set(SPACE) or any(value not in SPACE[key] for key, value in params.items()):
        raise ValueError('Only the five requested XGBoost search parameters are allowed.')
    weight = int((labels == 0).sum()) / int((labels == 1).sum())
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', XGBClassifier(
            objective='binary:logistic', tree_method='hist', device='cpu',
            random_state=42, n_jobs=1, scale_pos_weight=weight, **params,
        )),
    ])


def xgboost_probabilities(params, X_fit, y_fit, X_validation, *, without_b=False):
    if TARGET in X_fit or TARGET in X_validation:
        raise ValueError('Target must never enter X.')
    pipeline = create_xgboost_pipeline(params, y_fit)
    if without_b:
        X_fit, X_validation = X_fit.drop(columns='b'), X_validation.drop(columns='b')
    pipeline.fit(X_fit, binary_labels(y_fit))
    column = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(X_validation)[:, column]


def xgboost_inner_score(params, X, y, folds):
    oof = np.full(len(X), np.nan)
    for fit, validation in folds:
        oof[validation] = xgboost_probabilities(params, X.iloc[fit], y.iloc[fit], X.iloc[validation])
    result = evaluate_review_budgets(y, pd.DataFrame({'xgboost': oof}, index=y.index), budgets=[.3]).iloc[0].to_dict()
    result['average_precision'] = average_precision_score(y == POSITIVE_CLASS, oof)
    return result


def select_xgboost(candidates, X, y, folds, *, workers=1):
    results = pd.DataFrame(Parallel(n_jobs=workers, prefer='threads')(
        delayed(xgboost_inner_score)(params, X, y, folds) for params in candidates))
    results['candidate'] = range(len(candidates))
    results['parameters'] = [repr(params) for params in candidates]
    winner = int(results.sort_values(['recall', 'average_precision', 'candidate'], ascending=[False, False, True]).iloc[0].candidate)
    results['selected'] = results.candidate.eq(winner)
    return candidates[winner], results


def final_decision(pooled):
    """Predeclared product decision; b exclusion remains diagnostic only.

    Use conservative AP >= reference (zero AP deterioration tolerance).
    Workloads must be within 0.5 percentage points of each other and 30%.
    """
    rf, xgb = pooled.loc[MODELS[0]], pooled.loc['xgboost']
    recall_gain, caught_gain = float(xgb.recall - rf.recall), int(xgb.tp - rf.tp)
    comparable = (abs(xgb.flagged_percent - rf.flagged_percent) <= .5
                  and abs(xgb.flagged_percent - 30) <= .5 and abs(rf.flagged_percent - 30) <= .5)
    ap_ok = bool(xgb.average_precision >= rf.average_precision)
    material = bool((recall_gain >= .02 - 1e-12 or caught_gain >= 30) and comparable and ap_ok)
    return {'selected_model': 'xgboost' if material else MODELS[0],
            'recall_gain': recall_gain, 'additional_defects_caught': caught_gain,
            'ap_change': float(xgb.average_precision - rf.average_precision),
            'comparable_workload': bool(comparable), 'ap_not_worse': ap_ok,
            'material_improvement': material, 'model_exploration_finished': True}


def challenge(X, y, *, workers=1, rf_candidates=None, xgb_candidates=None):
    outer, audits = audited_folds(X, y, training_folds(X, y), 'outer', 0)
    rf_candidates = search_candidates('weighted_random_forest') if rf_candidates is None else rf_candidates
    xgb_candidates = xgboost_candidates() if xgb_candidates is None else xgb_candidates
    oof = pd.DataFrame(np.nan, index=y.index, columns=MODELS)
    assignments = pd.Series(0, index=y.index)
    searches, choices, per_fold, weights, ablation_audits = [], [], [], [], []
    for number, (fit, validation) in enumerate(outer, 1):
        X_fit, y_fit = X.iloc[fit], y.iloc[fit]
        X_validation, y_validation = X.iloc[validation], y.iloc[validation]
        inner_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        inner, inner_audits = audited_folds(X_fit, y_fit,
            inner_splitter.split(X_fit, y_fit, groups=feature_groups(X_fit)), 'inner', number)
        audits.extend(inner_audits)
        boundaries = [('outer', 0, X_fit, X_validation, y_fit)] + [
            ('inner', i, X_fit.iloc[inner_fit], X_fit.iloc[inner_validation], y_fit.iloc[inner_fit])
            for i, (inner_fit, inner_validation) in enumerate(inner, 1)]
        for level, inner_number, left, right, labels in boundaries:
            overlap = feature_overlap(left.drop(columns='b'), right.drop(columns='b'))
            if overlap['shared_feature_vectors']:
                raise RuntimeError('Excluding b creates cross-boundary feature-vector contamination.')
            ablation_audits.append({'level': level, 'outer_fold': number, 'inner_fold': inner_number, **overlap})
            positives = int((labels == POSITIVE_CLASS).sum())
            negatives = int((labels == NEGATIVE_CLASS).sum())
            weights.append({'level': level, 'outer_fold': number, 'inner_fold': inner_number,
                            'fit_positive': positives, 'fit_negative': negatives,
                            'scale_pos_weight': negatives / positives})
        rf_params, rf_search = select_candidate('weighted_random_forest', rf_candidates, X_fit, y_fit, inner, workers=workers)
        xgb_params, xgb_search = select_xgboost(xgb_candidates, X_fit, y_fit, inner, workers=workers)
        for model, params, search in [(MODELS[0], rf_params, rf_search), ('xgboost', xgb_params, xgb_search)]:
            search['model'], search['outer_fold'] = model, number
            searches.append(search)
            choices.append({'model': model, 'outer_fold': number, 'parameters': repr(params)})
        choices.append({'model': 'xgboost_without_b', 'outer_fold': number, 'parameters': repr(xgb_params)})
        oof.loc[X_validation.index, MODELS[0]] = fitted_probabilities('weighted_random_forest', rf_params, X_fit, y_fit, X_validation)
        oof.loc[X_validation.index, 'xgboost'] = xgboost_probabilities(xgb_params, X_fit, y_fit, X_validation)
        oof.loc[X_validation.index, 'xgboost_without_b'] = xgboost_probabilities(xgb_params, X_fit, y_fit, X_validation, without_b=True)
        assignments.iloc[validation] = number
        scores = evaluate_review_budgets(y_validation, oof.loc[X_validation.index], budgets=[.3]).reset_index()
        scores['outer_fold'] = number
        scores['average_precision'] = [average_precision_score(y_validation == POSITIVE_CLASS, oof.loc[X_validation.index, model]) for model in scores.model]
        per_fold.append(scores)
        print(f'Completed final challenge outer fold {number}/5.', flush=True)
    pooled = evaluate_review_budgets(y, oof, budgets=[.3]).droplevel('budget')
    pooled['average_precision'] = [average_precision_score(y == POSITIVE_CLASS, oof[model]) for model in pooled.index]
    folds = pd.concat(per_fold, ignore_index=True)
    metrics = ['recall', 'precision', 'f1', 'average_precision', 'tp', 'fp', 'tn', 'fn', 'flagged_modules', 'flagged_percent']
    summary = folds.groupby('model')[metrics].agg(['mean', lambda values: values.std(ddof=0)])
    summary.columns = [f'{metric}_{"mean" if stat == "mean" else "std"}' for metric, stat in summary.columns]
    return {'pooled': pooled, 'per-fold': folds, 'summary': summary, 'audits': pd.DataFrame(audits),
            'ablation-audits': pd.DataFrame(ablation_audits), 'fit-weights': pd.DataFrame(weights),
            'inner-search': pd.concat(searches, ignore_index=True), 'selected-parameters': pd.DataFrame(choices),
            'oof-probabilities': oof.assign(target=y, outer_fold=assignments)}, final_decision(pooled)


def main():
    X_train, _, y_train, _ = split_dataset(load_jm1())
    results, decision = challenge(X_train, y_train, workers=2)
    destination = Path('docs/final-model-challenge-results')
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(destination / f'{name}.csv')
    (destination / 'decision.json').write_text(json.dumps(decision, indent=2))
    print(results['pooled'].to_string(float_format=lambda value: f'{value:.6f}'))
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
