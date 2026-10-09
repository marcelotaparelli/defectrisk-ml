"""Training-only feature quality and paired representation ablations."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from defectrisk.baseline import POSITIVE_CLASS
from defectrisk.audit import feature_overlap
from defectrisk.cross_validation import training_folds
from defectrisk.data import TARGET, load_jm1
from defectrisk.gradient_boosting_baseline import create_boosting_comparison_pipeline, evaluate_review_budgets
from defectrisk.split import feature_groups, split_dataset
from defectrisk.tree_model_tuning import audited_folds, search_candidates, select_candidate

RATIOS = {
    'branch_density': ('branchCount', 'loc'),
    'complexity_density': ('v(g)', 'loc'),
    'comment_ratio': ('lOComment', 'loc'),
    'blank_ratio': ('lOBlank', 'loc'),
    'operator_density': ('total_Op', 'loc'),
    'operand_density': ('total_Opnd', 'loc'),
    'unique_operator_ratio': ('uniq_Op', 'total_Op'),
    'unique_operand_ratio': ('uniq_Opnd', 'total_Opnd'),
}
COUNT_FEATURES = ('loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'lOCode', 'lOComment',
                  'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd',
                  'total_Op', 'total_Opnd', 'branchCount')
VARIANTS = ('original', 'ratios', 'logs', 'engineered', 'pruned')
FAMILIES = ('weighted_random_forest', 'hist_gradient_boosting')


def safe_divide(numerator, denominator):
    """Undefined/missing/nonfinite or negative-count ratios remain missing."""
    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (numerator >= 0) & (denominator > 0)
    return (numerator.where(valid) / denominator.where(valid)).replace([np.inf, -np.inf], np.nan)


class FeatureRepresentation(TransformerMixin, BaseEstimator):
    """No target-derived choices; skew and redundancy rules use fit features only."""

    def __init__(self, variant='engineered'):
        self.variant = variant

    def fit(self, X, y=None):
        if self.variant not in VARIANTS or TARGET in X.columns or not X.columns.is_unique:
            raise ValueError('Valid feature-only schema and representation required.')
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.log_features_ = []
        self.drop_features_ = []
        if self.variant in ('logs', 'engineered'):
            for name in COUNT_FEATURES:
                if name in X:
                    observed = X[name].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(observed) > 2 and observed.min() >= 0 and observed.skew() >= 2:
                        self.log_features_.append(name)
        if self.variant == 'pruned':
            for redundant, retained in [('b', 'v'), ('t', 'e'), ('n', 'total_Op')]:
                if redundant in X and retained in X:
                    correlation = X[[redundant, retained]].corr().iloc[0, 1]
                    if np.isfinite(correlation) and abs(correlation) >= .99:
                        self.drop_features_.append(redundant)
        return self

    def transform(self, X):
        if not X.columns.equals(pd.Index(self.feature_names_in_)):
            raise ValueError('Feature schema must match the fit schema.')
        result = X.astype(float).replace([np.inf, -np.inf], np.nan).copy()
        if self.variant in ('ratios', 'engineered'):
            for name, (numerator, denominator) in RATIOS.items():
                result[name] = safe_divide(result[numerator], result[denominator])
        for name in self.log_features_:
            result[f'log1p__{name}'] = np.log1p(result[name].where(result[name] >= 0))
            result = result.drop(columns=name)
        return result.drop(columns=self.drop_features_)


def univariate_associations(X, y):
    """Descriptive observed-row rank association, directional AUC, class medians."""
    rows = []
    binary = (y == POSITIVE_CLASS).astype(int)
    for name in X:
        observed = X[name].notna() & np.isfinite(X[name])
        values, labels = X.loc[observed, name], binary.loc[observed]
        usable = values.nunique() > 1 and labels.nunique() == 2
        auc = roc_auc_score(labels, values) if usable else .5
        rows.append({'feature': name, 'observed_rows': len(values),
                     'spearman_target': spearmanr(values, labels).statistic if usable else 0.,
                     'auc_increasing': auc, 'rank_biserial': 2 * auc - 1,
                     'clean_median': values[labels == 0].median(),
                     'defective_median': values[labels == 1].median()})
    return pd.DataFrame(rows).set_index('feature')


def feature_quality_audit(X, y):
    # Validate target/index/schema via the same trusted entrypoint; never split test here.
    list(training_folds(X, y))
    distributions = X.describe(percentiles=[.25, .5, .75, .9, .99]).T
    distributions['missing'] = X.isna().sum()
    distributions['zero_rows'] = X.eq(0).sum()
    distributions['negative_rows'] = X.lt(0).sum()
    distributions['noninteger_rows'] = ((X % 1 != 0) & X.notna()).sum()
    distributions['skew'] = X.skew()
    pearson, spearman = X.corr(), X.corr(method='spearman')
    pairs = []
    for i, left in enumerate(X.columns):
        for right in X.columns[i+1:]:
            p, s = pearson.loc[left, right], spearman.loc[left, right]
            if abs(p) >= .95 or abs(s) >= .95:
                pairs.append({'left': left, 'right': right, 'pearson': p, 'spearman': s})
    groups = feature_groups(X)
    counts = pd.crosstab(groups, y).reindex(columns=['false', 'true'], fill_value=0)
    conflicts = counts[(counts['false'] > 0) & (counts['true'] > 0)]
    conflict_stats = {'training_rows': len(X), 'conflicting_groups_in_training': len(conflicts),
                      'rows_in_conflicting_training_groups': int(conflicts.to_numpy().sum()),
                      'clean_rows_in_conflicting_groups': int(conflicts['false'].sum()),
                      'defective_rows_in_conflicting_groups': int(conflicts['true'].sum()),
                      'minimum_row_errors_for_identical_vector_predictions': int(conflicts.min(axis=1).sum())}
    formulas = []
    for name, actual, expected, tolerance in [
        ('b vs v/3000', X.b, X.v / 3000, .0051),
        ('b vs e^(2/3)/3000', X.b, X.e.clip(lower=0) ** (2/3) / 3000, .0051),
        ('t vs e/18', X.t, X.e / 18, .0051),
        ('n vs total_Op + total_Opnd', X.n, X.total_Op + X.total_Opnd, .0001),
    ]:
        valid = actual.notna() & expected.notna()
        residual = abs(actual[valid] - expected[valid])
        formulas.append({'formula': name, 'observed_rows': int(valid.sum()),
                         'within_rounding_rows': int((residual <= tolerance).sum()),
                         'rounding_tolerance': tolerance, 'max_absolute_error': residual.max()})
    engineered = FeatureRepresentation('ratios').fit_transform(X)
    return {'distributions': distributions, 'pearson': pearson, 'spearman': spearman,
            'redundant-pairs': pd.DataFrame(pairs), 'target-associations': univariate_associations(X, y),
            'ratio-associations': univariate_associations(engineered[list(RATIOS)], y),
            'conflicting-groups': conflicts, 'formula-checks': pd.DataFrame(formulas)}, conflict_stats


def representation_pipeline(model, params, variant):
    existing = create_boosting_comparison_pipeline(model)
    existing.set_params(**{f'classifier__{key}': value for key, value in params.items()})
    return Pipeline([('features', FeatureRepresentation(variant)), *existing.steps])


def positive_probabilities(pipeline, X):
    column = list(pipeline.classes_).index(POSITIVE_CLASS)
    return pipeline.predict_proba(X)[:, column]


def ratio_permutation(pipeline, X, y, model, fold):
    """Held-out sensitivity, not causal importance or feature selection."""
    derived = pipeline.named_steps['features'].transform(X)
    rest = pipeline[1:]
    baseline = positive_probabilities(rest, derived)

    def metrics(probabilities):
        recall = evaluate_review_budgets(y, pd.DataFrame({'model': probabilities}, index=y.index), budgets=[.3]).iloc[0].recall
        return recall, average_precision_score(y == POSITIVE_CLASS, probabilities)

    base_recall, base_ap = metrics(baseline)
    rng = np.random.RandomState(42)
    rows = []
    for name in RATIOS:
        for repeat in range(3):
            perturbed = derived.copy()
            perturbed[name] = derived[name].to_numpy()[rng.permutation(len(derived))]
            recall, ap = metrics(positive_probabilities(rest, perturbed))
            rows.append({'model': model, 'outer_fold': fold, 'feature': name, 'repeat': repeat,
                         'recall_drop': base_recall - recall, 'ap_drop': base_ap - ap})
    return pd.DataFrame(rows)


def representation_boundary_audits(X, y):
    """Reject newly identical derived/reduced vectors at either CV level."""
    rows = []
    for outer_fold, (fit, validation) in enumerate(training_folds(X, y), 1):
        X_fit, y_fit = X.iloc[fit], y.iloc[fit]
        boundaries = [('outer', 0, X_fit, X.iloc[validation])]
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        for inner_fold, (inner_fit, inner_validation) in enumerate(
            splitter.split(X_fit, y_fit, groups=feature_groups(X_fit)), 1
        ):
            boundaries.append(('inner', inner_fold, X_fit.iloc[inner_fit], X_fit.iloc[inner_validation]))
        for level, inner_fold, left, right in boundaries:
            for variant in VARIANTS:
                transformer = FeatureRepresentation(variant).fit(left)
                overlap = feature_overlap(transformer.transform(left), transformer.transform(right))
                if overlap['shared_feature_vectors']:
                    raise RuntimeError(f'{variant} creates identical feature-vector contamination at {level} boundary.')
                rows.append({'level': level, 'outer_fold': outer_fold, 'inner_fold': inner_fold,
                             'variant': variant, **overlap})
    return pd.DataFrame(rows)


def feature_comparison(X, y, *, workers=1, candidates=None, permutation=True):
    representation_audits = representation_boundary_audits(X, y)
    outer, audits = audited_folds(X, y, training_folds(X, y), 'outer', 0)
    candidates = candidates if candidates is not None else {model: search_candidates(model) for model in FAMILIES}
    names = [f'{model}__{variant}' for model in FAMILIES for variant in VARIANTS]
    oof = pd.DataFrame(np.nan, index=y.index, columns=names)
    fold_ids = pd.Series(0, index=y.index)
    searches, choices, per_fold, importance = [], [], [], []
    for fold, (fit, validation) in enumerate(outer, 1):
        X_fit, y_fit, X_validation, y_validation = X.iloc[fit], y.iloc[fit], X.iloc[validation], y.iloc[validation]
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        inner, audit = audited_folds(X_fit, y_fit, splitter.split(X_fit, y_fit, groups=feature_groups(X_fit)), 'inner', fold)
        audits.extend(audit)
        for model in FAMILIES:
            params, search = select_candidate(model, candidates[model], X_fit, y_fit, inner, workers=workers)
            search['model'], search['outer_fold'] = model, fold
            searches.append(search)
            for variant in VARIANTS:
                pipeline = representation_pipeline(model, params, variant)
                pipeline.fit(X_fit, y_fit)
                name = f'{model}__{variant}'
                oof.loc[X_validation.index, name] = positive_probabilities(pipeline, X_validation)
                transformer = pipeline.named_steps['features']
                choices.append({'model': name, 'outer_fold': fold, 'parameters': repr(params),
                                'log_features': repr(transformer.log_features_),
                                'dropped_features': repr(transformer.drop_features_)})
                if variant == 'engineered' and permutation:
                    importance.append(ratio_permutation(pipeline, X_validation, y_validation, model, fold))
        fold_ids.iloc[validation] = fold
        scores = evaluate_review_budgets(y_validation, oof.loc[X_validation.index], budgets=[.3]).reset_index()
        scores['outer_fold'] = fold
        scores['average_precision'] = [average_precision_score(y_validation == POSITIVE_CLASS, oof.loc[X_validation.index, name]) for name in scores.model]
        per_fold.append(scores)
        print(f'Completed feature study outer fold {fold}/5.', flush=True)
    pooled = evaluate_review_budgets(y, oof, budgets=[.3]).droplevel('budget')
    pooled['average_precision'] = [average_precision_score(y == POSITIVE_CLASS, oof[name]) for name in pooled.index]
    folds = pd.concat(per_fold, ignore_index=True)
    metrics = ['recall', 'precision', 'f1', 'average_precision', 'flagged_percent']
    summary = folds.groupby('model')[metrics].agg(['mean', lambda values: values.std(ddof=0)])
    summary.columns = [f'{metric}_{"mean" if stat == "mean" else "std"}' for metric, stat in summary.columns]
    return {'pooled': pooled, 'per-fold': folds, 'summary': summary, 'audits': pd.DataFrame(audits),
            'representation-audits': representation_audits,
            'inner-search': pd.concat(searches, ignore_index=True), 'feature-choices': pd.DataFrame(choices),
            'ratio-permutation': pd.concat(importance, ignore_index=True) if importance else pd.DataFrame(),
            'oof-probabilities': oof.assign(target=y, outer_fold=fold_ids)}


def main():
    X_train, _, y_train, _ = split_dataset(load_jm1())
    destination = Path('docs/feature-engineering-results')
    destination.mkdir(parents=True, exist_ok=True)
    quality, conflicts = feature_quality_audit(X_train, y_train)
    for name, frame in quality.items():
        frame.to_csv(destination / f'{name}.csv')
    (destination / 'conflict-summary.json').write_text(json.dumps(conflicts, indent=2))
    print(json.dumps(conflicts, indent=2), flush=True)
    results = feature_comparison(X_train, y_train, workers=2)
    for name, frame in results.items():
        frame.to_csv(destination / f'{name}.csv')
    print(results['pooled'].to_string(float_format=lambda value: f'{value:.6f}'))


if __name__ == '__main__':
    main()
