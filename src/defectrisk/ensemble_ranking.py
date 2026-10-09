"""Fixed training-only probability ensembles from trusted nested OOF artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from defectrisk.audit import feature_overlap
from defectrisk.baseline import POSITIVE_CLASS
from defectrisk.cross_validation import training_folds
from defectrisk.data import load_jm1
from defectrisk.gradient_boosting_baseline import evaluate_review_budgets, review_threshold
from defectrisk.split import split_dataset
from defectrisk.threshold_experiment import apply_threshold

BASES = ('rf', 'hgb', 'xgb')
# .5 RF + .5 XGB is already rf_xgb; do not report it as another experiment.
WEIGHTS = {
    'average_three': (1/3, 1/3, 1/3),
    'rf_hgb': (.5, .5, 0), 'rf_xgb': (.5, 0, .5), 'hgb_xgb': (0, .5, .5),
    'rf_50_hgb_25_xgb_25': (.5, .25, .25),
    'rf_60_hgb_20_xgb_20': (.6, .2, .2),
}


def validate_oof(y, probabilities):
    if tuple(probabilities.columns) != BASES or not y.index.is_unique or not probabilities.index.equals(y.index):
        raise ValueError('Three probability columns must align exactly to unique training rows.')
    if y.isna().any() or set(y) != {'false', 'true'}:
        raise ValueError('Both nonmissing target classes are required.')
    for model in BASES:
        apply_threshold(probabilities[model], .5)


def combine_probabilities(probabilities, weights=WEIGHTS):
    if tuple(probabilities.columns) != BASES:
        raise ValueError('Expected ordered RF/HGB/XGB columns.')
    for model in BASES:
        apply_threshold(probabilities[model], .5)
    result = probabilities.copy(deep=True)
    for name, values in weights.items():
        weight = np.asarray(values, dtype=float)
        if name in result or weight.shape != (3,) or not np.isfinite(weight).all() or (weight < 0).any() or not np.isclose(weight.sum(), 1):
            raise ValueError('Distinct names and three nonnegative weights summing to one required.')
        result[name] = probabilities.to_numpy() @ weight
    return result


def load_trusted_oof(X, y, tree_path, challenge_path):
    """Read training-only artifacts; reject alignment, label, fold or score changes."""
    tree = pd.read_csv(tree_path, index_col=0, dtype={'target': str})
    challenge = pd.read_csv(challenge_path, index_col=0, dtype={'target': str})
    expected_folds = pd.Series(0, index=X.index, name='outer_fold')
    audits = []
    for number, (fit, validation) in enumerate(training_folds(X, y), 1):
        overlap = feature_overlap(X.iloc[fit], X.iloc[validation])
        if overlap['shared_feature_vectors']:
            raise RuntimeError('Trusted fold has feature-vector contamination.')
        expected_folds.iloc[validation] = number
        audits.append({'fold': number, 'fit_rows': len(fit), 'validation_rows': len(validation), **overlap})
    for frame in (tree, challenge):
        if not frame.index.equals(X.index) or not frame.index.is_unique:
            raise ValueError('OOF artifact rows must match the complete M3 training partition.')
        if not np.array_equal(frame.target.to_numpy(), y.astype(str).to_numpy()):
            raise ValueError('OOF targets do not match training labels.')
        if not np.array_equal(frame.outer_fold.to_numpy(), expected_folds.to_numpy()):
            raise ValueError('OOF predictions do not use the exact trusted outer folds.')
    if not np.array_equal(tree.tuned_weighted_random_forest.to_numpy(), challenge.tuned_weighted_random_forest.to_numpy()):
        raise ValueError('RF probabilities differ between source experiments.')
    probabilities = pd.DataFrame({'rf': tree.tuned_weighted_random_forest,
                                  'hgb': tree.tuned_hist_gradient_boosting,
                                  'xgb': challenge.xgboost}, index=X.index)
    validate_oof(y, probabilities)
    manifest = {'sources': {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for path in [tree_path, challenge_path]},
                'training_rows': len(X), 'training_defective': int((y == POSITIVE_CLASS).sum()),
                'new_model_fits': 0, 'final_test_access': False,
                'candidate_definitions': WEIGHTS,
                'xgboost_variant': 'Full original-feature nested candidate; b ablation is not selected',
                'meaningful_rule': 'Recall gain >= .02 OR >=30 additional defects; AP >= RF; review within .5 points of 30% and RF'}
    return probabilities, expected_folds, pd.DataFrame(audits), manifest


def diversity_report(y, scores, pooled):
    flags = pd.DataFrame({name: scores[name] >= pooled.loc[name, 'threshold'] for name in scores}, index=y.index)
    defective = y == POSITIVE_CLASS
    rows = []
    for name in BASES:
        others = [other for other in BASES if other != name]
        caught = defective & flags[name]
        rows.append({'model': name, 'caught': int(caught.sum()),
                     'uniquely_caught_among_three': int((caught & ~flags[others].any(axis=1)).sum()),
                     'caught_not_by_rf': int((caught & ~flags.rf).sum()),
                     'rf_caught_not_by_model': int((defective & flags.rf & ~flags[name]).sum())})
    patterns = pd.DataFrame({'rf': flags.rf[defective], 'hgb': flags.hgb[defective], 'xgb': flags.xgb[defective]})
    patterns = patterns.value_counts(sort=False).rename('defects').reset_index()
    pairs = []
    for i, left in enumerate(BASES):
        for right in BASES[i+1:]:
            union = (flags[left] | flags[right]).sum()
            shared = (flags[left] & flags[right]).sum()
            pairs.append({'left': left, 'right': right, 'shared_flagged': int(shared),
                          'flagged_union': int(union), 'flagged_jaccard': shared / union if union else 0,
                          'shared_caught_defects': int((defective & flags[left] & flags[right]).sum()),
                          'probability_spearman': scores[left].corr(scores[right], method='spearman')})
    gains = []
    for name in scores.columns:
        if name in BASES:
            continue
        gains.append({'ensemble': name,
                      'new_defects_vs_rf': int((defective & flags[name] & ~flags.rf).sum()),
                      'rf_defects_lost': int((defective & flags.rf & ~flags[name]).sum()),
                      'new_caught_by_other_base': int((defective & flags[name] & ~flags.rf & (flags.hgb | flags.xgb)).sum())})
    return pd.DataFrame(rows), patterns, pd.DataFrame(pairs), pd.DataFrame(gains)


def ensemble_report(y, probabilities, fold_ids):
    validate_oof(y, probabilities)
    if not fold_ids.index.equals(y.index) or set(fold_ids) != {1, 2, 3, 4, 5}:
        raise ValueError('Exactly five aligned trusted fold assignments required.')
    scores = combine_probabilities(probabilities)
    pooled = evaluate_review_budgets(y, scores, budgets=[.3]).droplevel('budget')
    pooled['average_precision'] = [average_precision_score(y == POSITIVE_CLASS, scores[name]) for name in pooled.index]
    fold_results = []
    for fold in range(1, 6):
        mask = fold_ids == fold
        result = evaluate_review_budgets(y.loc[mask], scores.loc[mask], budgets=[.3]).reset_index()
        result['fold'] = fold
        result['average_precision'] = [average_precision_score(y.loc[mask] == POSITIVE_CLASS, scores.loc[mask, name]) for name in result.model]
        fold_results.append(result)
    per_fold = pd.concat(fold_results, ignore_index=True)
    summary = per_fold.groupby('model')[['recall', 'precision', 'f1', 'average_precision', 'flagged_percent']].agg(['mean', lambda s: s.std(ddof=0)])
    summary.columns = [f'{metric}_{"mean" if stat == "mean" else "std"}' for metric, stat in summary.columns]
    unique, patterns, pairs, gains = diversity_report(y, scores, pooled)
    rf = pooled.loc['rf']
    meaningful = []
    for name in WEIGHTS:
        candidate = pooled.loc[name]
        comparable = (abs(candidate.flagged_percent - rf.flagged_percent) <= .5
                      and abs(candidate.flagged_percent - 30) <= .5 and abs(rf.flagged_percent - 30) <= .5)
        meaningful.append({'ensemble': name, 'additional_defects': int(candidate.tp - rf.tp),
                           'recall_gain': candidate.recall - rf.recall,
                           'ap_change': candidate.average_precision - rf.average_precision,
                           'meaningful': bool(comparable and candidate.average_precision >= rf.average_precision
                               and (candidate.recall - rf.recall >= .02 - 1e-12 or candidate.tp - rf.tp >= 30))})
    return {'pooled': pooled, 'per-fold': per_fold, 'summary': summary, 'unique-defects': unique,
            'defect-patterns': patterns, 'ranking-overlap': pairs, 'ensemble-gains': gains,
            'meaningfulness': pd.DataFrame(meaningful),
            'ensemble-oof': scores.assign(target=y, outer_fold=fold_ids)}


def main():
    X_train, _, y_train, _ = split_dataset(load_jm1())
    probabilities, fold_ids, audits, manifest = load_trusted_oof(
        X_train, y_train, Path('docs/tree-model-tuning-results/oof-probabilities.csv'),
        Path('docs/final-model-challenge-results/oof-probabilities.csv'))
    results = ensemble_report(y_train, probabilities, fold_ids)
    destination = Path('docs/ensemble-ranking-results')
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in {**results, 'audits': audits}.items():
        frame.to_csv(destination / f'{name}.csv')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for name in ['pooled', 'unique-defects', 'ensemble-gains', 'meaningfulness']:
        print(name)
        print(results[name].to_string(float_format=lambda v: f'{v:.6f}'))


if __name__ == '__main__':
    main()
