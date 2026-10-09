"""Persist a frozen configuration before a single, guarded held-out evaluation."""

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from defectrisk.audit import feature_overlap
from defectrisk.baseline import POSITIVE_CLASS
from defectrisk.data import TARGET, load_jm1
from defectrisk.gradient_boosting_baseline import review_threshold
from defectrisk.random_forest_class_weight import create_weight_comparison_pipeline
from defectrisk.split import split_dataset
from defectrisk.threshold_experiment import evaluate_thresholds, apply_threshold

DESTINATION = Path('docs/final-evaluation-results')
STATEMENT = 'This held-out test was evaluated once after model selection was frozen.'


def now():
    return datetime.now(timezone.utc).isoformat()


def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return 'NaN'
    return value


def fingerprint(X, y):
    if TARGET in X or not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError('Training features must exclude target and align to unique label indices.')
    digest = hashlib.sha256()
    digest.update(json.dumps(list(X.columns)).encode())
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(y, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def existing_configuration(records):
    """Plurality of previous inner-CV winners, no new search or outer scoring."""
    rows = records[records.model == 'tuned_weighted_random_forest']
    if len(rows) != 5 or set(rows.outer_fold) != {1, 2, 3, 4, 5}:
        raise ValueError('Exactly the five trusted RF selection records are required.')
    configurations = [ast.literal_eval(value) for value in rows.parameters]
    keys = [json.dumps(params, sort_keys=True) for params in configurations]
    counts = Counter(keys)
    selected = max(range(len(keys)), key=lambda i: counts[keys[i]])
    return configurations[selected], counts[keys[selected]]


def freeze_configuration(X_train, y_train, records, destination=DESTINATION):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    params, count = existing_configuration(records)
    pipeline = create_weight_comparison_pipeline('weighted_random_forest')
    pipeline.set_params(**{f'classifier__{key}': value for key, value in params.items()})
    frozen = {
        'frozen_at_utc': now(), 'model': 'RandomForestClassifier',
        'selection': {'rule': 'Most frequent existing inner-CV winner; ties use earliest fold',
                      'selected_in_folds': count, 'total_folds': 5,
                      'source': 'docs/final-model-challenge-results/selected-parameters.csv',
                      'new_searches': 0},
        'classifier_parameters': pipeline.named_steps['classifier'].get_params(deep=False),
        'preprocessing': {step: normalized(pipeline.named_steps[step].get_params(deep=False))
                          for step in ['imputer', 'scaler']},
        'feature_columns': list(X_train.columns), 'target': TARGET, 'positive_class': POSITIVE_CLASS,
        'training': {'definition': 'Complete unchanged M3 group-aware 80/20 training partition of OpenML 1053 version 1; all rows retained',
                     'rows': len(X_train), 'defective': int((y_train == POSITIVE_CLASS).sum()),
                     'clean': int((y_train == 'false').sum()), 'sha256': fingerprint(X_train, y_train),
                     'split_random_state': 42, 'test_size': .2, 'groups': 'All original feature columns; excludes target'},
        'review_policy': {'capacity': .30, 'rule': 'Rank by P(true); choose closest attainable flagged count; retain all equal scores; distance ties prefer fewer reviews',
                          'threshold_uses_labels': False, 'comparison': 'probability >= chosen score cutoff'},
        'versions': {'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
    }
    with (destination / 'frozen-configuration.json').open('x') as handle:
        json.dump(frozen, handle, indent=2, allow_nan=False)
    return frozen


def frozen_pipeline(frozen):
    pipeline = create_weight_comparison_pipeline('weighted_random_forest')
    pipeline.set_params(classifier=RandomForestClassifier(**frozen['classifier_parameters']))
    for step, recorded in frozen['preprocessing'].items():
        if normalized(pipeline.named_steps[step].get_params(deep=False)) != recorded:
            raise RuntimeError('Preprocessing differs from frozen configuration.')
    return pipeline


def evaluate_once(loader, destination=DESTINATION):
    """One full-training fit, one test probability call, one fixed-policy scoring.

    Completed results are returned from disk without loading data or fitting.
    An interrupted attempt is never automatically retried.
    """
    destination = Path(destination)
    frozen_path = destination / 'frozen-configuration.json'
    frozen_bytes = frozen_path.read_bytes()
    frozen = json.loads(frozen_bytes)
    state_path = destination / 'evaluation-state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['status'] == 'complete':
            return json.loads((destination / 'result.json').read_text())
        raise RuntimeError('A held-out evaluation attempt already began; re-evaluation is forbidden.')
    pipeline = frozen_pipeline(frozen)
    if sklearn.__version__ != frozen['versions']['sklearn']:
        raise RuntimeError('Sklearn version differs from freeze.')
    with state_path.open('x') as handle:
        json.dump({'status': 'started', 'started_at_utc': now(),
                   'frozen_configuration_sha256': hashlib.sha256(frozen_bytes).hexdigest()}, handle, indent=2)
    X_train, X_test, y_train, y_test = loader()
    if fingerprint(X_train, y_train) != frozen['training']['sha256']:
        raise RuntimeError('Training data differs from frozen definition.')
    if list(X_train.columns) != frozen['feature_columns'] or not X_train.columns.equals(X_test.columns):
        raise RuntimeError('Frozen feature schema differs.')
    overlap = feature_overlap(X_train, X_test)
    if overlap['shared_feature_vectors']:
        raise RuntimeError('Train/test duplicate feature-vector contamination.')
    pipeline.fit(X_train, y_train)
    if pipeline.named_steps['classifier'].get_params(deep=False) != frozen['classifier_parameters']:
        raise RuntimeError('Classifier settings changed during fitting.')
    column = list(pipeline.classes_).index(frozen['positive_class'])
    probabilities = pd.Series(pipeline.predict_proba(X_test)[:, column], index=X_test.index, name='defective_probability')
    cutoff = review_threshold(probabilities, frozen['review_policy']['capacity'])
    metrics = evaluate_thresholds(y_test, probabilities, thresholds=[cutoff]).iloc[0].to_dict()
    from sklearn.metrics import average_precision_score
    metrics['average_precision'] = average_precision_score(y_test == POSITIVE_CLASS, probabilities)
    result = {'evaluated_at_utc': now(), 'test_rows': len(X_test),
              'test_defective': int((y_test == POSITIVE_CLASS).sum()),
              'review_cutoff': cutoff, **metrics, 'overlap': overlap,
              'full_training_fits': 1, 'test_predict_proba_calls': 1,
              'statement': STATEMENT, 'frozen_configuration_sha256': hashlib.sha256(frozen_bytes).hexdigest()}
    if frozen_path.read_bytes() != frozen_bytes:
        raise RuntimeError('Frozen configuration was modified.')
    predictions = apply_threshold(probabilities, cutoff)
    pd.DataFrame({'defective_probability': probabilities, 'flagged': predictions == POSITIVE_CLASS,
                  'target': y_test}).to_csv(destination / 'test-predictions.csv')
    joblib.dump(pipeline, destination / 'fitted-model.joblib')
    with (destination / 'result.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    state = json.loads(state_path.read_text())
    state.update(status='complete', completed_at_utc=now(), full_training_fits=1, test_predict_proba_calls=1)
    state_path.write_text(json.dumps(state, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'evaluate'])
    args = parser.parse_args()
    if args.phase == 'freeze':
        X_train, _, y_train, _ = split_dataset(load_jm1())
        records = pd.read_csv('docs/final-model-challenge-results/selected-parameters.csv')
        result = freeze_configuration(X_train, y_train, records)
    else:
        result = evaluate_once(lambda: split_dataset(load_jm1()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
