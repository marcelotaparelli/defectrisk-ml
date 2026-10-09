import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from defectrisk import final_evaluation as evaluation


def inputs():
    train = pd.DataFrame({'a': range(10), 'b': range(10)}, index=range(10))
    labels = pd.Series(['false', 'true'] * 5, index=train.index, name='defects')
    test = pd.DataFrame({'a': range(20, 30), 'b': range(20, 30)}, index=range(20, 30))
    targets = pd.Series(['true', 'true', 'false', 'false', 'false', 'false', 'false', 'false', 'false', 'true'], index=test.index, name='defects')
    return train, test, labels, targets


def records():
    params = {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 3,
              'max_features': 'sqrt', 'class_weight': 'balanced_subsample'}
    others = [{**params, 'max_depth': 16}, params, {**params, 'min_samples_leaf': 5},
              {**params, 'max_features': .7}, params]
    return pd.DataFrame({'model': ['tuned_weighted_random_forest']*5,
                         'outer_fold': range(1, 6), 'parameters': [repr(p) for p in others]})


def test_configuration_comes_from_existing_inner_winners_without_new_search(tmp_path):
    X, _, y, _ = inputs()
    frozen = evaluation.freeze_configuration(X, y, records(), tmp_path)
    params = evaluation.frozen_pipeline(frozen).named_steps['classifier'].get_params(deep=False)
    assert frozen['classifier_parameters'] == params
    assert params['class_weight'] == 'balanced_subsample'
    assert params['max_depth'] == 8 and params['n_estimators'] == 200
    assert params['min_samples_leaf'] == 3 and params['max_features'] == 'sqrt'
    assert params['random_state'] == 42 and not params['warm_start']
    assert frozen['selection']['new_searches'] == 0
    assert frozen['selection']['selected_in_folds'] == 2
    assert frozen['feature_columns'] == ['a', 'b']
    assert frozen['review_policy']['threshold_uses_labels'] is False
    assert frozen['preprocessing']['imputer']['strategy'] == 'median'
    assert frozen['preprocessing']['scaler']['with_std']
    before = (tmp_path / 'frozen-configuration.json').read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.freeze_configuration(X, y, records(), tmp_path)
    assert (tmp_path / 'frozen-configuration.json').read_bytes() == before


def spy_pipeline(monkeypatch, frozen, data):
    pipeline = evaluation.frozen_pipeline(frozen)
    calls = {'fit': 0, 'predict': 0}

    class Spy:
        named_steps = pipeline.named_steps
        classes_ = np.array(['false', 'true'])

        def fit(self, X, y):
            calls['fit'] += 1
            pd.testing.assert_frame_equal(X, data[0])
            pd.testing.assert_series_equal(y, data[2])
            return self

        def predict_proba(self, X):
            calls['predict'] += 1
            pd.testing.assert_frame_equal(X, data[1])
            probs = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .05])
            return np.column_stack([1-probs, probs])

    monkeypatch.setattr(evaluation, 'frozen_pipeline', lambda _: Spy())
    monkeypatch.setattr(evaluation.joblib, 'dump', lambda *_: None)
    return calls


def test_one_full_fit_one_prediction_and_cache_prevents_re_evaluation(tmp_path, monkeypatch):
    data = inputs()
    frozen = evaluation.freeze_configuration(data[0], data[2], records(), tmp_path)
    frozen_bytes = (tmp_path / 'frozen-configuration.json').read_bytes()
    calls = spy_pipeline(monkeypatch, frozen, data)
    result = evaluation.evaluate_once(lambda: data, tmp_path)
    assert calls == {'fit': 1, 'predict': 1}
    assert result['tp'] == 2 and result['fp'] == 1 and result['tn'] == 6 and result['fn'] == 1
    assert result['flagged_modules'] == 3 and result['flagged_percent'] == 30
    assert result['recall'] == pytest.approx(2/3)
    assert result['statement'] == evaluation.STATEMENT
    assert result['overlap']['shared_feature_vectors'] == 0
    cached = evaluation.evaluate_once(lambda: pytest.fail('Must not reload or rescore test'), tmp_path)
    assert cached == result and calls == {'fit': 1, 'predict': 1}
    assert (tmp_path / 'frozen-configuration.json').read_bytes() == frozen_bytes
    assert json.loads((tmp_path / 'evaluation-state.json').read_text())['status'] == 'complete'


def test_review_cutoff_does_not_use_labels(tmp_path, monkeypatch):
    cutoffs = []
    # Synthetic populations only; verify label changes cannot affect review policy.
    for i, flip in enumerate([False, True]):
        X, X_test, y, y_test = inputs()
        if flip:
            y_test = y_test.replace({'true': 'false', 'false': 'true'})
        data = (X, X_test, y, y_test)
        dest = tmp_path / str(i)
        frozen = evaluation.freeze_configuration(X, y, records(), dest)
        spy_pipeline(monkeypatch, frozen, data)
        result = evaluation.evaluate_once(lambda: data, dest)
        cutoffs.append(result['review_cutoff'])
    assert cutoffs[0] == cutoffs[1] == .7


def test_interrupted_attempt_cannot_be_retried(tmp_path):
    X, _, y, _ = inputs()
    evaluation.freeze_configuration(X, y, records(), tmp_path)
    (tmp_path / 'evaluation-state.json').write_text(json.dumps({'status': 'started'}))
    with pytest.raises(RuntimeError, match='forbidden'):
        evaluation.evaluate_once(lambda: pytest.fail('No data access'), tmp_path)


def test_training_change_rejected_before_fit(tmp_path, monkeypatch):
    X, test, y, targets = inputs()
    frozen = evaluation.freeze_configuration(X, y, records(), tmp_path)
    changed = X.copy()
    changed.iloc[0, 0] = -5
    spy_pipeline(monkeypatch, frozen, (X, test, y, targets))
    with pytest.raises(RuntimeError, match='Training data differs'):
        evaluation.evaluate_once(lambda: (changed, test, y, targets), tmp_path)


def test_overlap_rejected_before_fit(tmp_path, monkeypatch):
    X, _, y, targets = inputs()
    test = X.copy()
    targets.index = test.index
    frozen = evaluation.freeze_configuration(X, y, records(), tmp_path)
    calls = spy_pipeline(monkeypatch, frozen, (X, test, y, targets))
    with pytest.raises(RuntimeError, match='contamination'):
        evaluation.evaluate_once(lambda: (X, test, y, targets), tmp_path)
    assert calls == {'fit': 0, 'predict': 0}
