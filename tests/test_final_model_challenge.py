import numpy as np
import pandas as pd
import pytest

from defectrisk import final_model_challenge as experiment
from defectrisk.cross_validation import training_folds
from defectrisk.data import TARGET


def training():
    group = np.repeat(np.arange(30, dtype=float), 3)
    X = pd.DataFrame({'a': group, 'b': group/3, 'c': group % 7}, index=range(1000, 1090))
    X.iloc[:3, 0] = np.nan
    y = pd.Series(['false', 'false', 'true']*30, index=X.index, name=TARGET)
    return X, y


def test_search_is_small_deterministic_and_covers_only_requested_values():
    candidates = experiment.xgboost_candidates()
    assert candidates == experiment.xgboost_candidates()
    assert len(candidates) == len({repr(params) for params in candidates}) == 8
    for params in candidates:
        assert set(params) == set(experiment.SPACE)
        assert all(value in experiment.SPACE[key] for key, value in params.items())
    for key, values in experiment.SPACE.items():
        assert {params[key] for params in candidates} == set(values)


def test_training_weight_binary_target_pipeline_and_reproducibility():
    X, y = training()
    params = experiment.xgboost_candidates()[0]
    pipeline = experiment.create_xgboost_pipeline(params, y)
    classifier = pipeline.named_steps['classifier']
    assert classifier.scale_pos_weight == 2
    assert classifier.random_state == 42 and classifier.n_jobs == 1
    assert classifier.tree_method == 'hist'
    assert classifier.objective == 'binary:logistic'
    assert list(pipeline.named_steps) == ['imputer', 'classifier']
    first = experiment.xgboost_probabilities(params, X.iloc[:60], y.iloc[:60], X.iloc[60:])
    second = experiment.xgboost_probabilities(params, X.iloc[:60], y.iloc[:60], X.iloc[60:])
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all() and ((first >= 0) & (first <= 1)).all()
    with pytest.raises(ValueError, match='classes'):
        experiment.create_xgboost_pipeline(params, pd.Series(['true']*5))
    with pytest.raises(ValueError, match='parameters'):
        experiment.create_xgboost_pipeline({**params, 'reg_lambda': 1}, y)
    with pytest.raises(ValueError, match='Target'):
        experiment.xgboost_probabilities(params, X.assign(defects=y), y, X)


def decision_table(recall_gain=0., caught_gain=0, ap_gain=0., workload=30.):
    return pd.DataFrame({'recall': [.56, .56+recall_gain], 'tp': [950, 950+caught_gain],
                         'average_precision': [.4, .4+ap_gain], 'flagged_percent': [30., workload]},
                        index=[experiment.MODELS[0], 'xgboost'])


@pytest.mark.parametrize('recall,caught,ap,workload,expected', [
    (.02, 29, 0, 30, True), (.019, 30, 0, 30, True), (.019, 29, .01, 30, False),
    (.03, 50, -.01, 30, False), (.03, 50, .01, 31, False), (0, 0, 0, 30, False),
])
def test_predeclared_materiality_and_ap_workload_guards(recall, caught, ap, workload, expected):
    result = experiment.final_decision(decision_table(recall, caught, ap, workload))
    assert result['material_improvement'] is expected
    assert result['selected_model'] == ('xgboost' if expected else experiment.MODELS[0])
    assert result['model_exploration_finished']


def test_inner_selection_uses_capacity_recall_not_accuracy(monkeypatch):
    X, y = training()
    candidates = experiment.xgboost_candidates()[:3]
    scores = [dict(recall=.4, average_precision=.9, accuracy=.99),
              dict(recall=.6, average_precision=.3, accuracy=.5),
              dict(recall=.6, average_precision=.4, accuracy=.4)]
    monkeypatch.setattr(experiment, 'xgboost_inner_score', lambda params, *_: scores[candidates.index(params)])
    params, scores = experiment.select_xgboost(candidates, X, y, [])
    assert params == candidates[2]
    assert scores.selected.tolist() == [False, False, True]


def test_nested_selection_weights_and_single_ablation(monkeypatch):
    X, y = training()
    candidates = experiment.xgboost_candidates()[:1]
    calls, selections = [], []
    real_select = experiment.select_xgboost

    def fake_probabilities(params, features, labels, validation, *, without_b=False):
        assert TARGET not in features
        assert not features.index.intersection(validation.index).size
        pd.testing.assert_series_equal(labels, y.loc[features.index])
        pipeline = experiment.create_xgboost_pipeline(params, labels)
        expected = (labels == 'false').sum() / (labels == 'true').sum()
        assert pipeline.named_steps['classifier'].scale_pos_weight == expected
        calls.append((features.index, validation.index, without_b, params))
        return (validation.index.to_numpy() % 11 + 1) / 13

    def selected(params, features, labels, folds, **kwargs):
        selections.append(features.index)
        return real_select(params, features, labels, folds, **kwargs)

    monkeypatch.setattr(experiment, 'xgboost_probabilities', fake_probabilities)
    monkeypatch.setattr(experiment, 'select_xgboost', selected)
    result, decision = experiment.challenge(X, y, rf_candidates=[{}], xgb_candidates=candidates)
    assert len(calls) == 25  # 15 inner + 5 outer + 5 b-only ablation; no ablation search.
    assert sum(call[2] for call in calls) == 5
    for i, (fit, validation) in enumerate(training_folds(X, y)):
        assert selections[i].equals(X.iloc[fit].index)
        outer = [call for call in calls if call[0].equals(X.iloc[fit].index)]
        assert len(outer) == 2 and {call[2] for call in outer} == {True, False}
        for record in outer:
            assert record[1].equals(X.iloc[validation].index)
            assert record[3] == candidates[0]
    for name in ['audits', 'ablation-audits']:
        assert len(result[name]) == 20 and result[name].shared_feature_vectors.eq(0).all()
    weights = result['fit-weights']
    np.testing.assert_allclose(weights.scale_pos_weight, weights.fit_negative / weights.fit_positive)
    assert len(result['oof-probabilities']) == len(X) and not result['oof-probabilities'].isna().any().any()
    assert (result['pooled'].tp + result['pooled'].fn).eq(30).all()
    assert (result['pooled'].tp + result['pooled'].fp).equals(result['pooled'].flagged_modules)


def test_b_actually_excluded_and_fit_weight_uses_only_fit_labels(monkeypatch):
    X, y = training()
    seen = {}
    original_factory = experiment.create_xgboost_pipeline

    def checked_factory(params, labels):
        pipeline = original_factory(params, labels)
        fit = pipeline.fit

        def checked_fit(features, binary):
            assert 'b' not in features and TARGET not in features
            assert set(binary) == {0, 1}
            fit(features, binary)
            np.testing.assert_allclose(pipeline.named_steps['imputer'].statistics_, features.median())
            seen['weight'] = pipeline.named_steps['classifier'].scale_pos_weight
            return pipeline

        monkeypatch.setattr(pipeline, 'fit', checked_fit)
        return pipeline

    monkeypatch.setattr(experiment, 'create_xgboost_pipeline', checked_factory)
    fit_labels = y.iloc[:60].copy()
    fit_labels.iloc[:10] = 'true'
    experiment.xgboost_probabilities(experiment.xgboost_candidates()[0], X.iloc[:60], fit_labels, X.iloc[60:], without_b=True)
    assert seen['weight'] == (fit_labels == 'false').sum() / (fit_labels == 'true').sum()


def test_main_never_accesses_final_test(monkeypatch, tmp_path):
    X, y = training()

    class Locked:
        def __getattribute__(self, name):
            pytest.fail('Final test must remain untouched')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, 'load_jm1', lambda: None)
    monkeypatch.setattr(experiment, 'split_dataset', lambda _: (X, Locked(), y, Locked()))

    def checked(features, labels, **kwargs):
        assert features is X and labels is y
        return {'pooled': pd.DataFrame({'value': [1]})}, {'model_exploration_finished': True}

    monkeypatch.setattr(experiment, 'challenge', checked)
    experiment.main()
    assert (tmp_path / 'docs/final-model-challenge-results/decision.json').exists()


def test_b_ablation_cannot_create_cross_boundary_duplicates(monkeypatch):
    X, y = training()
    X[['a', 'c']] = 0.
    monkeypatch.setattr(experiment, 'select_candidate', lambda *_args, **_kwargs: pytest.fail('Reject before tuning'))
    with pytest.raises(RuntimeError, match='Excluding b'):
        experiment.challenge(X, y)
