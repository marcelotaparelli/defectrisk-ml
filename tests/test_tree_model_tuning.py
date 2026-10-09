import numpy as np
import pandas as pd
import pytest

from defectrisk import tree_model_tuning as experiment
from defectrisk.cross_validation import training_folds
from defectrisk.gradient_boosting_baseline import evaluate_review_budgets


def training():
    X = pd.DataFrame({"a": np.repeat(np.arange(30, dtype=float), 3),
                      "b": np.repeat(np.arange(30) % 7, 3)}, index=range(1000, 1090))
    X.iloc[:3, 0] = np.nan
    y = pd.Series(["false", "false", "true"] * 30, index=X.index, name="defects")
    return X, y


def test_limited_reproducible_search_only_allowed_parameters():
    for model, space in [("weighted_random_forest", experiment.RF_SPACE),
                         ("hist_gradient_boosting", experiment.HGB_SPACE)]:
        candidates = experiment.search_candidates(model)
        assert candidates == experiment.search_candidates(model)
        assert len(candidates) == 9 and candidates[0] == {}
        assert len({repr(params) for params in candidates}) == 9
        for key, values in space.items():
            assert set(params[key] for params in candidates[1:]) == set(values)
        for params in candidates[1:]:
            assert set(params) == set(space)
            assert all(value in space[key] for key, value in params.items())
    with pytest.raises(ValueError, match="Unknown"):
        experiment.search_candidates("other")


def test_selection_prioritizes_budget_recall_then_ap_not_accuracy(monkeypatch):
    # Highest accuracy and even highest AP do not win over higher capacity recall.
    scores = [dict(recall=.4, average_precision=.8, accuracy=.99),
              dict(recall=.6, average_precision=.3, accuracy=.5),
              dict(recall=.6, average_precision=.4, accuracy=.4)]
    monkeypatch.setattr(experiment, "candidate_score", lambda model, params, *_: scores[params['id']])
    X, y = training()
    candidates = [{'id': i} for i in range(3)]
    best, results = experiment.select_candidate('model', candidates, X, y, [])
    assert best == {'id': 2}
    assert results.selected.tolist() == [False, False, True]


def test_nested_selection_never_sees_outer_validation_and_oof_complete(monkeypatch):
    X, y = training()
    originals = X.copy(deep=True), y.copy(deep=True)
    calls, selection_inputs = [], []
    actual_select = experiment.select_candidate

    def checked_select(model, candidates, features, labels, folds, **kwargs):
        selection_inputs.append(features.index)
        pd.testing.assert_series_equal(labels, y.loc[features.index])
        return actual_select(model, candidates, features, labels, folds, **kwargs)

    def probabilities(model, params, features, labels, validation):
        assert "defects" not in features
        assert not features.index.intersection(validation.index).size
        pd.testing.assert_series_equal(labels, y.loc[features.index])
        calls.append((model, features.index, validation.index))
        return (validation.index.to_numpy() % 11 + 1) / 13

    monkeypatch.setattr(experiment, 'select_candidate', checked_select)
    monkeypatch.setattr(experiment, 'fitted_probabilities', probabilities)
    candidates = {model: [{}] for model in experiment.FAMILIES.values()}
    results = experiment.nested_tree_comparison(X, y, candidates=candidates)
    assert len(calls) == 55  # 25 outer fits + 30 inner fits, no per-budget fitting.
    assert len(results['audits']) == 20
    assert results['audits'].shared_feature_vectors.eq(0).all()
    oof = results['oof-probabilities']
    assert oof.index.equals(X.index)
    assert np.isfinite(oof[list(experiment.REFERENCES) + list(experiment.FAMILIES)].to_numpy()).all()
    for number, (fit, validation) in enumerate(training_folds(X, y), 1):
        for indices in selection_inputs[(number - 1) * 2:number * 2]:
            assert indices.equals(X.iloc[fit].index)
            assert not indices.intersection(X.iloc[validation].index).size
        # Each outer validation row is scored once, after selection, by every model.
        for model in (*experiment.REFERENCES, *experiment.FAMILIES.values()):
            assert any(fit_index.equals(X.iloc[fit].index) and val_index.equals(X.iloc[validation].index)
                       for name, fit_index, val_index in calls if name == model)
        assert oof.loc[X.iloc[validation].index, 'outer_fold'].eq(number).all()
    scores = results['pooled']
    assert (scores.tp + scores.fn).eq(30).all()
    assert (scores.tp + scores.fp).equals(scores.flagged_modules)
    pd.testing.assert_frame_equal(X, originals[0])
    pd.testing.assert_series_equal(y, originals[1])


def test_candidate_scores_use_pooled_inner_oof_not_fitting_predictions(monkeypatch):
    X, y = training()
    monkeypatch.setattr(experiment, 'fitted_probabilities',
                        lambda model, params, fit, labels, validation: (validation.index % 11 + 1).to_numpy() / 13)
    scores = experiment.candidate_score('weighted_random_forest', {}, X, y, list(training_folds(X, y)))
    expected = evaluate_review_budgets(y, pd.DataFrame({'model': (X.index % 11 + 1).to_numpy() / 13}, index=X.index), budgets=[.3]).iloc[0]
    assert scores['recall'] == expected.recall
    assert scores['flagged_modules'] == expected.flagged_modules


@pytest.mark.parametrize('level', ['outer', 'inner'])
def test_contamination_rejected_before_any_fit(level):
    X, y = training()
    folds = list(training_folds(X, y))
    fit, validation = folds[0]
    folds[0] = (np.append(fit[:-1], validation[0]), validation)
    with pytest.raises(RuntimeError, match='contamination|partition'):
        experiment.audited_folds(X, y, folds, level, 1)


def test_incomplete_oof_rejected():
    X, y = training()
    with pytest.raises(RuntimeError, match='Every row'):
        experiment.audited_folds(X, y, list(training_folds(X, y))[:-1], 'inner', 1)


def test_main_never_inspects_final_test(monkeypatch, tmp_path):
    X, y = training()

    class Locked:
        def __getattribute__(self, name):
            pytest.fail('Final test must remain untouched')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, 'load_jm1', lambda: None)
    monkeypatch.setattr(experiment, 'split_dataset', lambda _: (X, Locked(), y, Locked()))

    def checked(features, labels, **kwargs):
        assert features is X and labels is y
        return {'pooled': pd.DataFrame({'value': [1]}), 'selected-parameters': pd.DataFrame({'value': [1]})}

    monkeypatch.setattr(experiment, 'nested_tree_comparison', checked)
    experiment.main()
    assert (tmp_path / 'docs/tree-model-tuning-results/pooled.csv').exists()
