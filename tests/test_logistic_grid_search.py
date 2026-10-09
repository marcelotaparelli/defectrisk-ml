import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from defectrisk import logistic_grid_search as experiment
from defectrisk.baseline import create_baseline_pipeline
from defectrisk.cross_validation import METRICS, training_folds
from defectrisk.data import TARGET
from defectrisk.split import feature_groups, split_dataset


@pytest.fixture
def training():
    X = pd.DataFrame({"a": np.repeat(np.arange(30, dtype=float), 3),
                      "b": np.repeat(np.arange(30) % 7, 3)}, index=range(1000, 1090))
    X.loc[[1000, 1001, 1002], "a"] = np.nan
    y = pd.Series(["false", "false", "true"] * 30, index=X.index, name=TARGET)
    return X, y


def test_only_requested_c_grid_and_unchanged_pipeline():
    search = experiment.create_grid_search()
    assert search.param_grid == {"classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
    assert search.estimator.named_steps["classifier"].get_params() == LogisticRegression(
        random_state=42, max_iter=5000, class_weight="balanced"
    ).get_params()
    reference = create_baseline_pipeline()
    for step in ("imputer", "scaler"):
        assert search.estimator.named_steps[step].get_params() == reference.named_steps[step].get_params()
    assert search.refit is False and search.error_score == "raise"
    assert set(search.scoring) == set(METRICS)
    assert isinstance(search.cv, StratifiedGroupKFold)
    assert search.cv.n_splits == 5 and search.cv.shuffle and search.cv.random_state == 42


def test_fixed_threshold_and_defective_label_for_all_metrics():
    y = ["false", "false", "true", "true"]
    p = [0.1, 0.6, 0.4, 0.5]
    for metric in METRICS:
        assert experiment.threshold_metric(y, p, metric=metric) == pytest.approx(0.5)
    assert experiment.threshold_metric(y, [0.1] * 4, metric="precision") == 0
    assert experiment.threshold_metric(y, [0.1] * 4, metric="recall") == 0
    assert experiment.threshold_metric(y, [0.1] * 4, metric="f1") == 0


def test_f1_selection_does_not_use_accuracy_or_recall():
    results = pd.DataFrame({"mean_f1": [0.4, 0.5, 0.5], "mean_accuracy": [0.9, 0.6, 0.8],
                            "mean_recall": [0.8, 0.6, 0.7]}, index=[0.01, 1.0, 10.0])
    assert experiment.best_c_by_f1(results) == 1.0


def test_groups_passed_to_real_grid_search_and_25_fold_only_fits(training, monkeypatch):
    X, y = training
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    search = experiment.create_grid_search()
    fit_search = search.fit
    seen_groups, fitted = [], []
    trusted = list(training_folds(X, y))

    def checked_search_fit(features, labels, *, groups):
        pd.testing.assert_series_equal(groups, feature_groups(X))
        pd.testing.assert_frame_equal(features, X)
        pd.testing.assert_series_equal(labels, y)
        seen_groups.append(groups.copy())
        return fit_search(features, labels, groups=groups)

    original_fit = LogisticRegression.fit

    def checked_classifier_fit(classifier, features, labels, *args, **kwargs):
        # Pipeline transformations precede this call; no full-training refit.
        assert classifier.C in experiment.C_VALUES and classifier.class_weight == "balanced"
        assert classifier.random_state == 42 and classifier.max_iter == 5000
        fit_indices = trusted[len(fitted) % 5][0]
        pd.testing.assert_series_equal(labels, y.iloc[fit_indices])
        assert len(features) == len(fit_indices) < len(X)
        imputed = X.iloc[fit_indices].fillna(X.iloc[fit_indices].median())
        expected = (imputed - imputed.mean()) / imputed.std(ddof=0)
        np.testing.assert_allclose(features, expected.to_numpy(), atol=1e-12)
        fitted.append(classifier.C)
        return original_fit(classifier, features, labels, *args, **kwargs)

    monkeypatch.setattr(search, "fit", checked_search_fit)
    monkeypatch.setattr(experiment, "create_grid_search", lambda: search)
    monkeypatch.setattr(LogisticRegression, "fit", checked_classifier_fit)
    results, audits, returned = experiment.run_logistic_grid_search(X, y)
    assert returned is search and len(seen_groups) == 1
    assert fitted == [value for value in experiment.C_VALUES for _ in range(5)]
    assert not hasattr(search, "best_estimator_")
    assert audits["shared_feature_vectors"].eq(0).all()
    assert audits["validation_rows"].sum() == len(X)
    assert results.index.tolist() == experiment.C_VALUES
    for metric in METRICS:
        splits = np.array([search.cv_results_[f"split{fold}_test_{metric}"] for fold in range(5)])
        np.testing.assert_allclose(results[f"mean_{metric}"], splits.mean(axis=0))
        np.testing.assert_allclose(results[f"std_{metric}"], splits.std(axis=0, ddof=0))
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_contamination_rejected_before_grid_fit(training, monkeypatch):
    X, y = training
    search = experiment.create_grid_search()
    folds = list(training_folds(X, y))
    fit, validation = folds[0]
    folds[0] = (np.append(fit, validation[0]), validation)
    monkeypatch.setattr(search.cv, "split", lambda *args, **kwargs: iter(folds))
    monkeypatch.setattr(search, "fit", lambda *args, **kwargs: pytest.fail("Must reject before search fit"))
    monkeypatch.setattr(experiment, "create_grid_search", lambda: search)
    with pytest.raises(RuntimeError, match="contamination"):
        experiment.run_logistic_grid_search(X, y)


def test_different_group_folds_rejected_before_grid_fit(training, monkeypatch):
    X, y = training
    search = experiment.create_grid_search()
    folds = list(training_folds(X, y))[::-1]
    monkeypatch.setattr(search.cv, "split", lambda *args, **kwargs: iter(folds))
    monkeypatch.setattr(search, "fit", lambda *args, **kwargs: pytest.fail("Must reject before search fit"))
    monkeypatch.setattr(experiment, "create_grid_search", lambda: search)
    with pytest.raises(RuntimeError, match="differ"):
        experiment.run_logistic_grid_search(X, y)


def test_command_never_inspects_final_test_outputs(training, monkeypatch, capsys):
    X, y = training
    frame = X.assign(**{TARGET: y})
    expected_X, _, expected_y, _ = split_dataset(frame)
    monkeypatch.setattr(experiment, "load_jm1", lambda: frame)

    class ForbiddenTestPartition:
        def __getattribute__(self, name):
            pytest.fail("Final test outputs must not be inspected")

    monkeypatch.setattr(experiment, "split_dataset", lambda _: (
        expected_X, ForbiddenTestPartition(), expected_y, ForbiddenTestPartition()
    ))

    def report(features, labels):
        pd.testing.assert_frame_equal(features, expected_X)
        pd.testing.assert_series_equal(labels, expected_y)
        return "Training-only C search"

    monkeypatch.setattr(experiment, "logistic_grid_report", report)
    experiment.main()
    assert capsys.readouterr().out.strip() == "Training-only C search"
