import numpy as np
import pandas as pd
import pytest

from defectrisk import cross_validation as cv
from defectrisk.audit import feature_overlap
from defectrisk.baseline import create_baseline_pipeline
from defectrisk.data import TARGET, load_jm1
from defectrisk.split import feature_groups, split_dataset


@pytest.fixture
def training():
    # Many conflicting-label duplicate groups, plus a missing-feature group.
    X = pd.DataFrame({"a": np.repeat(np.arange(60, dtype=float), 2),
                      "b": np.repeat(np.arange(60) % 7, 2)}, index=range(1000, 1120))
    X.loc[[1000, 1001], "a"] = np.nan
    y = pd.Series(["false", "true"] * 60, index=X.index, name=TARGET)
    return X, y


@pytest.mark.parametrize("source", ["synthetic", pytest.param("jm1", marks=pytest.mark.integration)])
def test_fold_integrity_coverage_balance_and_reproducibility(source, training):
    if source == "jm1":
        X, X_test, y, _ = split_dataset(load_jm1())
    else:
        X, y = training
        X_test = X.iloc[:0]
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    folds = list(cv.training_folds(X, y))
    assert len(folds) == 5
    validation_rows = []
    groups = feature_groups(X)
    for (fit, validation), (repeat_fit, repeat_validation) in zip(folds, cv.training_folds(X, y)):
        np.testing.assert_array_equal(fit, repeat_fit)
        np.testing.assert_array_equal(validation, repeat_validation)
        assert not set(fit) & set(validation)
        assert set(fit) | set(validation) == set(range(len(X)))
        assert TARGET not in X.iloc[fit] and TARGET not in X.iloc[validation]
        assert feature_overlap(X.iloc[fit], X.iloc[validation])["shared_feature_vectors"] == 0
        assert not set(X.iloc[fit].index) & set(X_test.index)
        assert not set(X.iloc[validation].index) & set(X_test.index)
        # Check all groups, including conflicting labels and matching NaNs.
        assert not set(groups.iloc[fit]) & set(groups.iloc[validation])
        assert len(validation) / len(X) == pytest.approx(0.20, abs=0.03)
        for indices in (fit, validation):
            assert (y.iloc[indices] == "true").mean() == pytest.approx((y == "true").mean(), abs=0.03)
        validation_rows.extend(validation)
    assert sorted(validation_rows) == list(range(len(X)))
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_fresh_pipeline_preprocessing_fits_only_fold_training(training, monkeypatch):
    X, y = training
    pipelines, seen_fit, seen_validation = [], [], []

    def factory():
        pipeline = create_baseline_pipeline()
        assert not hasattr(pipeline.named_steps["classifier"], "classes_")
        fit, predict = pipeline.fit, pipeline.predict

        def checked_fit(features, labels):
            assert features.index.equals(labels.index)
            seen_fit.append(features.index.tolist())
            fit(features, labels)
            imputer = pipeline.named_steps["imputer"]
            np.testing.assert_allclose(imputer.statistics_, features.median().to_numpy())
            np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_,
                                       imputer.transform(features).mean(axis=0))
            return pipeline

        def checked_predict(features):
            seen_validation.append(features.index.tolist())
            return predict(features)

        monkeypatch.setattr(pipeline, "fit", checked_fit)
        monkeypatch.setattr(pipeline, "predict", checked_predict)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(cv, "create_baseline_pipeline", factory)
    folds, summary = cv.cross_validate_baseline(X, y)
    assert len({id(pipeline) for pipeline in pipelines}) == 5
    for fit, validation in zip(seen_fit, seen_validation):
        assert not set(fit) & set(validation)
        assert set(fit) | set(validation) == set(X.index)
    assert folds["shared_feature_vectors"].eq(0).all()
    assert summary.index.tolist() == cv.METRICS


def test_positive_label_metrics_and_population_summary(training, monkeypatch):
    X, y = training

    class FixedPredictions:
        def fit(self, features, labels):
            return self

        def predict(self, features):
            # Every validation group has one clean and one defective row.
            # Predict all defective: accuracy=.5, precision=.5, recall=1, F1=2/3.
            return np.repeat("true", len(features))

    monkeypatch.setattr(cv, "create_baseline_pipeline", FixedPredictions)
    folds, summary = cv.cross_validate_baseline(X, y)
    for metric, expected in zip(cv.METRICS, [0.5, 0.5, 1.0, 2 / 3]):
        np.testing.assert_allclose(folds[metric], expected)
        assert summary.loc[metric, "mean"] == pytest.approx(expected)
        assert summary.loc[metric, "std"] == pytest.approx(0)
    # Independently vary predictions by fold to exercise nonzero std.
    calls = iter(["true", "false", "true", "false", "true"])
    monkeypatch.setattr(FixedPredictions, "predict", lambda self, features: np.repeat(next(calls), len(features)))
    folds, summary = cv.cross_validate_baseline(X, y)
    assert summary.loc["recall", "mean"] == pytest.approx(0.6)
    assert summary.loc["recall", "std"] == pytest.approx(np.std([1, 0, 1, 0, 1], ddof=0))


def test_contaminated_fold_is_rejected_before_fitting(training, monkeypatch):
    X, y = training
    monkeypatch.setattr(cv, "training_folds", lambda *_: iter([(np.array([0]), np.array([1]))]))
    monkeypatch.setattr(cv, "create_baseline_pipeline", lambda: pytest.fail("Must reject before creating pipeline"))
    with pytest.raises(RuntimeError, match="contamination"):
        cv.cross_validate_baseline(X, y)


@pytest.mark.parametrize("problem", ["target", "alignment", "missing_label", "few_groups", "duplicate_index"])
def test_invalid_training_input_is_rejected(training, problem):
    X, y = training
    if problem == "target":
        X[TARGET] = y
    elif problem == "alignment":
        y = y.iloc[::-1]
    elif problem == "missing_label":
        y.iloc[0] = None
    elif problem == "few_groups":
        X = X.iloc[:8]
        y = y.iloc[:8]
    else:
        X.index = [0] * len(X)
        y.index = X.index
    with pytest.raises(ValueError):
        list(cv.training_folds(X, y))


def test_command_passes_only_m3_training_partition(training, monkeypatch, capsys):
    X, y = training
    frame = X.assign(**{TARGET: y})
    expected_X, _, expected_y, _ = split_dataset(frame)
    monkeypatch.setattr(cv, "load_jm1", lambda: frame)

    def report(features, labels):
        pd.testing.assert_frame_equal(features, expected_X)
        pd.testing.assert_series_equal(labels, expected_y)
        return "Training-only CV"

    monkeypatch.setattr(cv, "cross_validation_report", report)
    cv.main()
    assert capsys.readouterr().out.strip() == "Training-only CV"
