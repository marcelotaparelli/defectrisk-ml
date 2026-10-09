import numpy as np
import pandas as pd
import pytest

from defectrisk import class_weight_experiment as experiment
from defectrisk import cross_validation as cv
from defectrisk.audit import feature_overlap
from defectrisk.baseline import create_baseline_pipeline
from defectrisk.data import TARGET
from defectrisk.split import split_dataset


@pytest.fixture
def training():
    X = pd.DataFrame({"a": np.repeat(np.arange(40, dtype=float), 3),
                      "b": np.repeat(np.arange(40) % 7, 3)}, index=range(1000, 1120))
    X.loc[[1000, 1001, 1002], "a"] = np.nan
    y = pd.Series(["false", "false", "true"] * 40, index=X.index, name=TARGET)
    return X, y


def test_only_class_weight_changes():
    baseline = experiment.create_experiment_pipeline()
    balanced = experiment.create_experiment_pipeline(balanced=True)
    reference = create_baseline_pipeline()
    assert list(baseline.named_steps) == list(balanced.named_steps) == list(reference.named_steps)
    for step in reference.named_steps:
        left = baseline.named_steps[step].get_params()
        right = balanced.named_steps[step].get_params()
        assert left == reference.named_steps[step].get_params()
        if step == "classifier":
            assert left["class_weight"] is None
            assert right.pop("class_weight") == "balanced"
            left.pop("class_weight")
        assert left == right


def test_paired_folds_fresh_preprocessing_and_preserved_data(training, monkeypatch):
    X, y = training
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    original_factory = experiment.create_experiment_pipeline
    seen = []

    def factory(*, balanced=False):
        pipeline = original_factory(balanced=balanced)
        assert not hasattr(pipeline.named_steps["classifier"], "classes_")
        fit, predict = pipeline.fit, pipeline.predict
        record = {"pipeline": pipeline, "balanced": balanced}
        seen.append(record)

        def checked_fit(features, labels):
            assert features.index.equals(labels.index)
            record["fit"] = features.copy()
            record["labels"] = labels.copy()
            fit(features, labels)
            imputer = pipeline.named_steps["imputer"]
            np.testing.assert_allclose(imputer.statistics_, features.median().to_numpy())
            np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_,
                                       imputer.transform(features).mean(axis=0))
            return pipeline

        def checked_predict(features):
            record["validation"] = features.copy()
            return predict(features)

        monkeypatch.setattr(pipeline, "fit", checked_fit)
        monkeypatch.setattr(pipeline, "predict", checked_predict)
        return pipeline

    monkeypatch.setattr(experiment, "create_experiment_pipeline", factory)
    folds, summary, matrices = experiment.compare_class_weights(X, y)
    assert len({id(record["pipeline"]) for record in seen}) == 10
    for fold, ((fit_indices, validation_indices), baseline, balanced) in enumerate(
        zip(cv.training_folds(X, y), seen[::2], seen[1::2]), 1
    ):
        assert not baseline["balanced"] and balanced["balanced"]
        for record in (baseline, balanced):
            pd.testing.assert_frame_equal(record["fit"], X.iloc[fit_indices])
            pd.testing.assert_series_equal(record["labels"], y.iloc[fit_indices])
            pd.testing.assert_frame_equal(record["validation"], X.iloc[validation_indices])
            assert feature_overlap(record["fit"], record["validation"])["shared_feature_vectors"] == 0
    assert folds["shared_feature_vectors"].eq(0).all()
    for name in ("baseline", "balanced"):
        assert matrices[name].sum() == len(X)
        np.testing.assert_array_equal(matrices[name].sum(axis=1), [80, 40])
        np.testing.assert_array_equal(matrices[name].ravel(), folds.loc[name, ["tn", "fp", "fn", "tp"]].sum())
        for metric in cv.METRICS:
            assert summary.loc[(name, metric), "mean"] == pytest.approx(folds.loc[name, metric].mean())
            assert summary.loc[(name, metric), "std"] == pytest.approx(folds.loc[name, metric].std(ddof=0))
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_metrics_and_aggregate_matrix_use_defective_as_positive(training, monkeypatch):
    X, y = training

    class FixedPredictions:
        def __init__(self, balanced):
            self.balanced = balanced

        def fit(self, features, labels):
            return self

        def predict(self, features):
            return np.repeat("true" if self.balanced else "false", len(features))

    monkeypatch.setattr(experiment, "create_experiment_pipeline",
                        lambda *, balanced=False: FixedPredictions(balanced))
    folds, summary, matrices = experiment.compare_class_weights(X, y)
    for name, expected in [("baseline", [2 / 3, 0, 0, 0]), ("balanced", [1 / 3, 1 / 3, 1, 0.5])]:
        for metric, value in zip(cv.METRICS, expected):
            np.testing.assert_allclose(folds.loc[name, metric], value)
            assert summary.loc[(name, metric), "mean"] == pytest.approx(value)
    np.testing.assert_array_equal(matrices["baseline"], [[80, 0], [40, 0]])
    np.testing.assert_array_equal(matrices["balanced"], [[0, 80], [0, 40]])


def test_contamination_rejected_before_either_model(training, monkeypatch):
    X, y = training
    monkeypatch.setattr(experiment, "training_folds", lambda *_: iter([(np.array([0]), np.array([1]))]))
    monkeypatch.setattr(experiment, "create_experiment_pipeline", lambda **_: pytest.fail("Must reject before fit"))
    with pytest.raises(RuntimeError, match="contamination"):
        experiment.compare_class_weights(X, y)


def test_command_passes_only_m3_training_partition(training, monkeypatch, capsys):
    X, y = training
    frame = X.assign(**{TARGET: y})
    expected_X, _, expected_y, _ = split_dataset(frame)
    monkeypatch.setattr(experiment, "load_jm1", lambda: frame)

    def report(features, labels):
        pd.testing.assert_frame_equal(features, expected_X)
        pd.testing.assert_series_equal(labels, expected_y)
        return "Paired training-only experiment"

    monkeypatch.setattr(experiment, "class_weight_report", report)
    experiment.main()
    assert capsys.readouterr().out.strip() == "Paired training-only experiment"
