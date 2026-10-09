import numpy as np
import pandas as pd
import pytest

from defectrisk import threshold_experiment as experiment
from defectrisk.class_weight_experiment import create_experiment_pipeline
from defectrisk.cross_validation import training_folds
from defectrisk.data import TARGET
from defectrisk.split import split_dataset


@pytest.fixture
def training():
    X = pd.DataFrame({"a": np.repeat(np.arange(40, dtype=float), 3),
                      "b": np.repeat(np.arange(40) % 7, 3)}, index=range(1000, 1120))
    X.loc[[1000, 1001, 1002], "a"] = np.nan
    y = pd.Series(["false", "false", "true"] * 40, index=X.index, name=TARGET)
    return X, y


def test_threshold_equality_and_endpoints():
    np.testing.assert_array_equal(experiment.apply_threshold([0.0, 0.49, 0.5, 1.0], 0.5),
                                  ["false", "false", "true", "true"])
    np.testing.assert_array_equal(experiment.apply_threshold([0.0, 1.0], 0.0), ["true", "true"])
    np.testing.assert_array_equal(experiment.apply_threshold([0.0, 1.0], 1.0), ["false", "true"])


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        experiment.apply_threshold([0.5], threshold)


@pytest.mark.parametrize("values", [[np.nan], [np.inf], [-0.1], [1.1], [[0.5]]])
def test_invalid_probabilities_rejected(values):
    with pytest.raises(ValueError, match="Probabilities"):
        experiment.apply_threshold(values, 0.5)


def test_counts_metrics_workload_monotonicity_and_preservation():
    y = pd.Series(["false", "false", "true", "true"], index=[10, 20, 30, 40])
    probabilities = pd.Series([0.1, 0.6, 0.4, 0.8], index=y.index)
    original_y, original_p = y.copy(), probabilities.copy()
    results = experiment.evaluate_thresholds(y, probabilities, thresholds=[0.2, 0.5, 0.7, 0.9])
    row = results.loc[0.5]
    assert row[["tp", "fp", "tn", "fn"]].tolist() == [1, 1, 1, 1]
    assert row[["precision", "recall", "f1", "accuracy"]].tolist() == [0.5] * 4
    assert row["flagged_modules"] == 2 and row["flagged_percent"] == 50
    assert results.loc[0.2, "recall"] == 1
    assert results.loc[0.2, "precision"] == pytest.approx(2 / 3)
    assert results.loc[0.2, "f1"] == pytest.approx(0.8)
    assert results.loc[0.9, ["precision", "recall", "f1", "flagged_modules"]].eq(0).all()
    assert results["tp"].is_monotonic_decreasing
    assert results["fp"].is_monotonic_decreasing
    assert results["fn"].is_monotonic_increasing
    assert results["flagged_modules"].is_monotonic_decreasing
    assert results[["tp", "fp", "tn", "fn"]].sum(axis=1).eq(len(y)).all()
    pd.testing.assert_series_equal(y, original_y)
    pd.testing.assert_series_equal(probabilities, original_p)


def test_label_probability_alignment_required():
    with pytest.raises(ValueError, match="matching"):
        experiment.evaluate_thresholds(pd.Series(["true", "false"]), pd.Series([0.8, 0.2], index=[1, 0]))


def test_only_five_fits_and_probability_calls_for_all_thresholds(training, monkeypatch):
    X, y = training
    seen = []
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)

    def factory(*, balanced):
        assert balanced is True
        pipeline = create_experiment_pipeline(balanced=balanced)
        assert pipeline.named_steps["classifier"].get_params()["class_weight"] == "balanced"
        record = {"pipeline": pipeline, "fit_calls": 0, "proba_calls": 0}
        seen.append(record)
        fit, predict_proba = pipeline.fit, pipeline.predict_proba

        def checked_fit(features, labels):
            record["fit_calls"] += 1
            record["fit"] = features.copy()
            fit(features, labels)
            imputer = pipeline.named_steps["imputer"]
            np.testing.assert_allclose(imputer.statistics_, features.median().to_numpy())
            np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_,
                                       imputer.transform(features).mean(axis=0))
            return pipeline

        def checked_proba(features):
            record["proba_calls"] += 1
            record["validation"] = features.copy()
            values = predict_proba(features)
            record["probabilities"] = values[:, list(pipeline.classes_).index("true")]
            return values

        monkeypatch.setattr(pipeline, "fit", checked_fit)
        monkeypatch.setattr(pipeline, "predict_proba", checked_proba)
        return pipeline

    monkeypatch.setattr(experiment, "create_experiment_pipeline", factory)
    probabilities, audits = experiment.balanced_oof_probabilities(X, y)
    results = experiment.evaluate_thresholds(y, probabilities)
    assert len(results) == 7
    assert len({id(record["pipeline"]) for record in seen}) == 5
    for record, (fit_indices, validation_indices) in zip(seen, training_folds(X, y)):
        assert record["fit_calls"] == record["proba_calls"] == 1
        pd.testing.assert_frame_equal(record["fit"], X.iloc[fit_indices])
        pd.testing.assert_frame_equal(record["validation"], X.iloc[validation_indices])
        np.testing.assert_allclose(probabilities.iloc[validation_indices], record["probabilities"])
    assert probabilities.index.equals(y.index) and probabilities.notna().all()
    assert audits["shared_feature_vectors"].eq(0).all()
    assert audits["validation_rows"].sum() == len(X)
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_correct_positive_probability_column_even_with_reversed_classes(training, monkeypatch):
    X, y = training

    class ReversedClasses:
        classes_ = np.array(["true", "false"])

        def fit(self, features, labels):
            return self

        def predict_proba(self, features):
            return np.tile([0.8, 0.2], (len(features), 1))

    monkeypatch.setattr(experiment, "create_experiment_pipeline", lambda **_: ReversedClasses())
    probabilities, _ = experiment.balanced_oof_probabilities(X, y)
    np.testing.assert_allclose(probabilities, 0.8)


def test_contamination_rejected_before_fitting(training, monkeypatch):
    X, y = training
    # Keep proper coverage but make the first fit group equal a validation group.
    folds = list(training_folds(X, y))
    fit, validation = folds[0]
    folds[0] = (np.append(fit, validation[0]), validation)
    monkeypatch.setattr(experiment, "training_folds", lambda *_: iter(folds))
    monkeypatch.setattr(experiment, "create_experiment_pipeline", lambda **_: pytest.fail("Must reject before fit"))
    with pytest.raises(RuntimeError, match="contamination"):
        experiment.balanced_oof_probabilities(X, y)


def test_oof_coverage_must_be_exactly_once(training, monkeypatch):
    X, y = training
    folds = list(training_folds(X, y))
    folds[-1] = folds[0]
    monkeypatch.setattr(experiment, "training_folds", lambda *_: iter(folds))
    monkeypatch.setattr(experiment, "create_experiment_pipeline", lambda **_: pytest.fail("Must reject before fit"))
    with pytest.raises(RuntimeError, match="exactly once"):
        experiment.balanced_oof_probabilities(X, y)


def test_command_discards_final_test_outputs(training, monkeypatch, capsys):
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
        return "Training-only threshold measurement"

    monkeypatch.setattr(experiment, "threshold_report", report)
    experiment.main()
    assert capsys.readouterr().out.strip() == "Training-only threshold measurement"
