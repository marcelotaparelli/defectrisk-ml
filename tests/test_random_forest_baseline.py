import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from defectrisk import random_forest_baseline as experiment
from defectrisk.cross_validation import training_folds
from defectrisk.data import TARGET
from defectrisk.split import split_dataset


@pytest.fixture
def training():
    X = pd.DataFrame({"a": np.repeat(np.arange(30, dtype=float), 3),
                      "b": np.repeat(np.arange(30) % 7, 3)}, index=range(1000, 1090))
    X.loc[[1000, 1001, 1002], "a"] = np.nan
    y = pd.Series(["false", "false", "true"] * 30, index=X.index, name=TARGET)
    return X, y


def test_exact_classifier_configs_and_common_preprocessing():
    lr = experiment.create_comparison_pipeline(experiment.MODELS[0])
    rf = experiment.create_comparison_pipeline(experiment.MODELS[1])
    assert rf.named_steps["classifier"].get_params() == RandomForestClassifier(random_state=42).get_params()
    assert rf.named_steps["classifier"].class_weight is None
    assert lr.named_steps["classifier"].class_weight == "balanced"
    assert lr.named_steps["classifier"].random_state == 42
    assert lr.named_steps["classifier"].max_iter == 5000
    for step in ("imputer", "scaler"):
        assert lr.named_steps[step].get_params() == rf.named_steps[step].get_params()


def test_paired_folds_fresh_models_preprocessing_and_aggregate_counts(training, monkeypatch):
    X, y = training
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    factory = experiment.create_comparison_pipeline
    seen = []

    def checked_factory(model):
        pipeline = factory(model)
        assert not hasattr(pipeline.named_steps["classifier"], "classes_")
        fit, proba = pipeline.fit, pipeline.predict_proba
        record = {"pipeline": pipeline, "model": model, "fit_calls": 0, "proba_calls": 0}
        seen.append(record)

        def checked_fit(features, labels):
            record["fit_calls"] += 1
            record["fit"] = features.copy()
            record["labels"] = labels.copy()
            fit(features, labels)
            imputer = pipeline.named_steps["imputer"]
            np.testing.assert_allclose(imputer.statistics_, features.median().to_numpy())
            np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_,
                                       imputer.transform(features).mean(axis=0))
            return pipeline

        def checked_proba(features):
            record["proba_calls"] += 1
            record["validation"] = features.copy()
            return proba(features)

        monkeypatch.setattr(pipeline, "fit", checked_fit)
        monkeypatch.setattr(pipeline, "predict_proba", checked_proba)
        return pipeline

    monkeypatch.setattr(experiment, "create_comparison_pipeline", checked_factory)
    folds, summary, pooled, audits = experiment.compare_random_forest(X, y)
    assert len({id(record["pipeline"]) for record in seen}) == 10
    for (fit, validation), lr, rf in zip(training_folds(X, y), seen[::2], seen[1::2]):
        assert (lr["model"], rf["model"]) == experiment.MODELS
        for record in (lr, rf):
            assert record["fit_calls"] == record["proba_calls"] == 1
            pd.testing.assert_frame_equal(record["fit"], X.iloc[fit])
            pd.testing.assert_series_equal(record["labels"], y.iloc[fit])
            pd.testing.assert_frame_equal(record["validation"], X.iloc[validation])
    assert audits["shared_feature_vectors"].eq(0).all()
    assert audits["validation_rows"].sum() == len(X)
    for model in experiment.MODELS:
        counts = pooled.loc[model, ["tp", "fp", "tn", "fn"]]
        assert counts.sum() == len(X)
        np.testing.assert_allclose(counts, folds.loc[model, ["tp", "fp", "tn", "fn"]].sum())
        assert pooled.loc[model, "flagged_modules"] == counts["tp"] + counts["fp"]
        for metric in folds.columns:
            assert summary.loc[(model, metric), "mean"] == pytest.approx(folds.loc[model, metric].mean())
            assert summary.loc[(model, metric), "std"] == pytest.approx(folds.loc[model, metric].std(ddof=0))
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_fixed_threshold_and_positive_column_with_reversed_classes(training, monkeypatch):
    X, y = training

    class FixedProbabilities:
        classes_ = np.array(["true", "false"])

        def __init__(self, model):
            self.positive = 0.5 if model == experiment.MODELS[0] else 0.49

        def fit(self, features, labels):
            return self

        def predict_proba(self, features):
            return np.tile([self.positive, 1 - self.positive], (len(features), 1))

    monkeypatch.setattr(experiment, "create_comparison_pipeline", FixedProbabilities)
    _, _, pooled, _ = experiment.compare_random_forest(X, y)
    assert pooled.loc[experiment.MODELS[0], ["tp", "fp", "tn", "fn"]].tolist() == [30, 60, 0, 0]
    assert pooled.loc[experiment.MODELS[1], ["tp", "fp", "tn", "fn"]].tolist() == [0, 0, 60, 30]
    assert pooled.loc[experiment.MODELS[0], "recall"] == 1
    assert pooled.loc[experiment.MODELS[1], "recall"] == 0


def test_contamination_rejected_before_either_model(training, monkeypatch):
    X, y = training
    folds = list(training_folds(X, y))
    fit, validation = folds[0]
    folds[0] = (np.append(fit, validation[0]), validation)
    monkeypatch.setattr(experiment, "training_folds", lambda *_: iter(folds))
    monkeypatch.setattr(experiment, "create_comparison_pipeline", lambda _: pytest.fail("Must reject before fit"))
    with pytest.raises(RuntimeError, match="contamination"):
        experiment.compare_random_forest(X, y)


def test_command_never_inspects_final_test_outputs(training, monkeypatch, capsys):
    X, y = training
    frame = X.assign(**{TARGET: y})
    expected_X, _, expected_y, _ = split_dataset(frame)
    monkeypatch.setattr(experiment, "load_jm1", lambda: frame)

    class ForbiddenTestPartition:
        def __getattribute__(self, name):
            pytest.fail("Final test partition must not be inspected")

    monkeypatch.setattr(experiment, "split_dataset", lambda _: (
        expected_X, ForbiddenTestPartition(), expected_y, ForbiddenTestPartition()
    ))

    def report(features, labels):
        pd.testing.assert_frame_equal(features, expected_X)
        pd.testing.assert_series_equal(labels, expected_y)
        return "Training-only forest comparison"

    monkeypatch.setattr(experiment, "random_forest_report", report)
    experiment.main()
    assert capsys.readouterr().out.strip() == "Training-only forest comparison"
