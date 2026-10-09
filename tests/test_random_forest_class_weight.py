import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from defectrisk import random_forest_baseline as evaluator
from defectrisk import random_forest_class_weight as experiment
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


def test_only_forest_class_weight_changes_and_lr_is_unchanged():
    lr, default, weighted = [experiment.create_weight_comparison_pipeline(model) for model in experiment.MODELS]
    assert default.named_steps["classifier"].get_params() == RandomForestClassifier(random_state=42).get_params()
    assert weighted.named_steps["classifier"].get_params() == RandomForestClassifier(
        random_state=42, class_weight="balanced"
    ).get_params()
    left, right = default.named_steps["classifier"].get_params(), weighted.named_steps["classifier"].get_params()
    assert left.pop("class_weight") is None
    assert right.pop("class_weight") == "balanced"
    assert left == right
    original_lr = evaluator.create_comparison_pipeline("balanced_logistic_regression")
    for step in lr.named_steps:
        assert lr.named_steps[step].get_params() == original_lr.named_steps[step].get_params()
    for step in ("imputer", "scaler"):
        assert lr.named_steps[step].get_params() == default.named_steps[step].get_params()
        assert default.named_steps[step].get_params() == weighted.named_steps[step].get_params()


def test_all_three_models_use_exact_existing_folds_and_fresh_pipelines(training, monkeypatch):
    X, y = training
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    factory = experiment.create_weight_comparison_pipeline
    seen = []

    def checked_factory(model):
        pipeline = factory(model)
        assert not hasattr(pipeline.named_steps["classifier"], "classes_")
        fit, proba = pipeline.fit, pipeline.predict_proba
        record = {"model": model, "pipeline": pipeline, "fit_calls": 0, "proba_calls": 0}
        seen.append(record)

        def checked_fit(features, labels):
            record["fit_calls"] += 1
            record["fit"], record["labels"] = features.copy(), labels.copy()
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

    monkeypatch.setattr(experiment, "create_weight_comparison_pipeline", checked_factory)
    folds, summary, pooled, audits = experiment.compare_forest_class_weights(X, y)
    assert len({id(record["pipeline"]) for record in seen}) == 15
    for fold, (fit, validation) in enumerate(training_folds(X, y)):
        records = seen[fold * 3:fold * 3 + 3]
        assert tuple(record["model"] for record in records) == experiment.MODELS
        for record in records:
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
        assert pooled.loc[model, "flagged_percent"] == pytest.approx((counts["tp"] + counts["fp"]) / len(X) * 100)
        for metric in folds.columns:
            assert summary.loc[(model, metric), "mean"] == pytest.approx(folds.loc[model, metric].mean())
            assert summary.loc[(model, metric), "std"] == pytest.approx(folds.loc[model, metric].std(ddof=0))
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_fixed_threshold_and_positive_class_for_all_three_models(training, monkeypatch):
    X, y = training

    class FixedProbabilities:
        classes_ = np.array(["true", "false"])

        def __init__(self, model):
            self.probability = 0.49 if model == "random_forest" else 0.50

        def fit(self, features, labels):
            return self

        def predict_proba(self, features):
            return np.tile([self.probability, 1 - self.probability], (len(features), 1))

    monkeypatch.setattr(experiment, "create_weight_comparison_pipeline", FixedProbabilities)
    _, _, pooled, _ = experiment.compare_forest_class_weights(X, y)
    for model in experiment.MODELS:
        expected = [0, 0, 60, 30] if model == "random_forest" else [30, 60, 0, 0]
        assert pooled.loc[model, ["tp", "fp", "tn", "fn"]].tolist() == expected


def test_contamination_rejected_before_any_model(training, monkeypatch):
    X, y = training
    folds = list(training_folds(X, y))
    fit, validation = folds[0]
    folds[0] = (np.append(fit, validation[0]), validation)
    monkeypatch.setattr(evaluator, "training_folds", lambda *_: iter(folds))
    monkeypatch.setattr(experiment, "create_weight_comparison_pipeline", lambda _: pytest.fail("Must reject before fit"))
    with pytest.raises(RuntimeError, match="contamination"):
        experiment.compare_forest_class_weights(X, y)


def test_command_does_not_inspect_final_test_outputs(training, monkeypatch, capsys):
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
        return "Training-only forest weight comparison"

    monkeypatch.setattr(experiment, "forest_class_weight_report", report)
    experiment.main()
    assert capsys.readouterr().out.strip() == "Training-only forest weight comparison"
