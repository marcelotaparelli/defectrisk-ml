import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from defectrisk import gradient_boosting_baseline as experiment
from defectrisk import random_forest_baseline as shared
from defectrisk.cross_validation import training_folds


def training():
    X = pd.DataFrame({"a": np.repeat(np.arange(30, dtype=float), 3),
                      "b": np.repeat(np.arange(30) % 7, 3)}, index=range(1000, 1090))
    X.iloc[:3, 0] = np.nan
    y = pd.Series(["false", "false", "true"] * 30, index=X.index, name="defects")
    return X, y


def test_configuration_and_no_internal_holdout():
    pipeline = experiment.create_boosting_comparison_pipeline("hist_gradient_boosting")
    assert list(pipeline.named_steps) == ["imputer", "classifier"]
    assert pipeline.named_steps["imputer"].strategy == "median"
    assert pipeline.named_steps["classifier"].get_params() == HistGradientBoostingClassifier(
        random_state=42, class_weight="balanced", early_stopping=False,
    ).get_params()
    lr = experiment.create_boosting_comparison_pipeline("balanced_logistic_regression")
    assert lr.named_steps["classifier"].C == 1
    assert lr.named_steps["classifier"].max_iter == 5000
    rf = experiment.create_boosting_comparison_pipeline("weighted_random_forest")
    assert rf.named_steps["classifier"].class_weight == "balanced"
    with pytest.raises(ValueError, match="Unknown"):
        experiment.create_boosting_comparison_pipeline("other")


def test_paired_oof_fits_once_per_model_fold_and_budget_counts(monkeypatch):
    X, y = training()
    original = experiment.create_boosting_comparison_pipeline
    records = []

    def factory(model):
        pipeline = original(model)
        fit, proba = pipeline.fit, pipeline.predict_proba
        record = {"model": model, "pipeline": pipeline, "fits": 0, "predictions": 0}
        records.append(record)

        def checked_fit(features, labels):
            record["fits"] += 1
            record["fit_index"] = features.index
            assert "defects" not in features
            fit(features, labels)
            np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, features.median())
            if model == "hist_gradient_boosting":
                assert pipeline.named_steps["classifier"].do_early_stopping_ is False
            return pipeline

        def checked_proba(features):
            record["predictions"] += 1
            record["validation_index"] = features.index
            return proba(features)

        monkeypatch.setattr(pipeline, "fit", checked_fit)
        monkeypatch.setattr(pipeline, "predict_proba", checked_proba)
        return pipeline

    monkeypatch.setattr(experiment, "create_boosting_comparison_pipeline", factory)
    folds, summary, pooled, audits, oof, budgets, curves, ap = experiment.compare_gradient_boosting(X, y)
    assert len(records) == len({id(r["pipeline"]) for r in records}) == 15
    for i, (fit, validation) in enumerate(training_folds(X, y)):
        for record, model in zip(records[i * 3:i * 3 + 3], experiment.MODELS):
            assert record["model"] == model
            assert record["fits"] == record["predictions"] == 1
            assert record["fit_index"].equals(X.iloc[fit].index)
            assert record["validation_index"].equals(X.iloc[validation].index)
    assert audits["shared_feature_vectors"].eq(0).all()
    assert oof.index.equals(y.index)
    assert np.isfinite(oof.to_numpy()).all()
    assert oof.shape == (len(y), 3)
    for model in experiment.MODELS:
        np.testing.assert_allclose(pooled.loc[model, ["tp", "fp", "tn", "fn"]],
                                   folds.loc[model, ["tp", "fp", "tn", "fn"]].sum())
        for metric in folds.columns:
            assert summary.loc[(model, metric), "std"] == pytest.approx(folds.loc[model, metric].std(ddof=0))
        counts = budgets.loc[model]
        assert (counts.tp + counts.fn).eq(30).all()
        assert (counts.tp + counts.fp).equals(counts.flagged_modules)
        assert counts.flagged_modules.is_monotonic_increasing
        assert counts.tp.is_monotonic_increasing
        assert 0 <= ap[model] <= 1
    assert set(curves.model) == set(experiment.MODELS)


def test_budget_threshold_ties_and_label_independence():
    probabilities = pd.Series([0.1, 0.2, 0.2, 0.9])
    assert experiment.review_threshold(probabilities, .25) == .9
    # Target count 2: either 1 or 3 reviews; prefer fewer, never split equal scores.
    assert experiment.review_threshold(probabilities, .5) == .9
    assert experiment.review_threshold(probabilities, .75) == .2
    y = pd.Series(["false", "true", "true", "false"])
    first = experiment.evaluate_review_budgets(y, probabilities.to_frame("model"))
    second = experiment.evaluate_review_budgets(y.iloc[::-1].reset_index(drop=True), probabilities.to_frame("model"))
    pd.testing.assert_series_equal(first.threshold, second.threshold)
    scores = first.loc[("model", .3)]
    assert scores["tp"] == 0 and scores["fp"] == 1 and scores["fn"] == 2
    assert scores["flagged_percent"] == 25


@pytest.mark.parametrize("scores,budget", [([], .2), ([np.nan], .2), ([1.2], .2),
                                           ([[.2]], .2), ([.2], 0), ([.2], 1), ([.2], np.nan)])
def test_invalid_budget_inputs(scores, budget):
    with pytest.raises(ValueError):
        experiment.review_threshold(scores, budget)


def test_boosting_contamination_rejected_before_fit(monkeypatch):
    X, y = training()
    folds = list(training_folds(X, y))
    fit, validation = folds[0]
    folds[0] = (np.append(fit, validation[0]), validation)
    monkeypatch.setattr(shared, "training_folds", lambda *_: iter(folds))
    monkeypatch.setattr(experiment, "create_boosting_comparison_pipeline", lambda _: pytest.fail("No fit"))
    with pytest.raises(RuntimeError, match="contamination"):
        experiment.compare_gradient_boosting(X, y)


def test_command_discards_final_test_partitions(monkeypatch, tmp_path):
    X, y = training()

    class Locked:
        def __getattribute__(self, name):
            pytest.fail("Final test must remain untouched")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment, "load_jm1", lambda: None)
    monkeypatch.setattr(experiment, "split_dataset", lambda _: (X, Locked(), y, Locked()))
    frames = [pd.DataFrame({"value": [1]}) for _ in range(8)]

    def checked(features, labels):
        assert features is X and labels is y
        return frames

    monkeypatch.setattr(experiment, "compare_gradient_boosting", checked)
    experiment.main()
    assert (tmp_path / "docs/gradient-boosting-results/oof-probabilities.csv").exists()
