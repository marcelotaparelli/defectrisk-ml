import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from defectrisk.audit import feature_overlap
from defectrisk.baseline import baseline_report, create_baseline_pipeline, main
from defectrisk.data import TARGET
from defectrisk.split import split_dataset


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "loc": [float(i % 20) for i in range(99)] + [None],
            "complexity": [float(i % 7) for i in range(100)],
            TARGET: ["false"] * 80 + ["true"] * 20,
        },
        index=range(1000, 1100),
    )


def test_pipeline_can_be_created():
    pipeline = create_baseline_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert not hasattr(pipeline.named_steps["classifier"], "classes_")


def test_pipeline_contains_preprocessing_and_classifier():
    pipeline = create_baseline_pipeline()
    assert list(pipeline.named_steps) == ["imputer", "scaler", "classifier"]
    assert isinstance(pipeline.named_steps["imputer"], SimpleImputer)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_target_is_not_in_features(frame):
    X_train, _, y_train, _ = split_dataset(frame)
    assert TARGET not in X_train.columns
    assert "Accuracy:" in baseline_report(X_train, y_train)


def test_fit_succeeds_with_missing_values_and_predict_preserves_length(frame):
    pipeline = create_baseline_pipeline()
    X = frame.drop(columns=TARGET)
    pipeline.fit(X, frame[TARGET])
    assert set(pipeline.named_steps["classifier"].classes_) == {"false", "true"}
    # Include the missing-value row when predicting.
    assert len(pipeline.predict(X.iloc[-7:])) == 7


def test_target_in_features_is_rejected(frame):
    with pytest.raises(ValueError, match="must not be included"):
        baseline_report(frame, frame[TARGET])


def test_misaligned_labels_are_rejected(frame):
    with pytest.raises(ValueError, match="matching lengths and indices"):
        baseline_report(frame.drop(columns=TARGET), frame[TARGET].iloc[::-1])


def test_preprocessing_learns_only_from_fit_subset(frame, monkeypatch):
    X_train, X_test, y_train, _ = split_dataset(frame)
    pipeline = create_baseline_pipeline()
    original_fit = pipeline.fit
    original_predict = pipeline.predict
    seen = {}

    def record_fit(X, y):
        seen["fit"] = X.copy()
        assert X.index.equals(y.index)
        return original_fit(X, y)

    def record_predict(X):
        seen["validation"] = X.copy()
        return original_predict(X)

    monkeypatch.setattr(pipeline, "fit", record_fit)
    monkeypatch.setattr(pipeline, "predict", record_predict)
    monkeypatch.setattr("defectrisk.baseline.create_baseline_pipeline", lambda: pipeline)
    baseline_report(X_train, y_train)

    fit = seen["fit"]
    validation = seen["validation"]
    assert len(fit) == 64 and len(validation) == 16
    assert fit.index.intersection(validation.index).empty
    assert set(fit.index) | set(validation.index) == set(X_train.index)
    assert fit.index.intersection(X_test.index).empty
    assert validation.index.intersection(X_test.index).empty
    assert feature_overlap(fit, validation)["shared_feature_vectors"] == 0
    assert feature_overlap(X_train, X_test)["shared_feature_vectors"] == 0
    assert pipeline.named_steps["imputer"].statistics_ == pytest.approx(fit.median().to_numpy())
    imputed_fit = pipeline.named_steps["imputer"].transform(fit)
    assert pipeline.named_steps["scaler"].mean_ == pytest.approx(imputed_fit.mean(axis=0))


def test_report_uses_defective_as_positive_class(monkeypatch):
    # Asymmetric class results expose an accidentally reversed positive class.
    # TN=3, FP=1, FN=1, TP=1.
    X = pd.DataFrame({"loc": range(12)})
    y = pd.Series(["false"] * 6 + ["true"] * 6, name=TARGET)
    actual = pd.Series(["false", "false", "false", "false", "true", "true"], name=TARGET)

    def fixed_validation(*args, **kwargs):
        return X.iloc[:6], X.iloc[6:], y.iloc[:6], actual

    class FixedPredictions:
        def fit(self, X_fit, y_fit):
            return self

        def predict(self, X_validation):
            return ["false", "false", "false", "true", "false", "true"]

    monkeypatch.setattr("defectrisk.baseline.group_holdout", fixed_validation)
    monkeypatch.setattr("defectrisk.baseline.create_baseline_pipeline", FixedPredictions)
    report = baseline_report(X, y)
    assert "Accuracy: 0.6667" in report
    for metric in ["Precision (defective)", "Recall (defective)", "F1 (defective)"]:
        assert f"{metric}: 0.5000" in report
    assert "Positive class: 'true' (defective)" in report
    assert "[[TN, FP], [FN, TP]]" in report
    rows = [line.split() for line in report.splitlines()]
    assert ["actual", "clean", "3", "1"] in rows
    assert ["actual", "defective", "1", "1"] in rows


def test_command_passes_only_m3_training_partition(frame, monkeypatch, capsys):
    expected_X, _, expected_y, _ = split_dataset(frame)
    monkeypatch.setattr("defectrisk.baseline.load_jm1", lambda: frame)

    def check_training_partition(X_train, y_train):
        pd.testing.assert_frame_equal(X_train, expected_X)
        pd.testing.assert_series_equal(y_train, expected_y)
        return "Training-side report"

    monkeypatch.setattr("defectrisk.baseline.baseline_report", check_training_partition)
    main()
    assert capsys.readouterr().out.strip() == "Training-side report"
