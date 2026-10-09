import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from defectrisk import logistic_coefficients as analysis
from defectrisk.data import TARGET
from defectrisk.split import split_dataset


@pytest.fixture
def training():
    rng = np.random.RandomState(42)
    X = pd.DataFrame({"large_feature": rng.normal(size=100) * 1000,
                      "small_feature": rng.normal(size=100)}, index=range(1000, 1100))
    y = pd.Series(np.where(X["large_feature"] > 0, "true", "false"), index=X.index, name=TARGET)
    X.loc[1000, "small_feature"] = np.nan
    return X, y


def test_full_training_fit_fixed_settings_and_no_prediction(training, monkeypatch):
    X, y = training
    original_X, original_y = X.copy(deep=True), y.copy(deep=True)
    original_fit = LogisticRegression.fit
    seen = []

    def checked_fit(classifier, features, labels, *args, **kwargs):
        assert classifier.get_params() == LogisticRegression(
            random_state=42, max_iter=5000, class_weight="balanced", C=1.0
        ).get_params()
        pd.testing.assert_series_equal(labels, y)
        assert len(features) == len(X)
        imputed = X.fillna(X.median())
        expected = (imputed - imputed.mean()) / imputed.std(ddof=0)
        np.testing.assert_allclose(features, expected.to_numpy(), atol=1e-12)
        seen.append(len(features))
        return original_fit(classifier, features, labels, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Interpretability fit must not predict or score")

    monkeypatch.setattr(LogisticRegression, "fit", checked_fit)
    for method in ("predict", "predict_proba", "score"):
        monkeypatch.setattr(LogisticRegression, method, forbidden)
    table, intercept, pipeline = analysis.fit_training_coefficients(X, y)
    assert seen == [100]
    assert set(table["feature"]) == set(X.columns)
    assert table["absolute_coefficient"].is_monotonic_decreasing
    coefficients = table.set_index("feature").loc[X.columns, "coefficient"].to_numpy()
    np.testing.assert_array_equal(coefficients, pipeline.named_steps["classifier"].coef_[0])
    assert intercept == pipeline.named_steps["classifier"].intercept_[0]
    for row in table.itertuples():
        assert row.absolute_coefficient == abs(row.coefficient)
        assert row.direction == ("toward defective" if row.coefficient > 0 else "toward clean")
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_series_equal(y, original_y)


def test_mapping_sorting_and_zero_direction_with_known_coefficients(training):
    X, y = training
    _, _, pipeline = analysis.fit_training_coefficients(X, y)
    classifier = pipeline.named_steps["classifier"]
    classifier.coef_ = np.array([[0.0, -3.0]])
    classifier.intercept_ = np.array([0.75])
    table, intercept = analysis.coefficient_table(pipeline)
    assert table["feature"].tolist() == ["small_feature", "large_feature"]
    assert table["coefficient"].tolist() == [-3.0, 0.0]
    assert table["absolute_coefficient"].tolist() == [3.0, 0.0]
    assert table["direction"].tolist() == ["toward clean", "neutral"]
    assert intercept == 0.75


def test_reported_coefficients_reconstruct_standardized_linear_score(training):
    X, y = training
    table, intercept, pipeline = analysis.fit_training_coefficients(X, y)
    coefficients = table.set_index("feature").loc[X.columns, "coefficient"].to_numpy()
    transformed = pipeline[:-1].transform(X)
    np.testing.assert_allclose(transformed @ coefficients + intercept, pipeline.decision_function(X))


@pytest.mark.parametrize("problem", ["target", "alignment", "missing_labels"])
def test_invalid_training_inputs_rejected(training, problem):
    X, y = training
    if problem == "target":
        X[TARGET] = y
    elif problem == "alignment":
        y = y.iloc[::-1]
    else:
        y.iloc[0] = None
    with pytest.raises(ValueError):
        analysis.fit_training_coefficients(X, y)


def test_command_discards_final_test_outputs(training, monkeypatch, capsys):
    X, y = training
    frame = X.assign(**{TARGET: y})
    expected_X, _, expected_y, _ = split_dataset(frame)
    monkeypatch.setattr(analysis, "load_jm1", lambda: frame)

    class ForbiddenTestPartition:
        def __getattribute__(self, name):
            pytest.fail("Final test outputs must not be inspected")

    monkeypatch.setattr(analysis, "split_dataset", lambda _: (
        expected_X, ForbiddenTestPartition(), expected_y, ForbiddenTestPartition()
    ))

    def report(features, labels):
        pd.testing.assert_frame_equal(features, expected_X)
        pd.testing.assert_series_equal(labels, expected_y)
        return "Training-only coefficient analysis"

    monkeypatch.setattr(analysis, "logistic_coefficient_report", report)
    analysis.main()
    assert capsys.readouterr().out.strip() == "Training-only coefficient analysis"
