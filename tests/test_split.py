import pandas as pd
import pytest

from defectrisk.data import TARGET, load_jm1
from defectrisk.audit import feature_overlap
from defectrisk.split import feature_groups, group_holdout, main, split_dataset, split_report


@pytest.fixture(
    scope="module",
    params=["synthetic", pytest.param("jm1", marks=pytest.mark.integration)],
)
def frame(request):
    if request.param == "jm1":
        return load_jm1()
    # Uneven size exercises rounding; nonconsecutive indices exercise alignment.
    return pd.DataFrame(
        {
            "loc": [float(i % 10) for i in range(100)] + [None],
            "v(g)": [float(i % 3) for i in range(101)],
            TARGET: ["false"] * 81 + ["true"] * 20,
        },
        index=pd.Index(range(1000, 1303, 3), name="module_row"),
    )


def test_train_and_test_are_nonempty(frame):
    X_train, X_test, y_train, y_test = split_dataset(frame)
    assert all(len(part) > 0 for part in (X_train, X_test, y_train, y_test))


def test_features_and_labels_remain_aligned(frame):
    X_train, X_test, y_train, y_test = split_dataset(frame)
    for X, y in [(X_train, y_train), (X_test, y_test)]:
        assert len(X) == len(y)
        assert X.index.equals(y.index)
        pd.testing.assert_series_equal(y, frame.loc[X.index, TARGET])


def test_target_is_excluded_and_all_features_are_preserved(frame):
    X_train, X_test, _, _ = split_dataset(frame)
    expected = frame.drop(columns=TARGET)
    for X in (X_train, X_test):
        assert TARGET not in X.columns
        pd.testing.assert_frame_equal(X, expected.loc[X.index])


def test_indices_are_disjoint_and_cover_every_original_row(frame):
    X_train, X_test, y_train, y_test = split_dataset(frame)
    assert X_train.index.intersection(X_test.index).empty
    assert y_train.index.intersection(y_test.index).empty
    combined = X_train.index.append(X_test.index)
    assert combined.is_unique
    assert set(combined) == set(frame.index)


def test_sizes_are_approximately_eighty_twenty(frame):
    X_train, X_test, _, _ = split_dataset(frame)
    assert len(X_train) + len(X_test) == len(frame)
    # Whole groups can prevent exact row counts; allow two percentage points.
    assert len(X_test) / len(frame) == pytest.approx(0.20, abs=0.02)
    assert len(X_train) / len(frame) == pytest.approx(0.80, abs=0.02)


def test_stratification_preserves_class_proportions(frame):
    _, _, y_train, y_test = split_dataset(frame)
    original = frame[TARGET].value_counts(normalize=True)
    for y in (y_train, y_test):
        proportions = y.value_counts(normalize=True)
        assert set(y) == set(frame[TARGET])
        for label in original.index:
            assert proportions[label] == pytest.approx(original[label], abs=0.03)


def test_same_seed_reproduces_the_same_split(frame):
    first = split_dataset(frame, random_state=42)
    second = split_dataset(frame, random_state=42)
    for left, right in zip(first[:2], second[:2]):
        pd.testing.assert_frame_equal(left, right)
    for left, right in zip(first[2:], second[2:]):
        pd.testing.assert_series_equal(left, right)


def test_split_does_not_modify_input(frame):
    original = frame.copy(deep=True)
    split_dataset(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_report_prints_sizes_and_distributions():
    raw = pd.DataFrame({"loc": range(100), TARGET: ["false"] * 80 + ["true"] * 20})
    report = split_report(raw)
    for line in ["Full dataset size: 100", "Train size: 80", "Test size: 20"]:
        assert line in report
    for name in ["Full", "Train", "Test"]:
        section = report.split(f"{name} class distribution:\n")[1].split("\n\n")[0]
        assert "count" in section and "percent" in section
        assert "false" in section and "80.00" in section
        assert "true" in section and "20.00" in section
    rows = [line.split() for line in report.splitlines()]
    assert ["false", "80", "80.00"] in rows
    assert ["false", "64", "80.00"] in rows
    assert ["false", "16", "80.00"] in rows
    assert ["true", "20", "20.00"] in rows
    assert ["true", "16", "20.00"] in rows
    assert ["true", "4", "20.00"] in rows


def test_report_command_uses_existing_loader(monkeypatch, capsys):
    raw = pd.DataFrame({"loc": range(100), TARGET: ["false"] * 80 + ["true"] * 20})
    monkeypatch.setattr("defectrisk.split.load_jm1", lambda: raw)
    main()
    assert "Full dataset size: 100" in capsys.readouterr().out


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="target column"):
        split_dataset(pd.DataFrame({"loc": range(10)}))


def test_duplicate_index_labels_are_rejected():
    raw = pd.DataFrame({"loc": [1, 2], TARGET: ["false", "true"]}, index=[0, 0])
    with pytest.raises(ValueError, match="unique row index"):
        split_dataset(raw)


def test_zero_feature_overlap_at_both_boundaries(frame):
    X_train, X_test, y_train, _ = split_dataset(frame)
    X_fit, X_validation, y_fit, y_validation = group_holdout(X_train, y_train)
    for left, right in [(X_train, X_test), (X_fit, X_validation)]:
        assert feature_overlap(left, right) == {
            "shared_feature_vectors": 0, "left_rows_with_match": 0,
            "right_rows_with_match": 0, "matching_row_pairs": 0,
        }
        assert TARGET not in left and TARGET not in right
    assert X_fit.index.equals(y_fit.index)
    assert X_validation.index.equals(y_validation.index)
    assert set(X_fit.index) | set(X_validation.index) == set(X_train.index)
    assert len(X_validation) / len(X_train) == pytest.approx(0.20, abs=0.03)
    original = y_train.value_counts(normalize=True)
    for labels in (y_fit, y_validation):
        proportions = labels.value_counts(normalize=True)
        for label in original.index:
            assert proportions[label] == pytest.approx(original[label], abs=0.05)
    repeated = group_holdout(X_train, y_train)
    for first, second in zip((X_fit, X_validation, y_fit, y_validation), repeated):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)


def test_conflicting_label_groups_stay_together_at_both_boundaries(frame):
    X_train, X_test, y_train, _ = split_dataset(frame)
    X_fit, X_validation, _, _ = group_holdout(X_train, y_train)
    for X, y, left, right in [
        (frame.drop(columns=TARGET), frame[TARGET], X_train, X_test),
        (X_train, y_train, X_fit, X_validation),
    ]:
        groups = feature_groups(X)
        conflicting = y.groupby(groups).nunique()
        conflicting = conflicting.index[conflicting > 1]
        assert len(conflicting) > 0
        for group in conflicting:
            indices = set(X.index[groups == group])
            assert indices <= set(left.index) or indices <= set(right.index)


def test_missing_feature_values_group_together_even_with_conflicting_labels():
    frame = pd.DataFrame({"a": [None, None, 1, 2, 3, 4, 5, 6, 7, 8],
                          TARGET: ["false", "true"] * 5})
    X_train, X_test, _, _ = split_dataset(frame)
    assert {0, 1} <= set(X_train.index) or {0, 1} <= set(X_test.index)


def test_impossible_single_group_split_is_rejected_without_breaking_group():
    frame = pd.DataFrame({"a": [1, 1], TARGET: ["false", "true"]})
    with pytest.raises(ValueError, match="two distinct feature groups"):
        split_dataset(frame)


@pytest.mark.parametrize("test_size", [0, 1, -0.2, 1.2])
def test_invalid_holdout_fraction_is_rejected(test_size):
    frame = pd.DataFrame({"a": range(10), TARGET: ["false", "true"] * 5})
    with pytest.raises(ValueError, match="test_size"):
        split_dataset(frame, test_size=test_size)


def test_target_cannot_enter_group_key():
    frame = pd.DataFrame({"a": range(10), TARGET: ["false", "true"] * 5})
    with pytest.raises(ValueError, match="must not be included"):
        group_holdout(frame, frame[TARGET])
