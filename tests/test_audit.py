import pandas as pd
import pytest

from defectrisk.audit import duplicate_audit, feature_overlap
from defectrisk.data import TARGET


def test_overlap_counts_vectors_rows_and_pairs_with_missing_values():
    left = pd.DataFrame({"a": [1, 1, None, 2], "b": [0, 0, 3, 4]}, index=[10, 11, 12, 13])
    right = pd.DataFrame({"a": [1, 1, 1, None, 2], "b": [0, 0, 0, 3, 5]}, index=[20, 21, 22, 23, 24])
    assert feature_overlap(left, right) == {
        "shared_feature_vectors": 2,
        "left_rows_with_match": 3,
        "right_rows_with_match": 4,
        "matching_row_pairs": 7,
    }


def test_disjoint_vectors_even_with_shared_row_indices():
    assert feature_overlap(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})) == {
        "shared_feature_vectors": 0, "left_rows_with_match": 0,
        "right_rows_with_match": 0, "matching_row_pairs": 0,
    }


def test_audit_counts_conflicts_and_preserves_input():
    # Three duplicate groups: one consistent, two conflicting (including NaN).
    frame = pd.DataFrame({
        "a": [1, 1, 2, 2, 2, None, None] + list(range(3, 16)),
        TARGET: pd.Categorical(["false", "false", "false", "true", "true", "false", "true"]
                               + ["false", "true"] * 6 + ["false"]),
    }, index=range(100, 120))
    original = frame.copy(deep=True)
    result = duplicate_audit(frame)
    assert result["unique_feature_vectors"] == 16
    assert result["duplicate_feature_rows_beyond_first"] == 4
    assert result["duplicate_full_rows_beyond_first"] == 2
    assert result["duplicate_feature_groups"] == 3
    assert result["same_label_duplicate_groups"] == 1
    assert result["conflicting_label_duplicate_groups"] == 2
    assert result["conflicting_examples"][0]["row_indices"] == [102, 103, 104]
    assert result["conflicting_examples"][0]["target_counts"] == {"true": 2, "false": 1}
    assert result["conflicting_examples"][1]["row_indices"] == [105, 106]
    sizes = result["partition_rows"]
    assert sizes["train"] + sizes["test"] == len(frame)
    assert sizes["fit"] + sizes["validation"] == sizes["train"]
    assert sizes["test"] / len(frame) == pytest.approx(0.20, abs=0.05)
    assert duplicate_audit(frame, example_limit=0)["conflicting_examples"] == []
    pd.testing.assert_frame_equal(frame, original)


def test_overlap_rejects_different_feature_schemas():
    with pytest.raises(ValueError, match="schema"):
        feature_overlap(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))


def test_audited_validation_matches_actual_baseline_without_fitting(monkeypatch):
    from defectrisk import baseline
    from defectrisk.split import split_dataset

    frame = pd.DataFrame({"a": [i % 9 for i in range(100)],
                          TARGET: ["false", "true"] * 50})
    X_train, _, y_train, _ = split_dataset(frame)
    actual_split = baseline.group_holdout
    seen = {}

    def record_split(*args, **kwargs):
        parts = actual_split(*args, **kwargs)
        seen["overlap"] = feature_overlap(parts[0], parts[1])
        seen["sizes"] = (len(parts[0]), len(parts[1]))
        return parts

    class StopBeforePipeline(Exception):
        pass

    def stop():
        raise StopBeforePipeline

    monkeypatch.setattr(baseline, "group_holdout", record_split)
    monkeypatch.setattr(baseline, "create_baseline_pipeline", stop)
    with pytest.raises(StopBeforePipeline):
        baseline.baseline_report(X_train, y_train)
    result = duplicate_audit(frame)
    assert result["fit_validation_overlap"] == seen["overlap"]
    assert (result["partition_rows"]["fit"], result["partition_rows"]["validation"]) == seen["sizes"]
