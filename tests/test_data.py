from types import SimpleNamespace

import pandas as pd
import pytest

from defectrisk.data import TARGET, dataset_report, load_jm1


@pytest.fixture(scope="module")
def jm1():
    return load_jm1()


@pytest.mark.integration
def test_successful_data_loading(jm1):
    assert isinstance(jm1, pd.DataFrame)
    assert jm1.shape == (10885, 22)


@pytest.mark.integration
def test_target_exists(jm1):
    assert TARGET in jm1.columns


@pytest.mark.integration
def test_target_has_exactly_two_classes(jm1):
    assert jm1[TARGET].notna().all()
    assert set(jm1[TARGET]) == {"false", "true"}


@pytest.mark.integration
def test_dataset_is_nonempty(jm1):
    assert not jm1.empty


def test_loader_requests_fixed_dataset_and_preserves_raw_frame(monkeypatch, tmp_path):
    raw = pd.DataFrame({"loc": [1.0, 1.0, None], TARGET: ["false", "false", "true"]})
    calls = []

    def fake_fetch_openml(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(frame=raw)

    monkeypatch.setattr("defectrisk.data.fetch_openml", fake_fetch_openml)
    loaded = load_jm1(data_home=tmp_path)
    assert loaded is raw
    assert calls == [
        {
            "data_id": 1053,
            "target_column": "defects",
            "as_frame": True,
            "parser": "pandas",
            "data_home": str(tmp_path),
            "cache": True,
        }
    ]


def test_report_counts_missing_duplicates_and_classes_without_mutating():
    frame = pd.DataFrame(
        {"loc": [1.0, 1.0, None, 2.0], TARGET: ["false", "false", "true", "false"]}
    )
    original = frame.copy(deep=True)
    report = dataset_report(frame)
    assert "Rows: 4; feature columns: 1" in report
    assert "Rows with any missing value: 1" in report
    assert "Duplicate full rows (beyond first): 1" in report
    assert "Duplicate feature rows (beyond first): 1" in report
    assert "75.00" in report
    assert "25.00" in report
    assert "Minority class: true" in report
    pd.testing.assert_frame_equal(frame, original)
