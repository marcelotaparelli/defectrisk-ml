import numpy as np
import pandas as pd
import pytest

from defectrisk import feature_engineering as study
from defectrisk.cross_validation import training_folds
from defectrisk.data import TARGET


def training():
    group = np.repeat(np.arange(30, dtype=float), 3)
    columns = ('loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't',
               'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd',
               'total_Op', 'total_Opnd', 'branchCount')
    X = pd.DataFrame({name: group + 1 for name in columns}, index=range(1000, 1090))
    X['loc'] = group * 3
    X['b'], X['t'] = X.v / 3000, X.e / 18
    X['n'] = 2 * X.total_Op
    y = pd.Series(['false', 'false', 'true'] * 30, index=X.index, name=TARGET)
    return X, y


def test_safe_division_missing_zero_negative_and_infinite():
    numerator = pd.Series([2., 0., 2., np.nan, np.inf, -1.])
    denominator = pd.Series([4., 0., -1., 1., 1., 1.])
    result = study.safe_divide(numerator, denominator)
    assert result.iloc[0] == .5
    assert result.iloc[1:].isna().all()


def test_ratios_derived_before_logs_preserve_data_and_schema():
    X, y = training()
    original = X.copy(deep=True)
    transform = study.FeatureRepresentation('engineered').fit(X, y)
    result = transform.transform(X)
    assert result.shape[1] == X.shape[1] + 8
    assert result.index.equals(X.index)
    for name, (numerator, denominator) in study.RATIOS.items():
        pd.testing.assert_series_equal(result[name], study.safe_divide(X[numerator], X[denominator]), check_names=False)
    assert not np.isinf(result.to_numpy()).any()
    assert TARGET not in result
    pd.testing.assert_frame_equal(X, original)
    with pytest.raises(ValueError, match='schema'):
        transform.transform(X[X.columns[::-1]])
    with pytest.raises(ValueError, match='schema'):
        study.FeatureRepresentation().fit(X.assign(defects=y))


def test_skew_and_redundancy_are_fit_only_ignore_target():
    X = pd.DataFrame({'loc': [0.] * 19 + [1000.], 'v': range(20),
                      'b': np.arange(20) / 3000., 'e': range(20),
                      't': np.arange(20) / 18., 'total_Op': range(20), 'n': np.arange(20)*2.})
    log = study.FeatureRepresentation('logs').fit(X, ['true']*20)
    assert log.log_features_ == ['loc']
    validation = X.copy()
    validation.loc[0, 'loc'] = -1
    transformed = log.transform(validation)
    assert transformed['log1p__loc'].iloc[0] != transformed['log1p__loc'].iloc[0]
    assert 'loc' not in transformed
    assert transformed['log1p__loc'].iloc[-1] == pytest.approx(np.log1p(1000))
    prune = study.FeatureRepresentation('pruned').fit(X)
    assert prune.drop_features_ == ['b', 't', 'n']
    validation.b = np.arange(20)[::-1]
    assert 'b' not in prune.transform(validation)
    assert study.FeatureRepresentation('logs').fit(X, ['false']*20).log_features_ == log.log_features_


def test_quality_conflicts_associations_and_no_rows_modified():
    X, y = training()
    before_X, before_y = X.copy(deep=True), y.copy(deep=True)
    tables, conflicts = study.feature_quality_audit(X, y)
    assert conflicts['conflicting_groups_in_training'] == 30
    assert conflicts['rows_in_conflicting_training_groups'] == 90
    assert conflicts['clean_rows_in_conflicting_groups'] == 60
    assert conflicts['defective_rows_in_conflicting_groups'] == 30
    assert conflicts['minimum_row_errors_for_identical_vector_predictions'] == 30
    assert len(tables['target-associations']) == 21
    assert len(tables['ratio-associations']) == 8
    assert tables['formula-checks'].iloc[0].within_rounding_rows == 90
    assert tables['distributions'].loc['loc', 'zero_rows'] == 3
    # A rank measure correctly handles extreme scales and a binary target.
    labels = pd.Series(['false']*10 + ['true']*10)
    scores = study.univariate_associations(pd.DataFrame({'count': np.arange(20)**4}), labels)
    assert scores.loc['count', 'auc_increasing'] == 1
    assert scores.loc['count', 'rank_biserial'] == 1
    pd.testing.assert_frame_equal(X, before_X)
    pd.testing.assert_series_equal(y, before_y)


def test_nested_same_parameters_per_variant_and_original_groups(monkeypatch):
    X, y = training()
    seen = []
    original_factory = study.representation_pipeline
    original_select = study.select_candidate
    selections = []

    def select(model, candidates, features, labels, folds, **kwargs):
        selections.append(features.index)
        return original_select(model, candidates, features, labels, folds, **kwargs)

    def factory(model, params, variant):
        pipeline = original_factory(model, params, variant)
        fit = pipeline.fit
        record = {'model': model, 'params': params, 'variant': variant, 'pipeline': pipeline}
        seen.append(record)

        def checked_fit(features, labels):
            record['fit'] = features.index
            fit(features, labels)
            transform = pipeline.named_steps['features']
            imputer = pipeline.named_steps['imputer']
            np.testing.assert_allclose(imputer.statistics_, transform.transform(features).median())
            return pipeline

        monkeypatch.setattr(pipeline, 'fit', checked_fit)
        return pipeline

    monkeypatch.setattr(study, 'select_candidate', select)
    monkeypatch.setattr(study, 'representation_pipeline', factory)
    result = study.feature_comparison(X, y, candidates={model: [{}] for model in study.FAMILIES}, permutation=False)
    assert len(seen) == 50
    assert len(result['audits']) == 20
    assert result['audits'].shared_feature_vectors.eq(0).all()
    assert len(result['representation-audits']) == 100
    assert result['representation-audits'].shared_feature_vectors.eq(0).all()
    for i, (fit, validation) in enumerate(training_folds(X, y)):
        for indices in selections[i*2:i*2+2]:
            assert indices.equals(X.iloc[fit].index)
            assert not indices.intersection(X.iloc[validation].index).size
        for record in seen[i*10:i*10+10]:
            assert record['fit'].equals(X.iloc[fit].index)
            assert record['params'] == {}
    oof = result['oof-probabilities']
    assert len(oof) == len(X) and not oof.isna().any().any()
    pooled = result['pooled']
    assert (pooled.tp + pooled.fn).eq(30).all()
    assert (pooled.tp + pooled.fp).equals(pooled.flagged_modules)


def test_new_reduced_vector_collisions_rejected(monkeypatch):
    X, y = training()
    monkeypatch.setattr(study.FeatureRepresentation, 'transform',
                        lambda self, features: pd.DataFrame({'constant': 1.}, index=features.index))
    with pytest.raises(RuntimeError, match='contamination'):
        study.representation_boundary_audits(X, y)


def test_no_final_test_access_in_command(monkeypatch, tmp_path):
    X, y = training()

    class Locked:
        def __getattribute__(self, name):
            pytest.fail('Final test must not be accessed')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(study, 'load_jm1', lambda: None)
    monkeypatch.setattr(study, 'split_dataset', lambda _: (X, Locked(), y, Locked()))

    def checked(features, labels, **kwargs):
        assert features is X and labels is y
        return {'pooled': pd.DataFrame({'value': [1]})}

    monkeypatch.setattr(study, 'feature_comparison', checked)
    study.main()
    assert (tmp_path / 'docs/feature-engineering-results/conflict-summary.json').exists()
