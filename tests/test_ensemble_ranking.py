import numpy as np
import pandas as pd
import pytest

from defectrisk import ensemble_ranking as experiment
from defectrisk.cross_validation import training_folds


def source_fixture(tmp_path):
    X = pd.DataFrame({'a': np.repeat(np.arange(30), 3), 'b': np.repeat(np.arange(30) % 7, 3)}, index=range(1000,1090))
    y = pd.Series(['false','false','true']*30, index=X.index, name='defects')
    ids = pd.Series(0, index=X.index)
    for fold, (_, validation) in enumerate(training_folds(X,y), 1):
        ids.iloc[validation] = fold
    rng = np.random.RandomState(42)
    probabilities = pd.DataFrame(rng.uniform(size=(len(X),3)), index=X.index, columns=experiment.BASES)
    tree = pd.DataFrame({'tuned_weighted_random_forest': probabilities.rf,
                         'tuned_hist_gradient_boosting': probabilities.hgb,
                         'target': y, 'outer_fold': ids})
    challenge = pd.DataFrame({'tuned_weighted_random_forest': probabilities.rf,
                              'xgboost': probabilities.xgb, 'target': y, 'outer_fold': ids})
    tree_path, challenge_path = tmp_path/'tree.csv', tmp_path/'challenge.csv'
    tree.to_csv(tree_path)
    challenge.to_csv(challenge_path)
    return X,y,tree_path,challenge_path


def test_fixed_small_weights_arithmetic_and_no_duplicate_strategy():
    probabilities = pd.DataFrame({'rf': [.1,.6], 'hgb': [.4,.3], 'xgb': [.7,.9]})
    before = probabilities.copy()
    scores = experiment.combine_probabilities(probabilities)
    assert len(experiment.WEIGHTS) == 6 and len(scores.columns) == 9
    np.testing.assert_allclose(scores.average_three, probabilities.mean(axis=1))
    np.testing.assert_allclose(scores.rf_xgb, (probabilities.rf+probabilities.xgb)/2)
    np.testing.assert_allclose(scores.rf_60_hgb_20_xgb_20, .6*probabilities.rf+.2*probabilities.hgb+.2*probabilities.xgb)
    pd.testing.assert_frame_equal(probabilities, before)
    assert len(set(experiment.WEIGHTS.values())) == len(experiment.WEIGHTS)


@pytest.mark.parametrize('weights', [(1,0), (-1,1,1), (.3,.3,.3), (np.nan,.5,.5)])
def test_invalid_weight_combinations(weights):
    with pytest.raises(ValueError):
        experiment.combine_probabilities(pd.DataFrame({'rf':[.3], 'hgb':[.4], 'xgb':[.5]}), {'bad':weights})


def test_trusted_rows_labels_folds_and_rf_agreement(tmp_path):
    X,y,tree,challenge = source_fixture(tmp_path)
    probabilities, ids, audits, manifest = experiment.load_trusted_oof(X,y,tree,challenge)
    assert probabilities.index.equals(X.index)
    assert set(ids) == {1,2,3,4,5}
    assert audits.shared_feature_vectors.eq(0).all()
    assert manifest['new_model_fits'] == 0 and not manifest['final_test_access']
    results = experiment.ensemble_report(y, probabilities, ids)
    pooled = results['pooled']
    assert len(pooled) == 9
    assert (pooled.tp+pooled.fn).eq(30).all()
    assert (pooled.tp+pooled.fp).equals(pooled.flagged_modules)
    assert (pooled.tp+pooled.fp+pooled.tn+pooled.fn).eq(len(y)).all()
    for model in pooled.index:
        per_fold = results['per-fold'].query('model == @model')
        assert len(per_fold) == 5
        assert results['summary'].loc[model,'recall_std'] == pytest.approx(per_fold.recall.std(ddof=0))
    for _, row in results['ensemble-gains'].iterrows():
        assert row.new_defects_vs_rf-row.rf_defects_lost == pooled.loc[row.ensemble,'tp']-pooled.loc['rf','tp']
    assert results['defect-patterns'].defects.sum() == 30


@pytest.mark.parametrize('corruption', ['rows','labels','folds','rf','probability'])
def test_corrupted_oof_sources_rejected(tmp_path, corruption):
    X,y,tree,challenge = source_fixture(tmp_path)
    frame = pd.read_csv(challenge,index_col=0,dtype={'target':str})
    if corruption == 'rows':
        frame = frame.iloc[::-1]
    elif corruption == 'labels':
        frame.iloc[0, frame.columns.get_loc('target')] = 'true'
    elif corruption == 'folds':
        frame.outer_fold = 0
    elif corruption == 'rf':
        frame.iloc[0, frame.columns.get_loc('tuned_weighted_random_forest')] += .01
    else:
        frame.iloc[0, frame.columns.get_loc('xgboost')] = np.nan
    frame.to_csv(challenge)
    with pytest.raises(ValueError):
        experiment.load_trusted_oof(X,y,tree,challenge)


def test_unique_defect_counts_and_pairwise_overlap():
    y = pd.Series(['true','true','true','true','false'])
    probabilities = pd.DataFrame({'rf':[.9,.1,.1,.9,.1], 'hgb':[.1,.9,.1,.9,.1],
                                  'xgb':[.1,.1,.9,.9,.1]})
    pooled = pd.DataFrame({'threshold':[.5]*3}, index=experiment.BASES)
    unique,patterns,pairs,gains = experiment.diversity_report(y,probabilities,pooled)
    assert unique.uniquely_caught_among_three.tolist() == [1,1,1]
    assert unique.caught.tolist() == [2,2,2]
    assert patterns.defects.sum() == 4
    assert pairs.shared_caught_defects.eq(1).all()
    assert pairs.flagged_jaccard.eq(1/3).all()
    assert gains.empty


def test_command_only_uses_training_partition_and_training_artifacts(tmp_path, monkeypatch):
    X,y,tree,challenge = source_fixture(tmp_path)

    class Locked:
        def __getattribute__(self, name):
            pytest.fail('Must not inspect final test')

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiment,'load_jm1',lambda: None)
    monkeypatch.setattr(experiment,'split_dataset',lambda _: (X,Locked(),y,Locked()))
    actual = experiment.load_trusted_oof

    def read_only_sources(features,labels,tree_source,challenge_source):
        assert features is X and labels is y
        assert 'final-evaluation' not in str(tree_source) + str(challenge_source)
        return actual(features,labels,tree,challenge)

    monkeypatch.setattr(experiment,'load_trusted_oof',read_only_sources)
    experiment.main()
    assert (tmp_path/'docs/ensemble-ranking-results/manifest.json').exists()
