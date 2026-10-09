import numpy as np
import pandas as pd
import pytest

from defectrisk import calibration_uncertainty as experiment
from defectrisk.split import feature_groups
from defectrisk.trust_policy import ConfidencePolicy, DefectProbability


def fixture():
    X = pd.DataFrame({'a': np.repeat(np.arange(30), 3),
                      'b': np.repeat(np.arange(30) % 7, 3)}, index=range(1000,1090))
    y = pd.Series(['false','false','true']*30, index=X.index, name='defects')
    return X,y


def test_frozen_rf_configuration_and_preprocessing():
    pipeline = experiment.create_selected_rf()
    params = pipeline.named_steps['classifier'].get_params()
    assert {key:params[key] for key in ['n_estimators','max_depth','min_samples_leaf','max_features','class_weight','random_state']} == {
        'n_estimators':200, 'max_depth':8, 'min_samples_leaf':3, 'max_features':'sqrt',
        'class_weight':'balanced_subsample', 'random_state':42}
    assert pipeline.named_steps['imputer'].strategy == 'median'
    assert type(pipeline.named_steps['scaler']).__name__ == 'StandardScaler'


@pytest.mark.parametrize('method', experiment.METHODS)
def test_calibration_maps_valid_ranges_and_determinism(method):
    scores = pd.Series(np.linspace(0,1,120))
    labels = pd.Series(['false','true','false']*40)
    a = experiment.ProbabilityMap(method).fit(scores,labels).predict(scores)
    b = experiment.ProbabilityMap(method).fit(scores,labels).predict(scores)
    np.testing.assert_array_equal(a,b)
    assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
    if method == 'raw':
        np.testing.assert_array_equal(a,scores)
    if method == 'isotonic':
        assert (np.diff(a) >= 0).all()


@pytest.mark.parametrize('scores,labels', [([np.nan,.3],['false','true']),
    ([.1,1.1],['false','true']), ([.1,.2],['false','wrong']),
    ([.1,.2],['false','false']), ([.1],['false','true'])])
def test_invalid_calibration_data(scores,labels):
    with pytest.raises(ValueError):
        experiment.ProbabilityMap('sigmoid').fit(scores,labels)


def test_calibration_labels_must_align():
    with pytest.raises(ValueError,match='align'):
        experiment.ProbabilityMap('raw').fit(pd.Series([.1,.3],index=[1,2]),
                                            pd.Series(['false','true'],index=[2,1]))


def test_group_folds_reproducible_conflicting_duplicates_indivisible():
    X,y = fixture()
    audits=[]
    a=experiment.group_folds(X,y,'calibration',1,audits)
    b=experiment.group_folds(X,y,'calibration',1,[])
    assert len(a)==3 and all(row['shared_feature_vectors']==0 for row in audits)
    for (fit,val),(other_fit,other_val) in zip(a,b):
        np.testing.assert_array_equal(fit,other_fit)
        np.testing.assert_array_equal(val,other_val)
        assert not set(feature_groups(X).iloc[fit]) & set(feature_groups(X).iloc[val])
    with pytest.raises(ValueError):
        experiment.group_folds(X.assign(defects=y),y,'bad',1,[])


def test_calibration_pack_never_predicts_fit_rows_or_outer_labels(monkeypatch):
    X,y=fixture()
    heldout = pd.DataFrame({'a':[101,102], 'b':[0,1]},index=[2000,2001])
    seen=[]

    class SpyRF:
        classes_=np.array(['false','true'])
        def fit(self,features,labels):
            assert features.index.equals(labels.index)
            assert 'defects' not in features
            self.fit_ids=set(features.index)
            self.fit_vectors=set(map(tuple,features.to_numpy()))
            seen.append(self.fit_ids)
            return self
        def predict_proba(self,features):
            assert not self.fit_ids & set(features.index)
            assert not self.fit_vectors & set(map(tuple,features.to_numpy()))
            p=.1+.8*features.a.to_numpy()/(features.a.max()+1)
            return np.column_stack([1-p,p])

    monkeypatch.setattr(experiment,'create_selected_rf',SpyRF)
    original=experiment.ProbabilityMap.fit
    calls=[]
    def map_fit(self,scores,labels):
        assert len(scores)==len(X) and labels.index.equals(y.index)
        assert np.isfinite(scores).all()
        calls.append(self.method)
        return original(self,scores,labels)
    monkeypatch.setattr(experiment.ProbabilityMap,'fit',map_fit)
    audits,sizes=[],[]
    result=experiment.calibrated_pack(X,y,heldout,level='test',outer_fold=1,audits=audits,sizes=sizes)
    assert set(result)=={'raw','sigmoid'}  # isotonic is not appropriate at this size
    assert len(seen)==4 and seen[-1]==set(X.index)
    assert calls==['raw','sigmoid'] and sizes[0]['isotonic_eligible'] is False
    assert all(row['shared_feature_vectors']==0 for row in audits)


def test_tail_candidates_keep_ties_and_count_distinct_original_groups():
    scores=pd.Series([.9,.9,.8,.2,.1])
    y=pd.Series(['true','false','true','false','false'])
    groups=pd.Series([1,1,2,3,4])
    high=experiment.tail_candidates(y,scores,groups,high=True)
    assert high['count'].tolist()==[2,3,4,5]
    assert high.groups.tolist()==[1,2,3,4]
    assert high.precision.tolist()==[.5,2/3,.5,.4]
    low=experiment.tail_candidates(y,scores,groups,high=False)
    assert low['count'].tolist()==[1,2,3,5]
    assert low.groups.tolist()==[1,2,3,4]


def test_confidence_selection_support_gates_disable_tiny_perfect_high():
    scores=pd.Series([.99]+[.2]*60+[.05]*60)
    labels=pd.Series(['true']+['false','true']*30+['false']*60)
    groups=pd.Series(np.arange(len(scores)))
    policy=experiment.select_confidence_policy(labels,scores,groups)
    assert policy.high_threshold is None and policy.low_threshold==.05
    groups[:]=1
    policy=experiment.select_confidence_policy(labels,scores,groups)
    assert policy.high_threshold is None and policy.low_threshold is None


def test_confidence_selection_can_enable_supported_high():
    scores=pd.Series([.95]*60+[.2]*60+[.05]*60)
    labels=pd.Series(['true']*60+['false','true']*30+['false']*60)
    policy=experiment.select_confidence_policy(labels,scores,pd.Series(np.arange(len(scores))))
    assert policy.high_threshold==.95 and policy.low_threshold==.05


def test_policy_count_arithmetic_and_empty_high_not_perfect():
    y=pd.Series(['false','true','false','true','true'])
    scores=pd.Series([.05,.10,.50,.90,.95])
    metrics=experiment.policy_metrics(y,scores,ConfidencePolicy(.10,.90))
    assert metrics['auto_count']==4 and metrics['uncertain_count']==1
    assert metrics['coverage_percent']==80 and metrics['high_precision']==1
    assert metrics['high_recall']==2/3 and metrics['low_false_negatives']==1
    assert metrics['auto_error_rate']==.25
    empty=experiment.policy_metrics(y,scores,ConfidencePolicy(None,None))
    assert np.isnan(empty['high_precision']) and np.isnan(empty['auto_error_rate'])
    assert empty['high_recall']==0 and empty['uncertain_percent']==100
    with pytest.raises(ValueError,match='align'):
        experiment.policy_metrics(y,scores.iloc[::-1],ConfidencePolicy(.1,.9))


def test_reliability_endpoints_empty_bins_and_counts():
    y=pd.Series(['false','true','true'])
    scores=pd.Series([0,.1,1])
    result=experiment.reliability_bins(y,scores,'raw')
    assert result['count'].sum()==3 and len(result)==10
    assert result.loc[0,'count']==1 and result.loc[1,'count']==1 and result.loc[9,'count']==1
    assert np.isnan(result.loc[5,'mean_probability'])


def test_nested_selection_uses_only_outer_training_labels(monkeypatch):
    X,y=fixture()
    pack_calls=[]
    def pack(features,labels,validation,**kwargs):
        assert not set(features.index)&set(validation.index)
        assert features.index.equals(labels.index)
        assert not set(feature_groups(X).loc[features.index]) & set(feature_groups(X).loc[validation.index])
        pack_calls.append(kwargs['level'])
        # Raw .95 is poorly calibrated at this 1/3 prevalence; sigmoid is best.
        return {'raw':np.full(len(validation),.95), 'sigmoid':np.full(len(validation),1/3),
                'isotonic':np.full(len(validation),.5)}
    monkeypatch.setattr(experiment,'calibrated_pack',pack)
    original=experiment.select_confidence_policy
    def select(labels,scores,groups):
        assert len(labels) < len(y) and labels.index.equals(scores.index)
        return original(labels,scores,groups)
    monkeypatch.setattr(experiment,'select_confidence_policy',select)
    results=experiment.study(X,y)
    assert results['choices'].calibration_method.eq('sigmoid').all()
    assert pack_calls.count('calibration-inner')==15 and pack_calls.count('calibration-outer')==5
    assert results['oof-probabilities'].outer_fold.value_counts().sum()==len(y)
    assert results['audits'].shared_feature_vectors.eq(0).all()


def test_training_runner_discards_locked_test_objects(tmp_path,monkeypatch):
    X,y=fixture()
    class Locked:
        def __getattribute__(self,name):
            pytest.fail('Historical final test must never be accessed')
    monkeypatch.setattr(experiment,'load_jm1',lambda: None)
    monkeypatch.setattr(experiment,'split_dataset',lambda _: (X,Locked(),y,Locked()))
    def study(features,labels):
        assert features is X and labels is y
        return {'test':pd.DataFrame({'safe':[True]})}
    monkeypatch.setattr(experiment,'study',study)
    experiment.run_training_study(tmp_path)
    assert (tmp_path/'manifest.json').exists()
    assert '"final_test_access": false' in (tmp_path/'manifest.json').read_text()


def test_model_response_has_metadata_and_no_policy(monkeypatch):
    class Pipeline:
        classes_=np.array(['true','false'])
        def predict_proba(self,X):
            return np.array([[.7,.3]])
    model=experiment.CalibratedDefectModel(Pipeline(),experiment.ProbabilityMap('raw'))
    response=model.predict(pd.DataFrame({'loc':[1]}))[0]
    assert response.probability==.7 and response.calibration_method=='raw'
    assert response.model_version==experiment.MODEL_VERSION
    assert not hasattr(response,'outcome')
    assert ConfidencePolicy(.1,.8).decide(response).outcome=='UNCERTAIN'


@pytest.mark.parametrize('value', [np.nan,np.inf,-.1,1.1])
def test_probability_and_policy_invalid_values(value):
    with pytest.raises(ValueError):
        DefectProbability(value,'rf','v1','sigmoid')
    with pytest.raises(ValueError):
        ConfidencePolicy(value,.9)


def test_policy_inclusive_boundaries_and_immutability():
    policy=ConfidencePolicy(.1,.9)
    def outcome(value):
        return policy.decide(DefectProbability(value,'rf','v1','sigmoid')).outcome
    assert [outcome(value) for value in [.1,.10001,.89999,.9]]==['LOW','UNCERTAIN','UNCERTAIN','HIGH']
    with pytest.raises(ValueError):
        ConfidencePolicy(.9,.9)
    with pytest.raises(AttributeError):
        policy.low_threshold=.2


def test_calibration_evaluation_overlap_rejected_before_any_fit(monkeypatch):
    X,y=fixture()
    validation=X.iloc[:3].copy()
    validation.index=[2000,2001,2002]  # different rows, identical original vectors
    monkeypatch.setattr(experiment,'create_selected_rf',lambda: pytest.fail('Must reject before fitting'))
    with pytest.raises(RuntimeError,match='contamination'):
        experiment.calibrated_pack(X,y,validation,level='bad',outer_fold=1,audits=[],sizes=[])


def test_calibration_partition_misaligned_labels_rejected():
    X,y=fixture()
    with pytest.raises(ValueError,match='matching'):
        experiment.group_folds(X,y.iloc[::-1],'bad',1,[])
