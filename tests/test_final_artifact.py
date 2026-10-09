import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from defectrisk import final_artifact as artifact
from defectrisk.calibration_uncertainty import ProbabilityMap
from defectrisk.model_spec import FEATURES, MODEL_VERSION, RF_PARAMETERS, TRAINING_OOF, TRAINING_OOF_SHA256


def training_fixture():
    groups=np.repeat(np.arange(40),3)
    X=pd.DataFrame({name:(groups*(column+1)).astype(float) for column,name in enumerate(FEATURES)},index=range(1000,1120))
    y=pd.Series(np.where(groups>=20,'true','false'),index=X.index,name='defects')
    # Retain conflicting-label duplicate vectors in a controlled fixture.
    y.iloc[np.flatnonzero((groups%7==0)&(np.arange(len(groups))%3==0))]=np.where(
        y.iloc[np.flatnonzero((groups%7==0)&(np.arange(len(groups))%3==0))]=='true','false','true')
    X.loc[X.index[:3],'uniq_Op']=np.nan
    return X,y


@pytest.fixture(scope='module')
def fitted_artifact(tmp_path_factory):
    X,y=training_fixture()
    destination=tmp_path_factory.mktemp('artifact')/'model'
    specification=artifact.train_artifact(X,y,destination)
    return X,y,destination,specification


def test_artifact_round_trip_configuration_and_schema(fitted_artifact):
    X,y,destination,spec=fitted_artifact
    model=artifact.load_artifact(destination)
    result=model.predict(X.iloc[:4])
    assert len(result)==4 and all(p.model_version==MODEL_VERSION and p.calibration_method=='sigmoid' for p in result)
    assert all(0<=p.probability<=1 for p in result)
    assert tuple(row['name'] for row in spec['feature_schema'])==FEATURES
    assert all(spec['classifier_parameters'][key]==value for key,value in RF_PARAMETERS.items())
    assert spec['calibration']['parameters']['C']=='inf'
    assert spec['calibration']['parameters']['class_weight'] is None
    assert spec['calibration']['n_splits']==3
    assert spec['training']['sha256']==artifact.training_fingerprint(X,y)
    assert not spec['training']['final_test_access']
    assert pd.read_csv(destination/'calibration-audits.csv').shared_feature_vectors.eq(0).all()
    with pytest.raises(FileExistsError):
        artifact.train_artifact(X,y,destination)
    with pytest.raises(ValueError,match='schema'):
        model.predict(X.iloc[:2,::-1])
    with pytest.raises(ValueError,match='schema'):
        model.predict(X.iloc[:2].assign(defects='false'))
    with pytest.raises(ValueError,match='Infinite'):
        model.predict(X.iloc[:2].assign(loc=np.inf))


def test_artifact_retraining_deterministic(fitted_artifact,tmp_path):
    X,y,destination,spec=fitted_artifact
    second=tmp_path/'repeat'
    repeated=artifact.train_artifact(X,y,second)
    assert repeated==spec
    a=artifact.load_artifact(destination).predict(X)
    b=artifact.load_artifact(second).predict(X)
    np.testing.assert_array_equal([p.probability for p in a],[p.probability for p in b])


@pytest.mark.parametrize('filename',['metadata.json','model.joblib','calibration-audits.csv'])
def test_corrupted_artifact_rejected_before_deserialization(fitted_artifact,tmp_path,monkeypatch,filename):
    _,_,source,_=fitted_artifact
    destination=tmp_path/'corrupt'; destination.mkdir()
    for path in source.iterdir():
        (destination/path.name).write_bytes(path.read_bytes())
    path=destination/filename
    path.write_bytes(path.read_bytes()+b' altered')
    monkeypatch.setattr(artifact.joblib,'load',lambda _:pytest.fail('Must check bytes before deserialization'))
    with pytest.raises(ValueError,match='checksum'):
        artifact.load_artifact(destination)


@pytest.mark.parametrize('change',['schema','version','forest','calibration','dependencies'])
def test_metadata_contract_rejected_even_with_recalculated_checksum(fitted_artifact,tmp_path,monkeypatch,change):
    _,_,source,_=fitted_artifact
    destination=tmp_path/'contract'; destination.mkdir()
    for path in source.iterdir():
        (destination/path.name).write_bytes(path.read_bytes())
    path=destination/'metadata.json'; spec=json.loads(path.read_text())
    if change=='schema': spec['feature_schema'].reverse()
    if change=='version': spec['model_version']='another-model'
    if change=='forest': spec['classifier_parameters']['max_depth']=16
    if change=='calibration': spec['calibration']['parameters']['class_weight']='balanced'
    if change=='dependencies': spec['dependencies']['scikit-learn']='0.0'
    path.write_text(json.dumps(spec))
    checksums=json.loads((destination/'checksums.json').read_text())
    checksums['metadata.json']=artifact.sha256(path)
    (destination/'checksums.json').write_text(json.dumps(checksums))
    monkeypatch.setattr(artifact.joblib,'load',lambda _:pytest.fail('Metadata contract must be checked first'))
    with pytest.raises(ValueError):
        artifact.load_artifact(destination)


def test_training_loader_selects_only_locked_ids_without_reconstructing_test(monkeypatch):
    reference=artifact.read_training_oof()
    observed=[]
    class LocGuard:
        def __getitem__(self,indices):
            assert indices.equals(reference.index)
            observed.append(indices)
            frame=pd.DataFrame({name:np.zeros(len(indices)) for name in FEATURES},index=indices)
            return frame.assign(defects=reference.target)
    class CachedDataset:
        loc=LocGuard()
        def __getattr__(self,name):
            pytest.fail('Only locked training row selection is allowed')
    monkeypatch.setattr(artifact,'load_jm1',lambda:CachedDataset())
    X,y=artifact.load_training_partition()
    assert len(observed)==1 and len(X)==8708 and y.eq('true').sum()==1685


def test_changed_training_source_rejected_before_dataset_access(tmp_path,monkeypatch):
    source=tmp_path/'changed.csv'; source.write_bytes(Path(TRAINING_OOF).read_bytes()+b'\n')
    monkeypatch.setattr(artifact,'load_jm1',lambda:pytest.fail('Invalid training IDs must not reach dataset loader'))
    with pytest.raises(ValueError,match='checksum'):
        artifact.load_training_partition(source)


def test_model_card_matches_actual_configuration_schema_version_and_checksum():
    card=Path('docs/model-card.md').read_text()
    spec=json.loads(Path('artifacts/rf-sigmoid-v1/metadata.json').read_text())
    checksums=json.loads(Path('artifacts/rf-sigmoid-v1/checksums.json').read_text())
    code=re.search(r'```python\n(RandomForestClassifier\(.*?\))\n```',card,re.S).group(1)
    call=ast.parse(code).body[0].value
    parameters={kw.arg:ast.literal_eval(kw.value) for kw in call.keywords}
    assert parameters==spec['classifier_parameters']==artifact.create_selected_rf().named_steps['classifier'].get_params(deep=False)
    schema=re.search(r'```text\n(.*?)\n```',card,re.S).group(1).split(', ')
    assert tuple(schema)==FEATURES==tuple(row['name'] for row in spec['feature_schema'])
    assert spec['model_version']==MODEL_VERSION and MODEL_VERSION in card
    assert artifact.sha256('artifacts/rf-sigmoid-v1/model.joblib')==checksums['model.joblib']
    assert checksums['model.joblib'] in card
    assert spec['training']['training_oof_source_sha256']==TRAINING_OOF_SHA256==artifact.sha256(TRAINING_OOF)


def test_evaluation_command_only_reuses_training_oof(monkeypatch,capsys):
    monkeypatch.setattr(artifact,'load_jm1',lambda:pytest.fail('Saved evidence needs no data loader'))
    monkeypatch.setattr(artifact,'create_selected_rf',lambda:pytest.fail('Saved evaluation must not fit a model'))
    monkeypatch.setattr('sys.argv',['final_artifact','evaluate'])
    artifact.main()
    result=json.loads(capsys.readouterr().out)
    assert result['sigmoid']['brier']==pytest.approx(.13862609543192814)
