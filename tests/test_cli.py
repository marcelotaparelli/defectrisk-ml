import csv
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from defectrisk import cli
from defectrisk.final_artifact import load_artifact
from defectrisk.model_spec import FEATURES, MODEL_VERSION


def write_csv(path, rows=None, header=None):
    header=list(FEATURES) if header is None else header
    rows=[[str(number+1) for number in range(len(header))]] if rows is None else rows
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle); writer.writerow(header); writer.writerows(rows)
    return path


@pytest.fixture(scope='module')
def frozen_model():
    return load_artifact()


def test_real_frozen_inference_ranking_json_and_no_training(monkeypatch,capsys):
    directory=Path('artifacts/rf-sigmoid-v1')
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    def forbidden(*args,**kwargs):
        pytest.fail('Inference must never fit or access dataset/training functions')
    for estimator in [Pipeline,RandomForestClassifier,LogisticRegression,SimpleImputer,StandardScaler]:
        monkeypatch.setattr(estimator,'fit',forbidden)
    monkeypatch.setattr('defectrisk.final_artifact.load_jm1',forbidden)
    monkeypatch.setattr('defectrisk.final_artifact.train_artifact',forbidden)
    # Go through actual load_artifact, checksum validation, and real predict_proba.
    assert cli.main(['rank','examples/modules.csv','--format','json'])==0
    output=capsys.readouterr()
    rows=json.loads(output.out)
    assert [row['module'] for row in rows]==['module_42','module_17','module_missing','module_03']
    assert [row['rank'] for row in rows]==[1,2,3,4]
    assert [row['risk_probability'] for row in rows]==sorted([row['risk_probability'] for row in rows],reverse=True)
    assert all(row['model_version']==MODEL_VERSION and row['calibration']=='sigmoid' for row in rows)
    assert all(set(row)=={'rank','module','risk_probability','model_version','calibration'} for row in rows)
    assert 'not certainty' in output.err and 'certainty' not in output.out
    assert before=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_table_output_and_explicit_artifact_path(monkeypatch,frozen_model,capsys):
    used=[]
    def loader(path):
        used.append(path)
        return frozen_model
    monkeypatch.setattr(cli,'load_artifact',loader)
    assert cli.main(['rank','examples/modules.csv','--artifact','custom/model','--format','table'])==0
    output=capsys.readouterr()
    assert used==[Path('custom/model')]
    assert output.out.splitlines()[0].split()==['RANK','MODULE','RISK']
    assert output.out.splitlines()[1].split()[:2]==['1','module_42']
    assert 'not certainty' in output.err


def test_features_without_identifiers_and_missing_values(tmp_path):
    values=['1']*21; values[0]=''; values[1]='NaN'; values[2]='NA'; values[3]='N/A'
    path=write_csv(tmp_path/'features.csv',rows=[values])
    modules,features=cli.read_modules(path)
    assert modules==['row_1'] and tuple(features.columns)==FEATURES
    assert features.iloc[0,:4].isna().all()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes)


@pytest.mark.parametrize('identifier',['module','module_id','id'])
def test_optional_identifier_is_excluded_regardless_of_position(tmp_path,identifier):
    header=list(FEATURES); header.insert(5,identifier)
    values=['1']*22; values[5]='NA'  # literal module ID, not a missing feature
    modules,features=cli.read_modules(write_csv(tmp_path/'modules.csv',header=header,rows=[values]))
    assert modules==['NA'] and tuple(features.columns)==FEATURES
    assert identifier not in features


def test_custom_identifier_requires_explicit_name(tmp_path):
    path=write_csv(tmp_path/'modules.csv',header=['path',*FEATURES],rows=[['src/a.c',*['1']*21]])
    with pytest.raises(ValueError,match='schema'):
        cli.read_modules(path)
    names,features=cli.read_modules(path,identifier_column='path')
    assert names==['src/a.c'] and tuple(features.columns)==FEATURES


@pytest.mark.parametrize('header', [list(FEATURES[:-1]), ['loc',*FEATURES],
    [*FEATURES,'extra_numeric'],list(reversed(FEATURES)),
    ['module','id',*FEATURES],['module',*FEATURES,'defects']])
def test_schema_mismatch_is_rejected_before_artifact_load(tmp_path,monkeypatch,header,capsys):
    path=write_csv(tmp_path/'invalid.csv',header=header)
    monkeypatch.setattr(cli,'load_artifact',lambda _:pytest.fail('Reject invalid CSV before model load'))
    with pytest.raises(SystemExit) as error:
        cli.main(['rank',str(path)])
    assert error.value.code==2 and not capsys.readouterr().out


@pytest.mark.parametrize('value',['not-a-number','true','inf','-inf','Infinity','1e999'])
def test_invalid_numeric_values(tmp_path,value):
    row=['1']*21; row[0]=value
    with pytest.raises(ValueError,match='numeric|Infinite'):
        cli.read_modules(write_csv(tmp_path/'invalid.csv',rows=[row]))


@pytest.mark.parametrize('content',['',','.join(FEATURES)+'\n'])
def test_empty_input(tmp_path,content):
    path=tmp_path/'empty.csv'; path.write_text(content)
    with pytest.raises(ValueError,match='empty|no modules'):
        cli.read_modules(path)


def test_malformed_row_not_silently_truncated_or_imputed(tmp_path):
    with pytest.raises(ValueError,match='fields'):
        cli.read_modules(write_csv(tmp_path/'short.csv',rows=[['1']*20]))


def test_missing_identifier_and_feature_cannot_be_an_identifier(tmp_path):
    path=write_csv(tmp_path/'features.csv')
    with pytest.raises(ValueError,match='missing'):
        cli.read_modules(path,identifier_column='module')
    with pytest.raises(ValueError,match='model feature'):
        cli.read_modules(path,identifier_column='loc')


def test_artifact_load_failure_has_nonzero_exit_and_no_results(tmp_path,monkeypatch,capsys):
    path=write_csv(tmp_path/'modules.csv')
    def failed(_):
        raise ValueError('Artifact checksum mismatch.')
    monkeypatch.setattr(cli,'load_artifact',failed)
    with pytest.raises(SystemExit) as error:
        cli.main(['rank',str(path)])
    output=capsys.readouterr()
    assert error.value.code==2 and not output.out
    assert 'Cannot load verified artifact' in output.err and 'checksum mismatch' in output.err


def test_actual_missing_artifact_fails_without_training(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(['rank','examples/modules.csv','--artifact','not-an-artifact'])
    assert error.value.code==2 and 'Cannot load verified artifact' in capsys.readouterr().err


def test_tied_risks_preserve_input_order(frozen_model,monkeypatch,tmp_path):
    values=['5']*21
    path=write_csv(tmp_path/'ties.csv',header=['module',*FEATURES],rows=[['second',*values],['first',*values]])
    monkeypatch.setattr(cli,'load_artifact',lambda _:frozen_model)
    assert [row['module'] for row in cli.rank_modules(path)]==['second','first']


def test_console_entrypoint_matches_callable():
    project=tomllib.loads(Path('pyproject.toml').read_text())
    assert project['project']['scripts']['defectrisk']=='defectrisk.cli:main'
    assert callable(cli.main)
