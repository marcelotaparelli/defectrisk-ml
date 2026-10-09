"""Reproducible frozen RF + sigmoid artifact, using M3 training rows only."""

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from defectrisk.calibration_uncertainty import (
    CalibratedDefectModel, ProbabilityMap, create_selected_rf, group_folds,
    positive_probabilities, probability_metrics,
)
from defectrisk.data import load_jm1
from defectrisk.model_spec import CALIBRATION_METHOD, FEATURES, MODEL_VERSION, TRAINING_OOF, SIGMOID_PARAMETERS, TRAINING_OOF_SHA256

DESTINATION = Path('artifacts/rf-sigmoid-v1')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    packages = ('defectrisk-ml', 'scikit-learn', 'pandas', 'numpy', 'scipy', 'joblib', 'threadpoolctl')
    return {'python': platform.python_version(),
            **{name: metadata.version(name) for name in packages}}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key,item in value.items()}
    if isinstance(value, (tuple,list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def validate_features(X):
    if tuple(X.columns) != FEATURES:
        raise ValueError('Features must match the ordered original 21-feature schema; target is excluded.')
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError('Features must be numeric; missing values are permitted.')
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature values are not supported.')


def training_fingerprint(X,y):
    digest=hashlib.sha256(json.dumps(list(X.columns)).encode())
    digest.update(pd.util.hash_pandas_object(X,index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(y,index=True).to_numpy().tobytes())
    return digest.hexdigest()


def read_training_oof(path=TRAINING_OOF):
    frame=pd.read_csv(path,index_col=0,dtype={'target':str})
    if not frame.index.is_unique or frame.target.isna().any() or set(frame.target) != {'false','true'}:
        raise ValueError('Invalid training OOF rows or labels.')
    if set(frame.outer_fold) != {1,2,3,4,5}:
        raise ValueError('Expected five training-only outer folds.')
    return frame


def load_training_partition(oof_path=TRAINING_OOF):
    """Select ONLY locked M3 training IDs; do not reconstruct/revisit test split."""
    if sha256(oof_path) != TRAINING_OOF_SHA256:
        raise ValueError('Locked M3 training source checksum changed.')
    reference=read_training_oof(oof_path)
    if len(reference) != 8708 or reference.target.eq('true').sum() != 1685:
        raise ValueError('Expected the established M3 training population.')
    # The cache source contains JM1, but no non-training row is selected/inspected.
    training=load_jm1().loc[reference.index]
    X=training.loc[:,list(FEATURES)]
    y=training['defects'].astype(str)
    if not np.array_equal(y.to_numpy(),reference.target.to_numpy()):
        raise ValueError('Training labels changed from the trusted OOF snapshot.')
    validate_features(X)
    return X,y


def train_artifact(X,y,destination=DESTINATION,*,source_hash=None):
    """Fit maps on three-fold grouped OOF scores, then refit one RF on all X."""
    validate_features(X)
    audits=[]
    folds=group_folds(X,y,'final-training-calibration',0,audits)
    destination=Path(destination)
    if destination.exists():
        raise FileExistsError('Use a fresh artifact directory; frozen artifacts are not overwritten.')
    raw_oof=np.full(len(X),np.nan)
    for fit,validation in folds:
        pipeline=create_selected_rf()
        pipeline.fit(X.iloc[fit],y.iloc[fit])
        raw_oof[validation]=positive_probabilities(pipeline,X.iloc[validation])
    mapper=ProbabilityMap(CALIBRATION_METHOD).fit(pd.Series(raw_oof,index=y.index),y)
    # Final mapping is monotone increasing: preserves full-model risk ranking.
    if mapper.estimator_.coef_[0,0] <= 0:
        raise RuntimeError('Unexpected non-increasing calibration; do not publish artifact.')
    pipeline=create_selected_rf()
    pipeline.fit(X,y)
    specification={
        'model_version':MODEL_VERSION, 'model_name':'weighted_random_forest',
        'purpose':'Risk ranking and human review prioritization; probability is not certainty',
        'classifier_parameters':pipeline.named_steps['classifier'].get_params(deep=False),
        'preprocessing':{step:json_safe(pipeline.named_steps[step].get_params(deep=False))
                         for step in ['imputer','scaler']},
        'feature_schema':[{'name':name,'dtype':str(X[name].dtype),'kind':'numeric','nullable':True} for name in FEATURES],
        'target':{'name':'defects','positive':'true','negative':'false'},
        'calibration':{'method':CALIBRATION_METHOD,'fit_scores':'3-fold stratified group-aware OOF, training only',
                       'n_splits':3,'shuffle':True,'random_state':42,'class_weight':None,
                       'parameters':json_safe(mapper.estimator_.get_params(deep=False)),
                       'slope':float(mapper.estimator_.coef_[0,0]),
                       'intercept':float(mapper.estimator_.intercept_[0])},
        'training':{'dataset':'OpenML 1053 / JM1 version 1',
                    'definition':'Complete existing M3 training partition; original raw feature groups; all rows retained',
                    'rows':len(X),'defective':int(y.eq('true').sum()),
                    'clean':int(y.eq('false').sum()),'sha256':training_fingerprint(X,y),
                    'training_oof_source_sha256':source_hash,'final_test_access':False},
        'review_policy':{'capacity':.30,'rule':'closest attainable count; keep score ties; distance ties prefer fewer',
                         'uses_labels':False,'automatic_high_precision_policy_enabled':False},
        'dependencies':versions(),
        'validation_status':'Training-only calibration CV; external once-only holdout still required',
    }
    destination.mkdir(parents=True)
    joblib.dump({'pipeline':pipeline,'calibration':mapper},destination/'model.joblib',compress=3)
    (destination/'metadata.json').write_text(json.dumps(specification,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(audits).to_csv(destination/'calibration-audits.csv',index=False)
    checksums={name:sha256(destination/name) for name in ['model.joblib','metadata.json','calibration-audits.csv']}
    (destination/'checksums.json').write_text(json.dumps(checksums,indent=2)+'\n')
    return specification


class FinalRiskModel:
    def __init__(self,bundle,specification):
        self.metadata=specification
        self.model=CalibratedDefectModel(bundle['pipeline'],bundle['calibration'],MODEL_VERSION)

    def predict(self,X):
        validate_features(X)
        return self.model.predict(X)


def load_artifact(destination=DESTINATION):
    """Verify bytes/schema/configuration before use. Load only trusted artifacts.

    Joblib/pickle is executable content; a checksum detects corruption but is
    not an authenticity signature. Do not deserialize downloaded/untrusted files.
    """
    destination=Path(destination)
    checksums=json.loads((destination/'checksums.json').read_text())
    expected={'model.joblib','metadata.json','calibration-audits.csv'}
    if set(checksums) != expected or any(sha256(destination/name) != checksums[name] for name in expected):
        raise ValueError('Artifact checksum mismatch.')
    spec=json.loads((destination/'metadata.json').read_text())
    if spec['model_version'] != MODEL_VERSION or tuple(row['name'] for row in spec['feature_schema']) != FEATURES:
        raise ValueError('Artifact version/schema mismatch.')
    for name in ['scikit-learn','numpy','pandas','scipy','joblib']:
        if spec['dependencies'][name] != versions()[name]:
            raise ValueError('Artifact dependency versions differ; recreate the recorded environment.')
    if spec['classifier_parameters'] != create_selected_rf().named_steps['classifier'].get_params(deep=False):
        raise ValueError('Artifact classifier configuration mismatch.')
    if (spec['calibration']['method'] != CALIBRATION_METHOD
            or spec['calibration']['parameters'] != json_safe(LogisticRegression(**SIGMOID_PARAMETERS).get_params(deep=False))):
        raise ValueError('Artifact calibration configuration mismatch.')
    bundle=joblib.load(destination/'model.joblib')
    pipeline,mapper=bundle['pipeline'],bundle['calibration']
    if tuple(pipeline.feature_names_in_) != FEATURES or pipeline.named_steps['classifier'].get_params(deep=False) != spec['classifier_parameters']:
        raise ValueError('Serialized model disagrees with metadata/schema.')
    if mapper.method != CALIBRATION_METHOD or json_safe(mapper.estimator_.get_params(deep=False)) != spec['calibration']['parameters']:
        raise ValueError('Serialized calibration disagrees with metadata.')
    if (float(mapper.estimator_.coef_[0,0]) != spec['calibration']['slope']
            or float(mapper.estimator_.intercept_[0]) != spec['calibration']['intercept']):
        raise ValueError('Serialized calibration coefficients disagree with metadata.')
    for step in ['imputer','scaler']:
        if json_safe(pipeline.named_steps[step].get_params(deep=False)) != spec['preprocessing'][step]:
            raise ValueError('Serialized preprocessing disagrees with metadata.')
    return FinalRiskModel(bundle,spec)


def evaluate_stored_training_oof(oof_path=TRAINING_OOF):
    """Recalculate existing CV evidence; no fit, prediction, or test access."""
    frame=read_training_oof(oof_path)
    return {method:probability_metrics(frame.target,frame[method]) for method in ['raw','sigmoid','isotonic']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate'])
    parser.add_argument('--output',type=Path,default=DESTINATION)
    args=parser.parse_args()
    if args.command=='train':
        X,y=load_training_partition()
        result=train_artifact(X,y,args.output,source_hash=sha256(TRAINING_OOF))
    else:
        result=evaluate_stored_training_oof()
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
