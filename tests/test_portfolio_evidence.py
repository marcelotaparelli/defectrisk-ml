import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from defectrisk import portfolio_evidence as evidence


def test_saved_sources_match_and_historical_figures_need_no_test_predictions(monkeypatch):
    original=pd.read_csv
    accessed=[]
    def read(path,*args,**kwargs):
        assert 'final-evaluation' not in str(path) and 'test-predictions' not in str(path)
        accessed.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_csv',read)
    scores,y,bins,facts=evidence.stored_evidence()
    assert len(scores)==8708 and y.eq('true').sum()==1685
    assert facts['cv_review30']['rf']['tp']==953 and facts['cv_review30']['lr']['tp']==906
    assert facts['cv_review30']['hgb']['tp']==945 and facts['cv_review30']['xgb']['tp']==926
    assert facts['new_model_fits']==0 and not facts['historical_test_prediction_files_accessed']
    h=facts['historical_held_out']
    assert h['tp']+h['fp']==h['flagged'] and h['tp']+h['fn']==h['defective']
    assert h['tp']+h['fp']+h['tn']+h['fn']==h['modules']
    assert round(h['tp']/h['defective'],4)==h['recall']
    assert round(h['tp']/h['flagged'],4)==h['precision']
    assert not h['calibrated_system'] and not h['new_evaluation']


def test_review_curve_keeps_complete_ties_and_measures_defects():
    y=pd.Series(['true','false','true','false'])
    scores=pd.Series([.9,.9,.5,.1])
    review,recall=evidence.review_curve(y,scores)
    np.testing.assert_array_equal(review,[0,50,75,100])
    np.testing.assert_array_equal(recall,[0,50,100,100])


def test_saved_figure_metrics_and_source_hashes_still_match():
    _,_,_,facts=evidence.stored_evidence()
    saved=json.loads(Path('docs/figures/evidence.json').read_text())
    assert facts==saved
