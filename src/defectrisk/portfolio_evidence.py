"""Render portfolio evidence from saved training OOF scores and archived facts.

No model fitting/prediction, dataset loading or historical test-file access.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from defectrisk.calibration_uncertainty import reliability_bins
from defectrisk.gradient_boosting_baseline import review_threshold
from defectrisk.threshold_experiment import apply_threshold, evaluate_thresholds

# Archived aggregate values explicitly supplied for finalization, not new scores.
HISTORICAL = {
    'population':'Historical once-only held-out / frozen raw RF',
    'modules':2177, 'defective':421, 'flagged':652, 'flagged_percent':29.95,
    'recall':.7126, 'precision':.4601, 'f1':.5592, 'average_precision':.6553,
    'tp':300, 'fp':352, 'tn':1404, 'fn':121,
    'calibrated_system':False, 'new_evaluation':False,
    'statement':'This held-out test was evaluated once after model selection was frozen.',
}
NAMES = {
    'lr':'Balanced Logistic Regression', 'rf':'Tuned weighted Random Forest',
    'hgb':'Tuned weighted HGB', 'xgb':'XGBoost (limited nested search)',
}
COLORS = {'lr':'#8C5AA9','rf':'#1763A6','hgb':'#148076','xgb':'#C76A17'}


def stored_evidence(docs=Path('docs')):
    docs=Path(docs)
    sources={name:docs/path for name,path in {
        'tree':'tree-model-tuning-results/oof-probabilities.csv',
        'challenge':'final-model-challenge-results/oof-probabilities.csv',
        'calibration':'calibration-and-uncertainty-results/oof-probabilities.csv',
        'bins':'calibration-and-uncertainty-results/reliability-bins.csv',
        'calibration_audits':'calibration-and-uncertainty-results/audits.csv',
        'tree_audits':'tree-model-tuning-results/audits.csv',
        'challenge_audits':'final-model-challenge-results/audits.csv',
    }.items()}
    frames={name:pd.read_csv(sources[name],index_col=0,dtype={'target':str})
            for name in ['tree','challenge','calibration']}
    tree,challenge,calibration=[frames[name] for name in ['tree','challenge','calibration']]
    for frame in frames.values():
        if not frame.index.is_unique or not frame.index.equals(tree.index) or not np.array_equal(frame.target,tree.target):
            raise ValueError('Training-only OOF rows and targets must match.')
        if not np.array_equal(frame.outer_fold,tree.outer_fold) or set(frame.outer_fold)!={1,2,3,4,5}:
            raise ValueError('Trusted outer folds must match.')
        if set(frame.target) != {'false','true'} or frame.target.isna().any():
            raise ValueError('Invalid target labels.')
    if not np.array_equal(tree.tuned_weighted_random_forest,challenge.tuned_weighted_random_forest):
        raise ValueError('Stored RF scores disagree.')
    scores=pd.DataFrame({'lr':tree.balanced_logistic_regression,
                         'rf':tree.tuned_weighted_random_forest,
                         'hgb':tree.tuned_hist_gradient_boosting,
                         'xgb':challenge.xgboost},index=tree.index)
    for name in scores:
        apply_threshold(scores[name],.5)
    for name in ['raw','sigmoid']:
        apply_threshold(calibration[name],.5)
    for name in ['calibration_audits','tree_audits','challenge_audits']:
        audit=pd.read_csv(sources[name])
        if not audit.shared_feature_vectors.eq(0).all():
            raise ValueError('Stored evaluation boundary is contaminated.')
    metrics={}
    for name in scores:
        threshold=review_threshold(scores[name],.3)
        metrics[name]=evaluate_thresholds(tree.target,scores[name],thresholds=[threshold]).iloc[0].to_dict()
        metrics[name]['average_precision']=average_precision_score(tree.target.eq('true'),scores[name])
    facts={
        'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources.values()},
        'cv_population':{'rows':len(tree),'defective':int(tree.target.eq('true').sum()),
                         'outer_folds':5,'nested_tree_inner_folds':3,
                         'description':'Pooled training OOF; fixed LR, inner-selected RF/HGB/XGB procedures'},
        'cv_review30':metrics, 'historical_held_out':HISTORICAL,
        'new_model_fits':0,'historical_test_prediction_files_accessed':False,
    }
    bins=pd.read_csv(sources['bins'],index_col=0)
    columns=['bin','lower','upper','count','mean_probability','observed_defect_rate']
    for method in ['raw','sigmoid']:
        observed=bins[bins.method==method][columns].to_numpy(dtype=float)
        expected=reliability_bins(calibration.target,calibration[method],method)[columns].to_numpy(dtype=float)
        if observed.shape != expected.shape or not np.allclose(observed,expected,equal_nan=True,rtol=1e-12,atol=1e-12):
            raise ValueError('Stored calibration bins disagree with training-only scores.')
    return scores,tree.target,bins,facts


def review_curve(y,scores):
    """Exact attainable review counts: retain complete probability ties."""
    order=np.argsort(-scores.to_numpy(),kind='stable')
    ranked=scores.to_numpy()[order]
    endpoints=np.flatnonzero(np.r_[ranked[:-1]!=ranked[1:],True])
    caught=y.eq('true').to_numpy()[order].cumsum()[endpoints]
    return np.r_[0,(endpoints+1)/len(y)*100],np.r_[0,caught/y.eq('true').sum()*100]


def generate_figures(docs=Path('docs'),output=Path('docs/figures')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from matplotlib import rcParams

    scores,y,bins,facts=stored_evidence(docs)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':15,
                     'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','savefig.dpi':220})
    def save(fig,name):
        for extension in ['svg','png','pdf']:
            fig.savefig(output/f'{name}.{extension}',bbox_inches='tight',metadata={'Creator':'DefectRisk stored evidence'})
        plt.close(fig)
    def note(fig,text):
        fig.text(.07,.015,text,fontsize=9,color='#454545',va='bottom')
    def percent_axes(ax):
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)

    fig,axes=plt.subplots(1,2,figsize=(11,5))
    fig.subplots_adjust(bottom=.24,wspace=.30)
    labels=['LR','RF','HGB','XGBoost']
    recalls=[facts['cv_review30'][name]['recall']*100 for name in scores]
    ap=[facts['cv_review30'][name]['average_precision'] for name in scores]
    for ax,values,title,limit in zip(axes,[recalls,ap],['Recall at ~30% review','Average Precision'],[(0,100),(0,1)]):
        bars=ax.bar(labels,values,color=[COLORS[name] for name in scores],width=.6)
        ax.set(title=title,ylim=limit)
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
        for bar,value in zip(bars,values):
            ax.text(bar.get_x()+bar.get_width()/2,value+limit[1]*.025,
                    f'{value:.2f}%' if limit[1]==100 else f'{value:.4f}',ha='center',fontsize=11)
    axes[0].yaxis.set_major_formatter(PercentFormatter(100))
    fig.suptitle('Training CV: more complexity, limited gain',fontweight='bold')
    note(fig,'Same five group-aware outer folds; RF/HGB/XGBoost selected inside three inner folds.\nLR fixed. Pooled OOF metrics; no statistical-superiority claim; historical test excluded.')
    save(fig,'model-comparison-cv')

    fig,ax=plt.subplots(figsize=(8,6)); fig.subplots_adjust(bottom=.23)
    for name in scores:
        precision,recall,_=precision_recall_curve(y.eq('true'),scores[name])
        ax.step(recall*100,precision*100,where='post',color=COLORS[name],lw=1.8,
                label=f'{NAMES[name]} · AP {facts["cv_review30"][name]["average_precision"]:.4f}')
    ax.axhline(y.eq('true').mean()*100,color='#737373',ls='--',label='Training prevalence')
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='Defective recall',ylabel='Defective precision',title='Precision–recall · training group-aware OOF')
    percent_axes(ax); ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(fontsize=9,loc='upper right')
    note(fig,'Nested selection for tree models; fixed LR. AP is Average Precision, not trapezoidal PR-AUC.\nSparse extremes can show high precision without useful coverage. Historical test excluded.')
    save(fig,'precision-recall-cv')

    fig,ax=plt.subplots(figsize=(8,6)); fig.subplots_adjust(bottom=.23)
    for name in scores:
        effort,recall=review_curve(y,scores[name])
        ax.step(effort,recall,where='post',color=COLORS[name],lw=1.9,label=NAMES[name])
        row=facts['cv_review30'][name]
        ax.scatter(row['flagged_percent'],row['recall']*100,color=COLORS[name],s=40,zorder=3)
    ax.axvline(30,color='#555555',ls='--',lw=1)
    ax.set(xlim=(0,100),ylim=(0,100),xlabel='Modules flagged for review',ylabel='Known defects captured',title='Review effort versus recall · training OOF')
    percent_axes(ax); ax.xaxis.set_major_formatter(PercentFormatter(100))
    ax.legend(loc='lower right',fontsize=9)
    note(fig,'Thresholds depend on scores and review capacity only; score ties remain together.\nAt ~30% review, tuned RF captures 56.56% (953/1,685). Historical test excluded.')
    save(fig,'review-effort-recall-cv')

    fig,(ax,support)=plt.subplots(2,1,figsize=(8,7),gridspec_kw={'height_ratios':[3,1]})
    fig.subplots_adjust(hspace=.44,bottom=.19)
    ax.plot([0,1],[0,1],'--',color='#737373',label='Perfect calibration')
    for method,color,label in [('raw','#C76A17','Raw fixed RF'),('sigmoid','#1763A6','Sigmoid fixed RF')]:
        b=bins[bins.method==method]; valid=b[b['count']>0]
        ax.plot(valid.mean_probability,valid.observed_defect_rate,'o-',color=color,label=label)
        support.plot((b.lower+b.upper)/2,b['count'],'o-',color=color)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted defect probability',ylabel='Observed defective fraction',title='Reliability · nested group-aware training CV')
    ax.grid(alpha=.2); ax.legend(loc='upper left')
    support.set(xlim=(0,1),xlabel='Probability bin midpoint',ylabel='Rows / bin'); support.grid(alpha=.2)
    note(fig,'Brier: raw 0.18692 → sigmoid 0.13863. Calibration is evaluated on outer held-out rows.\nTen equal-width bins; sparse tails are not confidence guarantees. Historical test excluded.')
    save(fig,'calibration-reliability-cv')

    h=HISTORICAL
    fig,ax=plt.subplots(figsize=(7,6)); fig.subplots_adjust(bottom=.24)
    matrix=np.array([[h['tn'],h['fp']],[h['fn'],h['tp']]])
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=matrix.max())
    for row in range(2):
        for col in range(2):
            ax.text(col,row,f'{matrix[row,col]:,}',ha='center',va='center',fontsize=24,
                    color='white' if matrix[row,col]>matrix.max()*.55 else '#172c43')
    ax.set(xticks=[0,1],xticklabels=['Not flagged','Flagged for review'],
           yticks=[0,1],yticklabels=['Recorded clean','Recorded defective'],
           xlabel='Frozen RF review policy (~30% capacity)',ylabel='Historical recorded label',
           title='Historical once-only held-out outcomes')
    note(fig,'2,177 modules · 421 defective · TP 300 / FP 352 / TN 1,404 / FN 121.\nArchived aggregate counts; no test predictions accessed or evaluation rerun. Raw RF, not calibrated system.')
    save(fig,'historical-confusion-matrix')

    fig,ax=plt.subplots(figsize=(10,4.8)); fig.subplots_adjust(left=.26,bottom=.25,top=.76)
    values=[h['flagged']/h['modules']*100,h['tp']/h['defective']*100]
    ax.barh([1,0],values,color=['#8CA4BC','#1763A6'],height=.5)
    ax.set(yticks=[1,0],yticklabels=['Modules reviewed','Known defects captured'],xlim=(0,100),xlabel='Share of the relevant held-out population')
    ax.xaxis.set_major_formatter(PercentFormatter(100)); ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    for height,value,count in zip([1,0],values,['652 / 2,177 modules','300 / 421 defects']):
        ax.text(value+1.5,height,f'{value:.2f}%\n{count}',va='center',fontsize=12,fontweight='bold')
    fig.suptitle('Review ~30% of modules → detect 71.26% of held-out defects',fontsize=17,fontweight='bold',y=.94)
    note(fig,'Historical once-only result for frozen raw RF: 121 known defects missed; 352 clean modules flagged.\nThis held-out test was evaluated once after model selection was frozen.\nIt is not final validation of the subsequently calibrated system.')
    save(fig,'hero-held-out')
    (output/'evidence.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    return facts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/figures'))
    args=parser.parse_args()
    facts=generate_figures(output=args.output)
    print(json.dumps(facts,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
