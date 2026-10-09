"""Nested group-aware calibration and abstention study; no product UI or CLI."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from defectrisk.audit import feature_overlap
from defectrisk.baseline import POSITIVE_CLASS
from defectrisk.cross_validation import training_folds
from defectrisk.gradient_boosting_baseline import review_threshold
from defectrisk.random_forest_class_weight import create_weight_comparison_pipeline
from defectrisk.model_spec import RF_PARAMETERS, SIGMOID_PARAMETERS
from defectrisk.data import load_jm1
from defectrisk.split import feature_groups, split_dataset
from defectrisk.threshold_experiment import apply_threshold, evaluate_thresholds
from defectrisk.tree_model_tuning import audited_folds
from defectrisk.trust_policy import ConfidencePolicy, DefectProbability

METHODS = ('raw', 'sigmoid', 'isotonic')
POLICIES = [(low, high) for low in (.05, .10, .15, .20) for high in (.50, .70, .90, .95)]
MODEL_VERSION = 'defectrisk-rf-calibration-study-v1'


def create_selected_rf():
    return create_weight_comparison_pipeline('weighted_random_forest').set_params(
        **{f'classifier__{key}': value for key, value in RF_PARAMETERS.items()}
    )


def positive_probabilities(pipeline, X):
    return pipeline.predict_proba(X)[:, list(pipeline.classes_).index(POSITIVE_CLASS)]


class ProbabilityMap:
    """Unweighted natural-prevalence sigmoid or nondecreasing isotonic map."""

    def __init__(self, method):
        if method not in METHODS:
            raise ValueError('Unknown probability map.')
        self.method = method

    def fit(self, scores, labels):
        apply_threshold(scores, .5)
        if len(scores) != len(labels) or set(labels) != {'false', 'true'}:
            raise ValueError('Calibration requires aligned, valid labels of both classes.')
        if isinstance(scores, pd.Series) and isinstance(labels, pd.Series) and not scores.index.equals(labels.index):
            raise ValueError('Calibration indices must align.')
        target = (np.asarray(labels) == POSITIVE_CLASS).astype(int)
        if set(target) != {0, 1}:
            raise ValueError('Calibration requires both classes.')
        if self.method == 'sigmoid':
            # Platt-style unpenalized logistic mapping on scalar RF probabilities.
            # No class weights: calibration must reflect natural prevalence.
            self.estimator_ = LogisticRegression(**SIGMOID_PARAMETERS)
            self.estimator_.fit(np.asarray(scores).reshape(-1, 1), target)
        elif self.method == 'isotonic':
            self.estimator_ = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds='clip')
            self.estimator_.fit(scores, target)
        return self

    def predict(self, scores):
        apply_threshold(scores, .5)
        if self.method == 'sigmoid':
            column = list(self.estimator_.classes_).index(1)
            return self.estimator_.predict_proba(np.asarray(scores).reshape(-1, 1))[:, column]
        if self.method == 'isotonic':
            return self.estimator_.predict(scores)
        return np.asarray(scores, dtype=float)


class CalibratedDefectModel:
    """Model response carries a probability and provenance; no policy decision."""

    def __init__(self, fitted_pipeline, fitted_map, model_version=MODEL_VERSION):
        self.model_version = model_version
        self.pipeline = fitted_pipeline
        self.probability_map = fitted_map

    def predict(self, X):
        calibrated = self.probability_map.predict(positive_probabilities(self.pipeline, X))
        return [DefectProbability(float(p), 'weighted_random_forest', self.model_version,
                                  self.probability_map.method) for p in calibrated]


def group_folds(X, y, level, outer_fold, audit_log):
    if not X.index.is_unique or not X.index.equals(y.index):
        raise ValueError('Training features and labels must have unique matching indices.')
    if y.isna().any() or set(y) != {'false', 'true'}:
        raise ValueError('Training requires exactly the non-missing labels false and true.')
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds, audit = audited_folds(X, y, splitter.split(X, y, groups=feature_groups(X)), level, outer_fold)
    audit_log.extend(audit)
    return folds


def calibrated_pack(X_fit, y_fit, X_validation, *, level, outer_fold, audits, sizes):
    if set(X_fit.index) & set(X_validation.index) or feature_overlap(X_fit, X_validation)['shared_feature_vectors']:
        raise RuntimeError('Calibration evaluation boundary has feature-vector contamination.')
    folds = group_folds(X_fit, y_fit, level, outer_fold, audits)
    raw_oof = np.full(len(X_fit), np.nan)
    for fit, validation in folds:
        model = create_selected_rf()
        model.fit(X_fit.iloc[fit], y_fit.iloc[fit])
        raw_oof[validation] = positive_probabilities(model, X_fit.iloc[validation])
    positives = int((y_fit == POSITIVE_CLASS).sum())
    groups = int(feature_groups(X_fit).nunique())
    # Approximate support gate; tail support and remaining overfit are evaluated.
    eligible = (len(X_fit) >= 1000 and groups >= 1000 and positives >= 100
                and len(X_fit)-positives >= 100 and len(np.unique(raw_oof)) >= 20)
    sizes.append({'level': level, 'outer_fold': outer_fold, 'calibration_rows': len(X_fit),
                  'calibration_groups': groups, 'positive': positives,
                  'isotonic_eligible': eligible})
    maps = {method: ProbabilityMap(method).fit(raw_oof, y_fit)
            for method in METHODS if method != 'isotonic' or eligible}
    model = create_selected_rf()
    model.fit(X_fit, y_fit)
    raw_validation = positive_probabilities(model, X_validation)
    return {name: mapper.predict(raw_validation) for name, mapper in maps.items()}


def probability_metrics(y, scores):
    binary = y == POSITIVE_CLASS
    fixed = evaluate_thresholds(y, scores, thresholds=[.5]).iloc[0].to_dict()
    budget = evaluate_thresholds(y, scores, thresholds=[review_threshold(scores, .3)]).iloc[0].to_dict()
    return {'brier': brier_score_loss(binary, scores),
            'average_precision': average_precision_score(binary, scores),
            **{f'default_{key}': fixed[key] for key in ['precision','recall','f1','flagged_percent']},
            **{f'budget30_{key}': budget[key] for key in ['precision','recall','f1','flagged_percent','tp','fn']}}


def policy_metrics(y, scores, policy):
    apply_threshold(scores, .5)
    if not y.index.equals(scores.index) or not y.index.is_unique:
        raise ValueError('Policy labels/scores must align.')
    if y.isna().any() or not set(y) <= {'false','true'}:
        raise ValueError('Invalid labels.')
    low = scores <= policy.low_threshold if policy.low_threshold is not None else pd.Series(False,index=y.index)
    high = scores >= policy.high_threshold if policy.high_threshold is not None else pd.Series(False,index=y.index)
    automatic = low | high
    defective = y == POSITIVE_CLASS
    tp, fp = int((high & defective).sum()), int((high & ~defective).sum())
    low_fn = int((low & defective).sum())
    return {'low_count': int(low.sum()), 'high_count': int(high.sum()),
            'auto_count': int(automatic.sum()), 'uncertain_count': int((~automatic).sum()),
            'coverage_percent': automatic.mean()*100, 'uncertain_percent': (~automatic).mean()*100,
            'high_percent': high.mean()*100, 'high_precision': tp/high.sum() if high.sum() else np.nan,
            'high_recall': tp/defective.sum() if defective.sum() else np.nan,
            'high_tp': tp, 'high_fp': fp, 'low_false_negatives': low_fn,
            'low_defect_rate': low_fn/low.sum() if low.sum() else np.nan,
            'auto_error_rate': (low_fn+fp)/automatic.sum() if automatic.sum() else np.nan,
            'uncertain_defects': int((~automatic & defective).sum())}


def tail_candidates(y, scores, groups, *, high):
    table = pd.DataFrame({'score': scores, 'defective': (y == POSITIVE_CLASS).astype(int), 'group': groups}, index=y.index)
    # Prefix counts at complete tie boundaries preserve indivisible score ties.
    table = table.sort_values('score', ascending=not high, kind='stable')
    cumulative_positive = table.defective.cumsum().to_numpy()
    cumulative_groups = (~table['group'].duplicated()).cumsum().to_numpy()
    endpoints = np.flatnonzero(~table.score.duplicated(keep='last').to_numpy())
    count = endpoints + 1
    rate = cumulative_positive[endpoints] / count
    return pd.DataFrame({'threshold': table.score.iloc[endpoints].to_numpy(),
                         'count': count, 'groups': cumulative_groups[endpoints],
                         'precision': rate, 'defect_rate': rate})


def select_confidence_policy(y, scores, groups, *, min_count=50, min_groups=20):
    """Fit-only empirical 90% HIGH / <=5% defective LOW, with support gates."""
    high = tail_candidates(y,scores,groups,high=True)
    qualified = high[(high.precision >= .9) & (high['count'] >= min_count) & (high.groups >= min_groups)]
    high_threshold = float(qualified.sort_values('count',ascending=False).iloc[0].threshold) if len(qualified) else None
    low = tail_candidates(y,scores,groups,high=False)
    qualified = low[(low.defect_rate <= .05) & (low['count'] >= min_count) & (low.groups >= min_groups)]
    if high_threshold is not None:
        qualified = qualified[qualified.threshold < high_threshold]
    low_threshold = float(qualified.sort_values('count',ascending=False).iloc[0].threshold) if len(qualified) else None
    return ConfidencePolicy(low_threshold, high_threshold)


def reliability_bins(y, scores, method):
    # Fixed 0.1-wide bins, including sparse/empty tail bins and support counts.
    bin_id = np.minimum((scores.to_numpy()*10).astype(int), 9)
    rows=[]
    for number in range(10):
        mask=bin_id==number
        values=scores.iloc[np.flatnonzero(mask)]
        labels=(y.iloc[np.flatnonzero(mask)]==POSITIVE_CLASS)
        rows.append({'method':method, 'bin':number, 'lower':number/10, 'upper':(number+1)/10,
                     'count':int(mask.sum()), 'mean_probability': values.mean() if len(values) else np.nan,
                     'observed_defect_rate':labels.mean() if len(values) else np.nan})
    return pd.DataFrame(rows)


def study(X,y):
    outer, audits = audited_folds(X,y,training_folds(X,y),'outer',0)
    oof=pd.DataFrame(np.nan,index=y.index,columns=[*METHODS,'inner_selected'])
    assignments=pd.Series(0,index=y.index)
    sizes, inner_results, choices, nested_policies, per_fold=[],[],[],[],[]
    for number,(fit,validation) in enumerate(outer,1):
        X_fit,y_fit=X.iloc[fit],y.iloc[fit]
        X_validation,y_validation=X.iloc[validation],y.iloc[validation]
        inner=group_folds(X_fit,y_fit,'selection',number,audits)
        inner_oof=pd.DataFrame(np.nan,index=y_fit.index,columns=METHODS)
        for inner_fit,inner_validation in inner:
            pack=calibrated_pack(X_fit.iloc[inner_fit],y_fit.iloc[inner_fit],X_fit.iloc[inner_validation],
                                level='calibration-inner',outer_fold=number,audits=audits,sizes=sizes)
            for method,probabilities in pack.items():
                inner_oof.loc[X_fit.iloc[inner_validation].index,method]=probabilities
        eligible=[method for method in METHODS if inner_oof[method].notna().all()]
        metrics={method:probability_metrics(y_fit,inner_oof[method]) for method in eligible}
        # Fixed method order favors lower complexity only for exact Brier ties.
        chosen=min(eligible,key=lambda method:metrics[method]['brier'])
        for method in eligible:
            inner_results.append({'outer_fold':number,'method':method,**metrics[method], 'selected':method==chosen})
        policy=select_confidence_policy(y_fit,inner_oof[chosen],feature_groups(X_fit))
        pack=calibrated_pack(X_fit,y_fit,X_validation,level='calibration-outer',outer_fold=number,audits=audits,sizes=sizes)
        for method,probabilities in pack.items():
            oof.loc[X_validation.index,method]=probabilities
            per_fold.append({'outer_fold':number,'method':method,
                             **probability_metrics(y_validation,pd.Series(probabilities,index=y_validation.index))})
        oof.loc[X_validation.index,'inner_selected']=pack[chosen]
        assignments.iloc[validation]=number
        choice={'outer_fold':number,'calibration_method':chosen,'low_threshold':policy.low_threshold,
                'high_threshold':policy.high_threshold}
        choices.append(choice)
        nested_policies.append({**choice,**policy_metrics(y_validation,oof.loc[X_validation.index,'inner_selected'],policy)})
        print(f'Completed calibration outer fold {number}/5; selected {chosen}.',flush=True)
    eligible=[name for name in oof if oof[name].notna().all()]
    pooled=pd.DataFrame({name:probability_metrics(y,oof[name]) for name in eligible}).T
    curves=pd.concat([reliability_bins(y,oof[name],name) for name in eligible],ignore_index=True)
    # Mixed inner-selected OOF is the honest estimate of method selection.
    probabilities=oof.inner_selected
    policies=[]
    for low,high in POLICIES:
        policy=ConfidencePolicy(low,high)
        policies.append({'low_threshold':low,'high_threshold':high,
                         **policy_metrics(y,probabilities,policy)})
    tails=tail_candidates(y,probabilities,feature_groups(X),high=True)
    nested=pd.DataFrame(nested_policies)
    totals=nested[['low_count','high_count','auto_count','uncertain_count','high_tp','high_fp','low_false_negatives','uncertain_defects']].sum()
    nested_summary=totals.to_dict()
    nested_summary.update(coverage_percent=100*totals.auto_count/len(y), uncertain_percent=100*totals.uncertain_count/len(y),
                          high_percent=100*totals.high_count/len(y),
                          high_precision=totals.high_tp/totals.high_count if totals.high_count else np.nan,
                          high_recall=totals.high_tp/(y==POSITIVE_CLASS).sum(),
                          low_defect_rate=totals.low_false_negatives/totals.low_count if totals.low_count else np.nan,
                          auto_error_rate=(totals.high_fp+totals.low_false_negatives)/totals.auto_count if totals.auto_count else np.nan)
    return {'calibration-pooled':pooled,'calibration-per-fold':pd.DataFrame(per_fold),
            'inner-calibration-selection':pd.DataFrame(inner_results),'choices':pd.DataFrame(choices),
            'calibration-support':pd.DataFrame(sizes),'audits':pd.DataFrame(audits),
            'reliability-bins':curves,'fixed-confidence-policies':pd.DataFrame(policies),
            'high-precision-tail':tails,'nested-confidence-per-fold':nested,
            'nested-confidence-pooled':pd.DataFrame([nested_summary]),
            'oof-probabilities':oof.assign(target=y,outer_fold=assignments)}


def run_training_study(destination=Path('docs/calibration-and-uncertainty-results')):
    """Reproduce offline evidence; deliberately discard both locked test objects.

    No final-test artifacts, labels, probabilities or fitted historical estimator
    are loaded. This function does not install a production model or policy.
    """
    X, _, y, _ = split_dataset(load_jm1())
    results = study(X, y)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in results.items():
        frame.to_csv(destination / f'{name}.csv')
    parameters = create_selected_rf().named_steps['classifier'].get_params()
    (destination / 'model-parameters.json').write_text(json.dumps(parameters, indent=2))
    (destination / 'manifest.json').write_text(json.dumps({
        'training_rows': len(X), 'defective_rows': int((y == POSITIVE_CLASS).sum()),
        'features': list(X.columns), 'random_state': 42,
        'outer_folds': 5, 'selection_folds': 3, 'calibration_folds': 3,
        'model_version': MODEL_VERSION, 'final_test_access': False,
        'calibration_selection': 'inner OOF minimum Brier',
        'confidence_selection': 'inner OOF: HIGH >=90%, LOW <=5% defects; >=50 rows, >=20 groups',
        'production_model_or_policy_promoted': False,
    }, indent=2))
    return results
