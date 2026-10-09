# DefectRisk model card

**DefectRisk predicts risk and ranking, not certainty.**

**The model should support human review prioritization, not replace testing, code review, or engineering judgment.**

Status: model exploration complete; selected RF configuration frozen; sigmoid-calibrated training artifact available. The calibrated/abstaining system has **training-only validation**, not a new independent final evaluation. Model version: **`defectrisk-rf-sigmoid-v1`**; package version: `0.1.0`. Finalized 2026-10-09.

## Problem and intended use

Rank software modules by estimated defect risk when inspection/testing capacity is limited. The product question is how many known defective modules are flagged when approximately 30% of modules can be reviewed. Precision measures unnecessary review; recall measures missed recorded defects; Average Precision (AP) summarizes ranking across thresholds. Accuracy is secondary.

Intended use is research and a reproducible engineering case study, with risk-informed review/testing prioritization on compatible metrics. Production use requires validation on contemporary, independent, prediction-time-aligned data. A module's score is not proof that it contains a bug, and an unflagged module is not certified clean.

Non-intended use: automatic release approval, skipping testing, proving correctness, causal explanations, individual developer performance scoring, or claiming applicability across languages/projects without validation.

## Dataset, features and target

JM1, fixed OpenML dataset **1053, version 1**, describes static module metrics from a historical C system. Source: [OpenML](https://www.openml.org/d/1053). The locally audited dataset has **10,885 rows**, 21 predictors and target `defects`: `true` is defective, `false` clean. Full-data counts recorded during the earlier audit are 2,106 defective / 8,779 clean (~19.35% / 80.65%). M3 training contains **8,708 rows: 1,685 defective, 7,023 clean**. The archived held-out cohort has 2,177 rows. No final-test data was inspected again for finalization.

Ordered original feature schema:

```text
loc, v(g), ev(g), iv(g), n, v, l, d, i, e, b, t, lOCode, lOComment, lOBlank, locCodeAndComment, uniq_Op, uniq_Opnd, total_Op, total_Opnd, branchCount
```

These are size, complexity, line composition, operator/operand and Halstead metrics. The target and row index never enter predictors or feature-group keys. All original features remain; no engineered features, pruning, deduplication, relabeling or resampling are promoted. Numeric inputs may contain missing values; the schema rejects missing/extra/reordered columns, nonnumeric values and infinities. Training dtypes and preprocessing settings are recorded in [artifact metadata](../artifacts/rf-sigmoid-v1/metadata.json).

`b` remains a static complexity-derived estimate, not treated as the observed label. The training audit found 8,700/8,708 values compatible with `v/3000` rounding; formula exceptions and unknown extraction/snapshot timing prevent certifying all provenance. Prediction-time availability must be verified for new deployments; see [feature-quality study](feature-engineering.md).

## Imbalance, duplicates and conflicting labels

A constant clean decision would achieve ~80.65% full-data accuracy while detecting zero defects. Class imbalance is handled by **`class_weight="balanced_subsample"`**, computed within the fitting/bootstrap data; calibration is unweighted to recover the natural training prevalence.

The archived full-data audit found **1,973 duplicate full rows beyond the first**, **2,061 duplicate feature rows beyond the first**, and **8,824 unique feature vectors**. Of 669 duplicate-feature groups, 581 have consistent labels and **88 have conflicting labels**. The training-only audit found **11 conflicting groups containing 22 rows**. These are different populations, not contradictory counts. Exact conflicts alone account for a minimum of 11 deterministic row classification errors in training, not the hundreds of misses observed. Broader label ambiguity remains possible but is not quantitatively established.

Initial random splitting shared 299 feature vectors across train/test and 227 across fit/validation. The replacement deterministic group-aware split keeps every identical complete original vector, including conflicting labels, on one side. Labels help balance partition sizes, but are excluded from group identity. All duplicates remain. Missing-value matches count as identical.

## Exact selected model and calibration

The selected forest has **200 trees, depth 8, leaf minimum 3, sqrt feature sampling, balanced_subsample weights, seed 42**. Every classifier parameter in the current recorded environment is shown below:

```python
RandomForestClassifier(
    bootstrap=True,
    ccp_alpha=0.0,
    class_weight='balanced_subsample',
    criterion='gini',
    max_depth=8,
    max_features='sqrt',
    max_leaf_nodes=None,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=3,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    monotonic_cst=None,
    n_estimators=200,
    n_jobs=None,
    oob_score=False,
    random_state=42,
    verbose=0,
    warm_start=False,
)
```

Preprocessing remains the trusted pipeline: `SimpleImputer(strategy="median")` → `StandardScaler()` → RF. The scaler is retained to preserve the frozen pipeline. Each evaluation fold fits its own imputer/scaler/forest; the final artifact fits them on the complete M3 training partition.

Sigmoid calibration was chosen using leakage-safe training/CV evidence. It is an **unweighted, unpenalized logistic map of raw scalar RF probability**, Platt-style rather than sklearn's smoothed-target Platt implementation: `LogisticRegression(C=inf, solver="lbfgs", max_iter=5000, random_state=42, class_weight=None)`. This regression is the already-tested probability map, not a new classifier family. Calibration labels correspond only to group-held-out RF scores.

For the final artifact, three `StratifiedGroupKFold(shuffle=True, random_state=42)` training-only folds generate raw OOF calibration scores; the sigmoid map fits those scores, then one unchanged RF pipeline refits all M3 training rows. Its fitted map is `sigmoid(4.862592413639 × raw_probability -3.563784158608)`. This positive-slope map preserves ranking within the final model. Its fitted parameters are **not** a new evaluation result.

## Evaluation protocol and CV evidence

Development used the same five stratified group-aware outer folds on M3 training only. Limited tree searches used three inner group-aware folds to separate hyperparameter selection from outer performance estimation. Calibration adds group-held-out score generation within each fitting partition; inner Brier chooses the method, and outer validation estimates the selected procedure. All 80 calibration-study boundaries had zero shared original feature vectors. Confidence policies selected inside outer training were evaluated on untouched outer validation rows.

At approximately 30% review, pooled training OOF evidence:

| Procedure | Recall | Precision | F1 | AP | Known defects caught |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced LR (fixed configuration) | 0.5377 | 0.3469 | 0.4217 | 0.3929 | 906/1,685 |
| RF (nested parameter selection) | 0.5656 | 0.3649 | 0.4436 | 0.4022 | 953/1,685 |
| HGB (nested parameter selection) | 0.5608 | 0.3618 | 0.4398 | 0.4051 | 945/1,685 |
| XGBoost (limited nested selection) | 0.5496 | 0.3545 | 0.4310 | 0.4024 | 926/1,685 |

These rows compare selection procedures, not four fitted production artifacts. The calibration study evaluates the **single already-selected RF configuration**: raw Brier **0.18692** → sigmoid **0.13863**; AP **0.40703** → **0.40619**. Sigmoid was selected in every inner comparison. At ~30% review, fixed RF raw/sigmoid OOF recall was 0.5573/0.5591 (939/942 defects). Different fold-specific calibration maps slightly reorder pooled scores; this is not new feature information. Isotonic was statistically supported enough to compare but did not improve selection evidence. See [calibration and uncertainty](calibration-and-uncertainty.md) for all results, reliability bins and SD.

Repeated development on the same training population remains exploratory. Five overlapping-training folds do not supply independent replications or proof of statistical superiority; group protection does not establish cross-project or future-version generalization.

## Historical once-only held-out result: raw frozen RF

**This held-out test was evaluated once after model selection was frozen.**

| Population | Flagged for review | Recall | Precision | F1 | AP |
| --- | --- | ---: | ---: | ---: | ---: |
| 2,177 modules; 421 defective | 652 (29.95%) | 0.7126 | 0.4601 | 0.5592 | 0.6553 |

TP=300, FP=352, TN=1,404, FN=121: 300 known defects caught and 121 missed. This archived result was better than CV expectation, but one within-JM1 holdout cannot establish deployment performance. “Caught” means a recorded defective module was flagged, not that human reviewers were experimentally shown to discover every underlying bug.

**This result belongs to the uncalibrated frozen RF. It is not independent final validation of the subsequently calibrated/abstaining system.** No historical prediction files were loaded, no test evaluation was rerun, and no settings were changed using this held-out result. See [historical evaluation](final-evaluation.md). The new system requires an **external independent holdout evaluated once after its model, calibration and policy are frozen**.

## Uncertainty and product policy

JM1 does not support broad high-confidence automatic classification. An exploratory ≥90%-precision sigmoid HIGH tail contained only four rows (0.046% coverage), selected post hoc from evaluation outcomes. Prespecified threshold 0.70 achieved 81.82% HIGH precision on 11 rows. Thresholds 0.90/0.95 produced no HIGH cases; empty precision is undefined.

The supported inner-selected policy targeting ≥90% HIGH precision and ≤5% LOW defect rate (minimum 50 rows / 20 feature groups) automatically classified **136 LOW, 0 HIGH, 8,572 UNCERTAIN**: **1.56% coverage, 98.44% uncertainty**. Seven LOW rows were defective (5.15% LOW/automatic error); even the empirical inner LOW constraint did not strictly hold on outer data. No ≥90%-precision automation promise is enabled.

The model response contains calibrated probability, model name/version and calibration-method metadata. A separate deterministic policy returns LOW/HIGH/UNCERTAIN with recommended action. Primary positioning is **risk ranking / review prioritization**. A ~30% score-only capacity policy preserves ties; it is not a claim of confident binary classification. HIGH inspection plus UNCERTAIN review can exceed a review budget, so abstention fraction must not be confused with total human workload.

## Limitations and ethical / operational caveats

Static metrics omit churn, ownership, history, test coverage and engineering context. Correlated features and formula exceptions limit individual feature interpretations. Exact conflicting labels demonstrate insufficient information for some cases but do not prove the dominant source of error. Snapshot timing, label horizon and measurement quality remain incompletely verified. An information ceiling is a plausible inference from plateaus, not a measured theorem.

Calibration is conditional on JM1 prevalence and feature distribution; it does not guarantee individual outcomes, causal effects, or out-of-distribution confidence. New project/language/version distributions require fresh validation. Do not use low scores to waive standard checks or compare developer worth. Review capacity, false-negative costs, severe-defect handling and accountability need engineering owners. Label ascertainment may itself reflect which modules received investigation; prioritization can reinforce such feedback unless outcomes are monitored.

## Artifact, metadata and reproducibility

The training-only artifact is [model.joblib](../artifacts/rf-sigmoid-v1/model.joblib), accompanied by [metadata/schema](../artifacts/rf-sigmoid-v1/metadata.json), [checksums](../artifacts/rf-sigmoid-v1/checksums.json) and [calibration boundary audits](../artifacts/rf-sigmoid-v1/calibration-audits.csv). Artifact SHA-256: `b076c27882d9b12c235a6eb3dffd84b62e5c51b47d2d796132ecd0fe735e40ff`. Loader checks bytes, schema, preprocessing, RF/calibration settings and recorded dependency versions. Checksums detect corruption; they are not authenticity signatures. Deserialize only trusted local Joblib artifacts.

Training rows are selected using the already stored M3 training OOF row IDs, not by reconstructing or revisiting the historical test allocation. Labels must match that training snapshot. Model/OOF/training fingerprints, ordered numeric feature schema and positive-class metadata are recorded. Seeds are 42; inference rejects mismatched schema. Reproducibility is conditional on recorded versions/platform; a seed alone does not guarantee identical behavior across library releases.

Recorded environment: Python 3.14.8, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, joblib 1.6.0. [Pinned reproduction dependencies](../requirements-repro.txt).

```sh
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps
# Training only; use a NEW directory (existing artifacts cannot be overwritten).
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m defectrisk.final_artifact train --output artifacts/reproduction
# Evaluate SAVED training-only OOF evidence: no fit, no test predictions.
python -m defectrisk.final_artifact evaluate
# Regenerate all figures from saved training evidence + archived aggregate facts.
python -m defectrisk.portfolio_evidence
python -m pytest -q
```

The evaluation command recalculates existing training-CV metrics; it does not evaluate the newly full-trained artifact on its own fitting data or provide new independent validation. The historical evaluation is intentionally absent from the reproduction instructions.

Finalization verification: **212 passed in 24.95 seconds** with the complete test suite. Artifact determinism, calibrated configuration, model-card/configuration agreement, checksum/schema rejection, locked training IDs, and stored-evidence provenance are tested. No historical test evaluation was rerun.
