# DefectRisk

DefectRisk ranks software modules by estimated defect risk using static code metrics. It helps engineers prioritize human review and testing when review capacity is limited.

> **Review ~30% of modules → capture 71.26% of known defects**
>
> Historical once-only evaluation of the frozen raw RF: 652/2,177 modules flagged, 300/421 known defects captured. This is not final validation of the later calibrated system.

**How it works:** static metrics → calibrated RF → risk ranking → human review

## Try it

```sh
python -m pip install -e .
defectrisk rank examples/modules.csv
```

```text
RANK  MODULE          RISK
1     module_42       0.4696
2     module_17       0.3743
3     module_missing  0.2008
4     module_03       0.1039
```

Real frozen-artifact inference on synthetic examples. Scores are estimated defect risk, not certainty; they do not certify modules clean or defective.

DefectRisk ranks risk rather than classifying at a 0.50 cutoff. With JM1's ~19% defect prevalence, a ~47% calibrated risk is substantially above baseline and useful for prioritization; ranking position is the primary operational signal.

**Technical credibility:**

- Leakage-safe evaluation with identical feature vectors kept together.
- Group-aware cross-validation.
- Class imbalance handled within fitting partitions.
- Controlled model comparison and nested tuning.
- Calibration selected from training-only CV evidence.

[Full technical case](docs/portfolio-case.md) · [Model card](docs/model-card.md) · [Data-quality article](docs/article-data-quality.md)

---

## CLI input and output

Run from the repository root with the artifact's recorded dependencies (see reproduction below):

```sh
defectrisk rank examples/modules.csv --format json
defectrisk rank examples/modules.csv --artifact artifacts/rf-sigmoid-v1 --format table
```

Input requires the exact ordered 21 original feature columns listed in the [model card](docs/model-card.md). One optional `module`, `module_id` or `id` column is excluded from inference; custom names use `--id-column NAME`. Without an identifier, rows receive `row_1`, etc. Blank numeric cells, `NA`, `N/A` and `NaN` use the frozen median imputer. Invalid/extra/reordered columns, nonnumeric values, infinities and empty input are rejected.

JSON includes rank, module, calibrated risk probability, model version and calibration method. Results are sorted by descending probability, keeping input order for ties. Table probabilities have four decimal places; JSON retains full precision. The risk note is printed to stderr so stdout remains valid JSON. Inference verifies the existing artifact's checksums/schema, performs no training and leaves the artifact unchanged.

## Historical result and product boundary

![Historical once-only held-out result](docs/figures/hero-held-out.svg)

The frozen raw RF's historical precision was **46.01%**, F1 **0.5592**, AP **0.6553**; **121 defects were missed**. This held-out test was evaluated once after model selection was frozen and was not reused for calibration or policy design.

**Broader high-confidence automatic classification was not supported by the available static metrics, so the system is positioned as risk prioritization for human review.**

## What this case demonstrates

- **Evaluation integrity:** feature-only groups protect train/test, validation and CV boundaries; exact overlap audits reject leakage. Conflicting-label duplicates remain together, without removing rows.
- **Controlled comparisons:** shared folds, training-only imbalance handling, nested limited tuning and an explicit recall-at-review-capacity objective.
- **Honest uncertainty:** sigmoid calibration improved Brier **0.18692 → 0.13863** with AP roughly stable. Supported high-confidence abstention left **98.44% UNCERTAIN**, so automated defect-oracle claims are disabled.
- **Engineering reproducibility:** frozen configuration, ordered feature schema, model/version metadata, training fingerprints, dependency pins and artifact checksums.
- **Stopping judgment:** HGB/XGBoost, feature engineering and ensembles produced diminishing returns. Model exploration is complete; better prediction-time data is the future recommendation.

## Final technical position

Selected model: **tuned weighted Random Forest**, with `n_estimators=200`, `max_depth=8`, `min_samples_leaf=3`, `max_features="sqrt"`, `class_weight="balanced_subsample"`, `random_state=42`.

Pipeline: median imputer → StandardScaler → RF → unweighted sigmoid/Platt-style probability calibration. Calibration was selected using nested group-aware **training-only** CV. Probability/model metadata and deterministic review policy are separate.

**DefectRisk predicts risk and ranking, not certainty.** The model should support human review prioritization, not replace testing, code review, or engineering judgment.

## Evidence at approximately 30% review capacity

| Training-only OOF procedure | Recall | Precision | F1 | AP | Known defects flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced Logistic Regression | 53.77% | 34.69% | 0.4217 | 0.3929 | 906/1,685 |
| Tuned weighted Random Forest | **56.56%** | **36.49%** | **0.4436** | 0.4022 | **953/1,685** |
| Tuned weighted HGB | 56.08% | 36.18% | 0.4398 | **0.4051** | 945/1,685 |
| XGBoost (limited nested search) | 54.96% | 35.45% | 0.4310 | 0.4024 | 926/1,685 |

Same five stratified group-aware outer folds; tree parameters selected in three inner folds, LR fixed. These estimate selection procedures, not the final full-training artifact. Calibration separately evaluated the single frozen RF; its sigmoid OOF recall was **55.91%** at ~30% review. Fold SDs are descriptive, not significance tests.

![Training-only model comparison](docs/figures/model-comparison-cv.svg)

[Precision–recall](docs/figures/precision-recall-cv.svg) · [Review effort vs recall](docs/figures/review-effort-recall-cv.svg) · [Reliability](docs/figures/calibration-reliability-cv.svg) · [Historical outcomes](docs/figures/historical-confusion-matrix.svg)

## Data and integrity

JM1 / [OpenML 1053, version 1](https://www.openml.org/d/1053): **10,885 modules**, 21 original static metrics, target `defects` (`true` = defective). The M3 training partition has **8,708 modules**, including 1,685 defective. Archived audit: 2,061 duplicate feature rows beyond the first and 88 full-data conflicting-label groups. Initial random splits shared 299 train/test vectors; deterministic group splitting eliminated this contamination.

All repeated vectors—including conflicting labels—stay on the same side of each boundary. Groups exclude the target and use original metrics even in feature-engineering experiments. Preprocessing/calibration/weights learn only from fitting partitions. No duplicate removal or relabeling is promoted.

## Reproduce the selected artifact and stored evidence

Use a fresh Python environment (recorded runtime **Python 3.14.8**, Linux; project supports Python ≥3.12). Install the exact recorded dependencies for comparable reproduction:

```sh
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps
python -m pytest -q

# Training only: locked M3 row IDs; new destination required.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m defectrisk.final_artifact train --output artifacts/reproduction

# Recalculate saved training-CV evidence: no model fits or test predictions.
python -m defectrisk.final_artifact evaluate

# Six figures, each in SVG/PNG/PDF, from saved evidence only.
python -m defectrisk.portfolio_evidence
```

The complete suite includes cached-JM1 integration tests and synthetic tests of historical evaluation guards; it does not re-evaluate the real final test. The first training/data load may need internet for OpenML. Training selects only established M3 training row IDs; the historical split is not reconstructed for artifact training.

The delivered [artifact](artifacts/rf-sigmoid-v1/model.joblib) has [metadata/schema](artifacts/rf-sigmoid-v1/metadata.json), [SHA-256 checksums](artifacts/rf-sigmoid-v1/checksums.json) and [calibration audits](artifacts/rf-sigmoid-v1/calibration-audits.csv). Load only trusted local artifacts; Joblib deserialization can execute code, and checksums are not authenticity signatures. The loader checks recorded dependency versions and schema before use. Seeds do not guarantee reproducibility across different library versions/platforms.

Verification: **239 tests passed** in the complete suite (212 at portfolio finalization). Real CLI inference uses the unchanged, verified frozen artifact; inference tests fail on any attempted fit or data-loader access. The artifact was trained on M3 training rows only.

## Limits and evidence trail

Static metrics omit process/history/context, and JM1 label/snapshot provenance remains imperfect. Calibration is not individual certainty, causality or confidence under distribution shift. **A new independent external holdout is required for once-only final validation of the calibrated/policy system. The historical test must not be reused.**

Detailed evidence: [integrity fix](docs/evaluation-integrity-fix.md), [class weighting](docs/class-weight-experiment.md), [thresholds](docs/threshold-experiment.md), [nested tree tuning](docs/tree-model-tuning.md), [feature plateau](docs/feature-engineering.md), [XGBoost challenge](docs/final-model-challenge.md), [ensemble plateau](docs/ensemble-ranking.md), [calibration/uncertainty](docs/calibration-and-uncertainty.md), and [historical once-only evaluation](docs/final-evaluation.md). Figure sources and hashes are recorded in [evidence.json](docs/figures/evidence.json).
