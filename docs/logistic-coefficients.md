# Balanced Logistic Regression: learned coefficients

The fixed current candidate was fitted **once on all 8,708 M3 training rows**,
using the unchanged group-aware final split and pipeline:

```python
SimpleImputer(strategy="median")
StandardScaler()
LogisticRegression(random_state=42, max_iter=5000, class_weight="balanced", C=1.0)
```

No tuning, prediction, scoring, or final test inspection/evaluation was performed.
Final test outputs were immediately discarded. This full-training fit describes
the learned model parameters; it does not produce a new performance estimate or
promote a model. The earlier C-grid candidate is not used.

## Learned model

**Intercept/bias: −0.108591198**, in standardized feature space.

For each feature, first replace missing values with its training median, then
standardize using its learned training mean and standard deviation. Writing
these standardized values as `z`, the fitted model is:

```text
log_odds(defective) = -0.108591198 + sum(coefficient[j] * z[j])
P(defective) = 1 / (1 + exp(-log_odds(defective)))
```

The intercept is the model's log-odds at `z=0`, meaning all imputed features are
at their training means. Because the fit uses balanced class weights, it should
not be interpreted as the raw dataset's defective prevalence.

All **21 coefficients** are matched to their original feature names using the
fitted preprocessing feature names, checked against the input column order.
The fitted class order is verified as clean (`"false"`), defective (`"true"`),
so the signs below describe defective log-odds. Sorted by absolute magnitude:

| Feature | Coefficient | Absolute coefficient | Direction as the feature increases |
| --- | ---: | ---: | --- |
| n | 1.816471194 | 1.816471194 | Toward defective |
| loc | 1.409048581 | 1.409048581 | Toward defective |
| v | -1.261695159 | 1.261695159 | Toward clean |
| total_Opnd | -1.194517991 | 1.194517991 | Toward clean |
| branchCount | 0.978422001 | 0.978422001 | Toward defective |
| v(g) | -0.762273712 | 0.762273712 | Toward clean |
| lOCode | -0.741808538 | 0.741808538 | Toward clean |
| b | 0.620130938 | 0.620130938 | Toward defective |
| total_Op | -0.545243166 | 0.545243166 | Toward clean |
| uniq_Opnd | 0.441711456 | 0.441711456 | Toward defective |
| uniq_Op | 0.275589771 | 0.275589771 | Toward defective |
| d | -0.169970104 | 0.169970104 | Toward clean |
| i | 0.145417763 | 0.145417763 | Toward defective |
| lOComment | -0.131648327 | 0.131648327 | Toward clean |
| ev(g) | -0.106952843 | 0.106952843 | Toward clean |
| locCodeAndComment | 0.086043639 | 0.086043639 | Toward defective |
| l | -0.083741552 | 0.083741552 | Toward clean |
| lOBlank | 0.053592521 | 0.053592521 | Toward defective |
| e | -0.007166472 | 0.007166472 | Toward clean |
| t | -0.007114989 | 0.007114989 | Toward clean |
| iv(g) | 0.002538492 | 0.002538492 | Toward defective |

## Interpretation limits

Holding other standardized features fixed, increasing a feature raises
defective log-odds when its coefficient is positive and lowers them when it is
negative. A one-unit change in a standardized, nonconstant feature corresponds
to one training standard deviation and changes log-odds by its coefficient.
**StandardScaler makes coefficient magnitudes more comparable across features
than coefficients on their different raw measurement scales.**

**Coefficient importance is NOT causality.** These are regularized conditional
associations in this fitted model, not evidence that changing a code metric
causes or prevents a defect. **Correlated features can make individual
coefficients unstable**, redistribute apparent influence among related metrics,
or make an individual sign misleading in isolation. Scaling does not remove
correlation. No coefficient-stability experiment was performed here. See
[scikit-learn's coefficient-interpretation guidance](https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html).

## Run and verification

```sh
python -m defectrisk.logistic_coefficients
python -m pytest -q
```

Implementation: `src/defectrisk/logistic_coefficients.py`.
Full console output: [logistic-coefficients-results.txt](logistic-coefficients-results.txt).
The real cached JM1 fit used scikit-learn 1.9.1; no convergence warning was
observed. **The complete suite passed: 109 tests**, without exclusions.
New tests cover the full-training-only fit, fixed classifier settings,
training-only preprocessing, exact feature-to-coefficient mapping, magnitude
sorting, signs/zero direction, intercept extraction, reconstruction of the
linear score on synthetic data, input preservation, and no final test inspection.
No explainability dependency was added.

Execution used the existing packages and JM1 cache with the compatible temporary
Python 3.14 interpreter under `/tmp`, because the workspace virtual environment's
interpreter symlinks remain broken. No defaults were changed and no model was
persisted or promoted.
