# Problem definition

## 1. Unit of observation

One software module.

## 2. Features (X)

Static software metrics such as complexity, size, and related code metrics.

## 3. Target (y)

A binary defect label: defective / clean.

## 4. ML task

Supervised binary classification.

## 5. Product interpretation

The model is intended to help engineering teams prioritize modules for additional
review or testing before release.

## 6. Error interpretation

- **False positive:** a clean module flagged as defective.
- **False negative:** a defective module predicted as clean.

False negatives are provisionally considered more costly because defects may
reach production. This working assumption will later influence metric selection;
it is not unquestionable and should be revisited as project context and evidence
become available.

## 7. Non-goals for M1

- No model.
- No train/test split.
- No preprocessing.
- No cross-validation.
- No GridSearchCV.
- No API.
- No deployment.
