# Lab 04: Data Pipeline and Quality

## Problem Definition
- **Dataset:** UCI Adult (OpenML v2)
- **Inputs:** 14 mixed numerical and categorical features (demographics, employment).
- **Output:** Binary classification (income >50K).
- **Success Metric:** F1 Score (primary) and Accuracy (secondary).
- **Constraints:** No data leakage (train/test split before imputation), fully reproducible pipeline.

## Execution Summary & Validation
- **Rows/Columns:** 48,842 rows, 15 columns.
- **Schema Validation:** 
  - Missing values identified: `workclass` (2799), `occupation` (2809), `native-country` (857). `?` markers were correctly converted to `NaN`.
  - Range violations: 0 (Age 17-90, Hours 1-99, Gain 0-99999, Loss 0-4356).
- **Label Quality:** Audit found **5** duplicate feature rows with conflicting income labels.

## Results & Feature Engineering
- **Baseline Logistic Regression F1:** 0.6604
- **Engineered Features F1:** 0.6693
- **Impact:** Engineering `capital_net` and `hours_per_age` improved the F1 score by **+0.0089**. This shows that capturing the net capital (rather than treating gain and loss separately) and age-normalized working hours added predictive value.
- **Leakage Prevention:** Duplicate rows were dropped from the training set *only* (train shape reduced, test set untouched). Imputation was fitted strictly on `X_train`.

## Reproducibility
- **Dataset Hash (SHA256):** [8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc]
- **Artifacts saved:** 
  - `results/preprocessor.joblib` (Fitted Pipeline)
  - `results/config.yaml` (Pipeline configuration)
  - `results/data_card.md` (Source, biases, privacy)
- **One-Command Execution:** 
  ```bash
  python pipeline.py