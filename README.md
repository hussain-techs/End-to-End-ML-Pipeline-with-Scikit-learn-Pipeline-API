# Customer Churn Prediction — End-to-End ML Pipeline

A reusable, production-ready machine learning pipeline for predicting telecom customer churn, built with scikit-learn's `Pipeline` and `ColumnTransformer` APIs.

## Objective

Build a pipeline that takes raw customer data and outputs a churn prediction (Yes/No) plus a churn probability, with all preprocessing steps (imputation, scaling, encoding) bundled together with the model so it can be deployed as a single artifact.

## Dataset

A synthetic Telco-style dataset (2,000 rows, 16 features) generated to mirror the structure of the public [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): demographics, account info (tenure, contract type, payment method), service subscriptions (internet, phone, streaming, support), and billing (monthly/total charges). ~2% of `TotalCharges` values are missing to simulate real-world data quality issues.

| | |
|---|---|
| Rows | 2,000 |
| Features | 16 (3 numeric, 13 categorical) |
| Target | `Churn` (Yes/No) — ~20% positive class |
| Missing values | ~2% in `TotalCharges` |

## Project Structure

```
.
├── churn_pipeline.py            # Main script — run this
├── churn_pipeline_results.png   # 9-panel results dashboard
├── lr_churn_pipeline.joblib     # Exported Logistic Regression pipeline
├── rf_churn_pipeline.joblib     # Exported Random Forest pipeline
└── README.md                    # This file
```

## Pipeline Architecture

```
ColumnTransformer
├── Numeric branch  (tenure, MonthlyCharges, TotalCharges)
│     → SimpleImputer(median) → StandardScaler
└── Categorical branch (13 columns)
      → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
            ↓
      Classifier (LogisticRegression  |  RandomForestClassifier)
```

Both branches and the classifier are wrapped in a single `sklearn.Pipeline`, so the exported `.joblib` file is a self-contained object — raw input in, prediction out.

## Hyperparameter Tuning

`GridSearchCV` with 5-fold `StratifiedKFold`, scored on ROC-AUC.

**Logistic Regression grid**
```python
{
  "classifier__C":       [0.01, 0.1, 1, 10],
  "classifier__penalty": ["l1", "l2"],
  "classifier__solver":  ["liblinear"],
}
```

**Random Forest grid**
```python
{
  "classifier__n_estimators":      [100, 200],
  "classifier__max_depth":         [None, 10, 20],
  "classifier__min_samples_split": [2, 5],
}
```

## Results

| Model | Best CV AUC | Test Accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.7205 | 0.80 | **0.737** |
| Random Forest | 0.6975 | 0.79 | 0.696 |

**Best Logistic Regression params:** `C=0.1`, `penalty=l1`, `solver=liblinear`
**Best Random Forest params:** `max_depth=10`, `min_samples_split=2`, `n_estimators=200`

Logistic Regression edged out Random Forest on this dataset — a reminder that simpler linear models can outperform ensembles on smaller, mostly-categorical tabular data.

See `churn_pipeline_results.png` for ROC curves, confusion matrices, feature importances, and GridSearch heatmaps.

## How to Run

```bash
pip install scikit-learn pandas numpy joblib matplotlib seaborn
python churn_pipeline.py
```

The script is self-contained (it generates its own data), so no external CSV is required. It will print metrics to the console and write the PNG dashboard and two `.joblib` files to the working directory.

## How to Reuse the Exported Pipeline

```python
import joblib
import pandas as pd

model = joblib.load("rf_churn_pipeline.joblib")  # or lr_churn_pipeline.joblib

# new_data must have the same 16 raw columns as training (no preprocessing needed)
predictions   = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]
```

Preprocessing (imputation, scaling, one-hot encoding) is baked into the pipeline, so raw data can be passed directly.

## Skills Demonstrated

- ML pipeline construction with `Pipeline` + `ColumnTransformer`
- Hyperparameter tuning with `GridSearchCV`
- Model export and reusability via `joblib`
- Production-readiness practices (single deployable artifact, reproducible splits, stratified CV)
