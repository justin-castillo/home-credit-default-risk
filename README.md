# Home Credit Default Risk Prediction

## Overview

This project builds a complete machine learning pipeline to predict the probability that a loan applicant will experience payment difficulties. It reflects a real-world credit scoring system using structured, relational data and advanced model interpretability tools. Both a sample dataset and the full Home Credit dataset were used to validate modeling decisions, engineering techniques, and generalization.

## Objective

To accurately identify high-risk borrowers for Home Credit by building an explainable model optimized for recall. In lending, false negatives (misclassifying a defaulter as low-risk) are more costly than false positives, so the focus is on catching as many risky clients as possible.

## Dataset Summary

**Source:** [Kaggle – Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)  
**Target Variable:** `TARGET`  
- `1` = client defaulted  
- `0` = client paid on time

| File | Description |
|------|-------------|
| `application_train.csv`, `application_test.csv` | Static client info (one row per loan) |
| `bureau.csv`, `bureau_balance.csv` | Credit history from other institutions |
| `previous_application.csv` | Past loan applications |
| `POS_CASH_balance.csv` | Point-of-sale & cash loan balances |
| `credit_card_balance.csv` | Credit card activity |
| `installments_payments.csv` | Installment-level repayment behavior |
| `HomeCredit_columns_description.csv` | Glossary for all variables |

## Project Structure

```text
home_credit_default_risk/
│
├── data/                     # Raw and processed CSVs
├── notebooks/                # Jupyter notebooks (EDA, modeling)
├── reports/                  # Visualizations, SHAP plots, and documentation
├── environment.yml           # Conda environment specification
└── README.md                 # Project overview and instructions
```

## Pipeline Overview

### EDA + Schema Understanding
- Explored distributions, missingness patterns, and early risk signals
- Identified strong class imbalance: ~92% non-default vs. 8% default

### Relational Aggregation + Feature Engineering
- Aggregated all 7 auxiliary tables using `SK_ID_CURR` and `SK_ID_PREV`
- Created ~380 final features including:
  - Ratio metrics (e.g., credit-to-income)
  - Log-transformed financial values
  - Repayment delays and approval rates

### Modeling (CatBoost Focused)
- Evaluated baseline models on a sample subset
- Scaled to full dataset using `CatBoostClassifier` with:
  - Native categorical variable support
  - Hyperparameter tuning via `Optuna`
  - Threshold optimization to maximize F1
  - Stratified K-Fold cross-validation (k=5)

### Interpretability
- Global feature importance via SHAP summary plots
- Local interpretability via SHAP force plots
- Grouped SHAP analysis by domain (application, bureau, etc.)

### Final Model
- Retrained CatBoost model on full dataset with optimal parameters
- Exported test set predictions with calibrated probabilities
- Visualized confusion matrix and prediction probability distributions

## Key Results

### Sample Dataset (Baseline Comparison)

| Model               | AUC   | Recall | Precision | F1 Score |
|--------------------|-------|--------|-----------|----------|
| CatBoost           | 0.779 | 0.583  | 0.327     | 0.415    |
| LightGBM           | 0.776 | 0.546  | 0.349     | 0.425    |
| XGBoost            | 0.774 | 0.512  | 0.321     | 0.395    |
| Random Forest      | 0.767 | 0.528  | 0.312     | 0.392    |
| Logistic Regression| 0.772 | 0.511  | 0.324     | 0.396    |

### Full Dataset (Final CatBoost Model)

| Metric      | Value     |
|-------------|-----------|
| AUC         | 0.778     |
| Recall      | 0.683     |
| Precision   | 0.328     |
| F1 Score    | 0.446     |
| Threshold   | 0.15      |

> Recall improved significantly when scaling up to the full dataset with enriched feature engineering and threshold tuning.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate homecredit-env

## Future Enhancements

- Add temporal validation to test for model drift  
- Build a model API using FastAPI or Streamlit  
- Explore ensemble methods (e.g., CatBoost + TabNet)  
- Improve domain grouping and SHAP visualization in dashboards

## Author

Justin Castillo  
Data Science Portfolio Project – July 2025