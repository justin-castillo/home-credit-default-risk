# Exploratory Data Analysis (EDA) – Home Credit Default Risk

## Objective

This analysis explores the `application_train.csv` dataset to understand the class distribution, identify potential predictive signals, and inform feature engineering. The EDA also supports modeling strategy decisions by evaluating data quality, skew, and feature relationships.

---

## 1. Dataset Overview

- Total rows: ~307,511
- Total columns: 122
- Target variable: `TARGET` (binary classification)
  - `0` = no default
  - `1` = payment difficulty (default)

---

## 2. Class Imbalance

Roughly **8%** of applicants experienced payment difficulties.

**Recommended Plot:**  
- `TARGET` distribution bar plot (e.g., `sns.countplot`)
- Annotated class imbalance ratio

---

## 3. Missing Values

Significant missingness found in:
- `OWN_CAR_AGE`
- `EXT_SOURCE_1`
- Building metadata: `*_AVG`, `*_MODE`, `*_MEDI`
- Credit bureau request counts (`AMT_REQ_CREDIT_BUREAU_*`)

**Action Taken:**
- Imputed using `SimpleImputer` (mean or median)
- Dropped high-missingness building metadata features if low variance or importance

**Recommended Plot:**  
- Horizontal bar plot: Top 20 columns by percent missing

---

## 4. Numerical Feature Distributions

Key observations:
- `AMT_INCOME_TOTAL` and `AMT_CREDIT` show heavy right skew
- Most clients are aged 25–65 (via `DAYS_BIRTH`)
- Skewed variables like income and credit were log-transformed

**Recommended Plots:**
- Histograms: `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `DAYS_BIRTH`
- KDE plots: `EXT_SOURCE_1/2/3` split by `TARGET`

---

## 5. Categorical Variable Insights

Variables with clear class separation:
- `NAME_CONTRACT_TYPE`: Cash loans correlate with higher default risk
- `NAME_FAMILY_STATUS`: Singles tend to default more
- `NAME_EDUCATION_TYPE`: Lower education correlates with higher default risk

**Recommended Plots:**
- Count plots of key categorical features split by `TARGET`
- Target rate by category (e.g., stacked bars or hue-colored bars)

---

## 6. Correlation Analysis

Top correlations with `TARGET` (absolute):
- `EXT_SOURCE_2`: -0.16
- `EXT_SOURCE_3`: -0.10
- `DAYS_BIRTH`: -0.08
- `DAYS_EMPLOYED`: -0.07

**Interpretation:**  
Lower external scores and younger applicants have higher default rates.

**Recommended Plot:**
- Correlation heatmap (filtered to top 30 features)
- Boxplots of continuous variables vs. `TARGET`

---

## 7. Bivariate Relationships

**Confirmed via Boxplots:**
- Defaulting clients are generally younger
- Have lower EXT_SOURCE scores
- Often lower in income but not always

**Recommended Plots:**
- Boxplots: `DAYS_BIRTH`, `EXT_SOURCE_2`, `CREDIT_TO_INCOME_RATIO` vs. `TARGET`
- Violin or strip plots for SHAP-relevant features

---

## 8. Feature Engineering Candidates

Proposed and implemented features based on EDA:

| Feature Name | Description |
|--------------|-------------|
| `CREDIT_TO_INCOME_RATIO` | AMT_CREDIT / AMT_INCOME_TOTAL |
| `ANNUITY_TO_INCOME_RATIO` | AMT_ANNUITY / AMT_INCOME_TOTAL |
| `EXT_SOURCES_MEAN` | Mean of EXT_SOURCE_1, 2, 3 |
| `DAYS_EMPLOYED_PERC` | DAYS_EMPLOYED / DAYS_BIRTH |
| `INCOME_PER_FAM_MEMBER` | AMT_INCOME_TOTAL / CNT_FAM_MEMBERS |

These were retained and validated via SHAP and F1-score improvements during model training.

---

## 9. Summary of EDA Findings

- **Target imbalance** is substantial and shaped modeling priorities
- **External risk scores** are the most predictive continuous features
- **Categorical variables** (contract type, education, family status) show clear stratification
- **Outliers and skew** addressed via log transforms and caps
- **High-missing features** handled with imputation or removal
- EDA directly informed ratio creation, SHAP grouping, and data prep for CatBoost

---

## 10. Next Step

Leverage these insights in a unified feature engineering pipeline, merge relational tables, and evaluate performance impacts through stratified CV and SHAP-based model introspection.
